"""Batting Performance ML Models with Time-Based Splits and Uncertainty Intervals."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.ml.preprocessing import MLDataPreprocessor
from src.ml.model_registry import ModelRegistry
from src.utils.logger import logger
from src.features.batting_features import BattingFeatureExtractor
from src.features.form_features import FormFeatureExtractor

BATTING_FEATURES = [
    "prior_bat_avg", "prior_strike_rate", "recent_form_5", "recent_sr_5", "innings"
]

class BattingModelTrainer:
    """Trains and evaluates predictive models for batter runs and strike rate."""

    @staticmethod
    def train_and_evaluate(time_split_ratio: float = 0.8) -> Dict[str, Any]:
        """
        Trains models using strict chronological time-based split.
        Never shuffles future matches into past training sets.
        """
        df = MLDataPreprocessor.prepare_batting_dataset()
        if len(df) < 50:
            raise ValueError("Insufficient data to train batting models.")

        # Temporal split: sort by date already done in preprocessor
        split_idx = int(len(df) * time_split_ratio)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        X_train = train_df[BATTING_FEATURES].fillna(0)
        y_train = train_df["bat_runs"]
        X_test = test_df[BATTING_FEATURES].fillna(0)
        y_test = test_df["bat_runs"]

        models = {
            "Ridge": Ridge(alpha=1.0),
            "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=80, max_depth=4, learning_rate=0.08, random_state=42)
        }

        eval_results = {}
        best_name = None
        best_rmse = float("inf")
        best_model = None

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae = float(mean_absolute_error(y_test, preds))
            rmse = float(root_mean_squared_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))

            eval_results[name] = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 3)}
            logger.info(f"Batting Model [{name}]: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_name = name
                best_model = model

        # Calculate residual standard deviation for prediction intervals
        train_preds = best_model.predict(X_train)
        residuals = y_train - train_preds
        residual_std = float(np.std(residuals))

        # Save best model to registry
        ModelRegistry.save_model(
            model_name="batting_runs_model",
            model_obj={
                "model": best_model,
                "model_type": best_name,
                "residual_std": residual_std,
                "features": BATTING_FEATURES
            },
            features=BATTING_FEATURES,
            target="bat_runs",
            metrics=eval_results[best_name]
        )

        return {
            "best_model": best_name,
            "comparison": eval_results,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "residual_std": round(residual_std, 2)
        }

    @staticmethod
    def predict_runs(
        player_name: str,
        innings: int = 1,
        pressure_factor: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generates point prediction and uncertainty intervals for upcoming match.
        """
        artifact = ModelRegistry.load_model("batting_runs_model")
        if not artifact:
            # Fallback heuristic if ML model not trained yet
            bf = BattingFeatureExtractor.extract_player_batting_features(player_name)
            avg = bf.get("batting_average", 25.0)
            return {
                "expected_runs": avg,
                "lower_bound": max(0, avg - 15),
                "upper_bound": avg + 20,
                "confidence_level": "80%",
                "model_used": "Baseline Historical Average"
            }

        model = artifact["model"]
        residual_std = artifact.get("residual_std", 15.0)

        # Extract features for player
        bf = BattingFeatureExtractor.extract_player_batting_features(player_name)
        form = FormFeatureExtractor.calculate_player_form(player_name, role="batting")

        prior_avg = bf.get("batting_average", 25.0)
        prior_sr = bf.get("strike_rate", 125.0)
        recent_form = form.get("recent_n_avg", prior_avg)
        recent_sr = prior_sr

        input_data = pd.DataFrame([{
            "prior_bat_avg": prior_avg,
            "prior_strike_rate": prior_sr,
            "recent_form_5": recent_form,
            "recent_sr_5": recent_sr,
            "innings": innings
        }])

        pred_point = float(model.predict(input_data[BATTING_FEATURES])[0]) * pressure_factor
        pred_point = max(2.0, round(pred_point, 1))

        # 80% prediction interval (~1.28 standard deviations)
        margin = round(1.28 * residual_std, 1)
        lower = max(0.0, round(pred_point - margin, 1))
        upper = round(pred_point + margin, 1)

        return {
            "player": player_name,
            "expected_runs": pred_point,
            "lower_bound": lower,
            "upper_bound": upper,
            "prediction_interval": f"{lower:.0f} - {upper:.0f} runs",
            "model_used": f"XGBoost/RF ({artifact.get('model_type')})",
            "input_features": input_data.to_dict(orient="records")[0]
        }
