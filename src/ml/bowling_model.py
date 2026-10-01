"""Bowling Performance ML Models with Time-Based Splits and Uncertainty Intervals."""

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
from src.features.bowling_features import BowlingFeatureExtractor
from src.features.form_features import FormFeatureExtractor

BOWLING_FEATURES = [
    "prior_econ", "recent_wkts_5", "recent_econ_5"
]

class BowlingModelTrainer:
    """Trains and evaluates predictive models for bowler wickets and economy."""

    @staticmethod
    def train_and_evaluate(time_split_ratio: float = 0.8) -> Dict[str, Any]:
        """Trains bowling models using time-based split."""
        df = MLDataPreprocessor.prepare_bowling_dataset()
        if len(df) < 50:
            raise ValueError("Insufficient data to train bowling models.")

        split_idx = int(len(df) * time_split_ratio)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        X_train = train_df[BOWLING_FEATURES].fillna(0)
        y_train = train_df["wickets"]
        X_test = test_df[BOWLING_FEATURES].fillna(0)
        y_test = test_df["wickets"]

        models = {
            "Ridge": Ridge(alpha=1.0),
            "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=70, max_depth=3, learning_rate=0.08, random_state=42)
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
            logger.info(f"Bowling Model [{name}]: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_name = name
                best_model = model

        # Residual std for uncertainty intervals
        residuals = y_train - best_model.predict(X_train)
        residual_std = float(np.std(residuals))

        ModelRegistry.save_model(
            model_name="bowling_wickets_model",
            model_obj={
                "model": best_model,
                "model_type": best_name,
                "residual_std": residual_std,
                "features": BOWLING_FEATURES
            },
            features=BOWLING_FEATURES,
            target="wickets",
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
    def predict_wickets(
        player_name: str,
        pressure_factor: float = 1.0
    ) -> Dict[str, Any]:
        """Predicts expected wickets and prediction range for upcoming match."""
        artifact = ModelRegistry.load_model("bowling_wickets_model")
        if not artifact:
            bw = BowlingFeatureExtractor.extract_player_bowling_features(player_name)
            avg_wkts = round(bw.get("wickets", 1) / max(1, bw.get("matches", 1)), 1)
            return {
                "expected_wickets": avg_wkts,
                "lower_bound": 0,
                "upper_bound": avg_wkts + 1.5,
                "model_used": "Baseline Historical Average"
            }

        model = artifact["model"]
        residual_std = artifact.get("residual_std", 0.9)

        bw = BowlingFeatureExtractor.extract_player_bowling_features(player_name)
        form = FormFeatureExtractor.calculate_player_form(player_name, role="bowling")

        prior_econ = bw.get("economy", 8.0)
        recent_wkts = form.get("recent_n_avg", 1.0)
        recent_econ = prior_econ

        input_data = pd.DataFrame([{
            "prior_econ": prior_econ,
            "recent_wkts_5": recent_wkts,
            "recent_econ_5": recent_econ
        }])

        pred_point = float(model.predict(input_data[BOWLING_FEATURES])[0]) * pressure_factor
        pred_point = max(0.1, round(pred_point, 1))

        margin = round(1.28 * residual_std, 1)
        lower = max(0.0, round(pred_point - margin, 1))
        upper = round(pred_point + margin, 1)

        return {
            "player": player_name,
            "expected_wickets": pred_point,
            "lower_bound": lower,
            "upper_bound": upper,
            "prediction_interval": f"{lower:.1f} - {upper:.1f} wickets",
            "model_used": f"XGBoost/RF ({artifact.get('model_type')})",
            "input_features": input_data.to_dict(orient="records")[0]
        }
