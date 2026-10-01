"""SHAP-Based Explainable AI Engine for Cricket Predictions."""

import shap
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from src.ml.model_registry import ModelRegistry
from src.ml.preprocessing import MLDataPreprocessor
from src.utils.logger import logger

class SHAPExplanationEngine:
    """Computes genuine SHAP values (TreeExplainer) for global and local prediction transparency."""

    @staticmethod
    def explain_batting_prediction(input_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes local SHAP attributions for a single batter run prediction.
        Breaks down contribution into positive and negative driving factors.
        """
        artifact = ModelRegistry.load_model("batting_runs_model")
        if not artifact:
            return {"error": "Model not trained yet."}

        model = artifact["model"]
        feature_names = artifact["features"]

        # Prepare 1-row DataFrame
        row_df = pd.DataFrame([input_features])[feature_names].fillna(0)

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(row_df)

            # Handle 1D / 2D shap output format
            if isinstance(shap_values, list):
                vals = shap_values[0][0]
            elif len(shap_values.shape) == 2:
                vals = shap_values[0]
            else:
                vals = shap_values

            base_value = float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[0])
            predicted_value = float(model.predict(row_df)[0])

            factors = []
            for name, val in zip(feature_names, vals):
                factors.append({
                    "feature": name.replace("_", " ").title(),
                    "impact": round(float(val), 2),
                    "raw_feature_name": name,
                    "direction": "Positive" if val >= 0 else "Negative"
                })

            factors_sorted = sorted(factors, key=lambda x: abs(x["impact"]), reverse=True)
            top_positive = [f for f in factors_sorted if f["impact"] > 0]
            top_negative = [f for f in factors_sorted if f["impact"] < 0]

            return {
                "base_value": round(base_value, 1),
                "predicted_value": round(predicted_value, 1),
                "factors": factors_sorted,
                "top_positive": top_positive,
                "top_negative": top_negative
            }
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {
                "error": str(e),
                "factors": []
            }

    @staticmethod
    def get_global_batting_importance() -> List[Dict[str, Any]]:
        """Calculates global mean absolute SHAP feature importance."""
        artifact = ModelRegistry.load_model("batting_runs_model")
        if not artifact:
            return []

        model = artifact["model"]
        feature_names = artifact["features"]

        df = MLDataPreprocessor.prepare_batting_dataset()
        if df.empty:
            return []

        sample_df = df[feature_names].fillna(0).head(150)
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(sample_df)
            mean_abs = np.mean(np.abs(shap_vals), axis=0)

            results = []
            for name, imp in zip(feature_names, mean_abs):
                results.append({
                    "feature": name.replace("_", " ").title(),
                    "importance": round(float(imp), 2)
                })
            return sorted(results, key=lambda x: x["importance"], reverse=True)
        except Exception as e:
            logger.warning(f"Global SHAP calculation failed: {e}")
            return []
