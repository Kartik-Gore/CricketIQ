"""Model Registry, Serialization, and Metadata Tracking."""

import json
from pathlib import Path
import joblib
from typing import Dict, Any, Optional
from src.config import MODELS_DIR
from src.utils.logger import logger

METADATA_FILE = MODELS_DIR / "metadata.json"

class ModelRegistry:
    """Manages model artifacts, feature signatures, versioning, and evaluation metrics."""

    @staticmethod
    def save_model(
        model_name: str,
        model_obj: Any,
        features: list,
        target: str,
        metrics: Dict[str, float],
        version: str = "1.0.0"
    ) -> Path:
        """Serializes model artifact and logs metadata."""
        file_path = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(model_obj, file_path)
        logger.info(f"Saved model {model_name} to {file_path}")

        # Update metadata.json
        metadata = ModelRegistry.get_all_metadata()
        import datetime
        metadata[model_name] = {
            "model_name": model_name,
            "version": version,
            "target": target,
            "features": features,
            "metrics": metrics,
            "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "artifact_path": str(file_path.relative_to(MODELS_DIR.parent))
        }
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return file_path

    @staticmethod
    def load_model(model_name: str) -> Optional[Any]:
        """Loads model object from disk."""
        file_path = MODELS_DIR / f"{model_name}.joblib"
        if not file_path.exists():
            return None
        return joblib.load(file_path)

    @staticmethod
    def get_all_metadata() -> Dict[str, Any]:
        """Reads full metadata registry."""
        if not METADATA_FILE.exists():
            return {}
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    @staticmethod
    def get_model_metadata(model_name: str) -> Optional[Dict[str, Any]]:
        """Returns metadata for a specific model."""
        meta = ModelRegistry.get_all_metadata()
        return meta.get(model_name)
