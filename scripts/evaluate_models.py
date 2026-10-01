"""Model Evaluation and Reporting CLI Script for CricketIQ."""

import sys
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.ml.model_registry import ModelRegistry

def main():
    print("=" * 60)
    print("CRICKETIQ MODEL EVALUATION REPORT")
    print("=" * 60)

    metadata = ModelRegistry.get_all_metadata()
    if not metadata:
        print("No trained models found in registry. Please run scripts/train_models.py first.")
        return

    for model_name, info in metadata.items():
        print(f"\nModel: {model_name} (Version: {info.get('version', '1.0.0')})")
        print(f"Target: {info.get('target')}")
        print(f"Features: {', '.join(info.get('features', []))}")
        print(f"Trained On: {info.get('training_date')}")
        print("Performance Metrics:")
        metrics = info.get("metrics", {})
        for m_name, val in metrics.items():
            print(f"  - {m_name}: {val}")
        print("-" * 50)

    print("=" * 60)

if __name__ == "__main__":
    main()
