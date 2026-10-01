"""Orchestrates end-to-end model training, evaluation, and artifact registration."""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.ml.batting_model import BattingModelTrainer
from src.ml.bowling_model import BowlingModelTrainer
from src.ml.clustering import PlayerClusteringEngine
from src.utils.logger import logger

def main():
    print("=" * 60)
    print("CRICKETIQ MACHINE LEARNING TRAINING PIPELINE")
    print("=" * 60)

    # 1. Batting Model Training
    print("\n[1/3] Training Batting Runs & Strike Rate Models...")
    try:
        bat_res = BattingModelTrainer.train_and_evaluate(time_split_ratio=0.8)
        print(f"  -> Best Batting Model: {bat_res['best_model']}")
        print(f"  -> Comparison: {bat_res['comparison']}")
        print(f"  -> Samples: Train={bat_res['train_samples']}, Test={bat_res['test_samples']}")
    except Exception as e:
        print(f"  ! Batting model training error: {e}")

    # 2. Bowling Model Training
    print("\n[2/3] Training Bowling Wickets & Economy Models...")
    try:
        bowl_res = BowlingModelTrainer.train_and_evaluate(time_split_ratio=0.8)
        print(f"  -> Best Bowling Model: {bowl_res['best_model']}")
        print(f"  -> Comparison: {bowl_res['comparison']}")
        print(f"  -> Samples: Train={bowl_res['train_samples']}, Test={bowl_res['test_samples']}")
    except Exception as e:
        print(f"  ! Bowling model training error: {e}")

    # 3. Unsupervised Player Archetypes
    print("\n[3/3] Discovering Player Archetypes via Unsupervised Clustering...")
    try:
        cluster_res = PlayerClusteringEngine.cluster_batters(min_balls=30)
        print(f"  -> Clustered {cluster_res.get('total_players_clustered', 0)} players.")
        print(f"  -> Silhouette Score: {cluster_res.get('silhouette_score')}")
        print(f"  -> Archetypes Discovered:")
        for c, name in cluster_res.get("archetype_names", {}).items():
            print(f"     Cluster {c}: {name}")
    except Exception as e:
        print(f"  ! Clustering error: {e}")

    print("\n" + "=" * 60)
    print("MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
