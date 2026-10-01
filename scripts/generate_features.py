"""Precomputes and validates feature matrices for all active cricketers."""

import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.database.queries import CricketIQQueries
from src.features.batting_features import BattingFeatureExtractor
from src.features.bowling_features import BowlingFeatureExtractor
from src.features.form_features import FormFeatureExtractor
from src.features.context_features import ContextualPerformanceEngine

def main():
    print("=" * 60)
    print("CRICKETIQ FEATURE GENERATOR")
    print("=" * 60)

    players = CricketIQQueries.get_all_player_names()
    print(f"Generating and verifying features for {len(players)} players...")

    bat_count = 0
    bowl_count = 0

    for i, p in enumerate(players):
        bf = BattingFeatureExtractor.extract_player_batting_features(p)
        if bf and bf.get("balls_faced", 0) > 0:
            bat_count += 1
        
        bw = BowlingFeatureExtractor.extract_player_bowling_features(p)
        if bw and bw.get("legal_balls", 0) > 0:
            bowl_count += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(players):
            print(f"Processed {i + 1}/{len(players)} players...")

    print(f"\nFeature generation completed:")
    print(f"  - Active Batters: {bat_count}")
    print(f"  - Active Bowlers: {bowl_count}")
    print("=" * 60)

if __name__ == "__main__":
    main()
