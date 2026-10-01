"""Builds and populates the CricketIQ SQLite database from raw datasets."""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from sqlalchemy import text

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import RAW_DATA_DIR, DATABASE_PATH
from src.utils.logger import logger
from src.data.ingestion import CricsheetZipAdapter, FlatCSVAdapter
from src.data.validation import DataValidator
from src.data.cleaning import DataCleaner
from src.data.transformation import DataTransformer, PLAYER_PROFILES
from src.database.connection import engine
from src.database.schema import init_db
from src.database.models import Player, Venue, Match, Delivery, PlayerMatchStats

def build_database(zip_path: Path, max_matches: int = None, drop_existing: bool = True):
    """Orchestrates database creation, quality validation, ingestion, and rollups."""
    print("=" * 60)
    print("CRICKETIQ DATABASE BUILDER")
    print("=" * 60)

    if not zip_path.exists():
        print(f"Error: Archive not found at {zip_path}")
        return False

    # 1. Initialize schema
    print("\n[1/5] Initializing Database Schema...")
    init_db(drop_existing=drop_existing)

    # 2. Ingest raw data
    print(f"\n[2/5] Ingesting Cricket Data from {zip_path.name}...")
    adapter = CricsheetZipAdapter(max_matches=max_matches)
    matches_df, deliveries_df = adapter.load(zip_path)
    print(f"Loaded {len(deliveries_df):,} deliveries across {len(matches_df):,} matches.")

    # 3. Clean and Validate
    print("\n[3/5] Cleaning and Validating Data...")
    matches_df = DataCleaner.clean_matches(matches_df)
    deliveries_df = DataCleaner.clean_deliveries(deliveries_df)

    is_valid, dq_report = DataValidator.validate_deliveries(deliveries_df)
    print("\n" + str(dq_report))

    # 4. Enrich Transformations
    print("\n[4/5] Deriving Features (Phases, Pressure Index, Player Metadata)...")
    deliveries_df = DataTransformer.enrich_deliveries(deliveries_df, matches_df)

    # 5. Populate Database Tables
    print("\n[5/5] Writing to Database...")
    
    # Venues
    venues = deliveries_df["venue"].dropna().unique()
    venues_df = pd.DataFrame({"name": venues, "city": None})
    venues_df.to_sql("venues", engine, if_exists="append", index=False)
    print(f"  -> Inserted {len(venues_df)} venues.")

    # Players
    all_batters = set(deliveries_df["batter"].dropna().unique())
    all_bowlers = set(deliveries_df["bowler"].dropna().unique())
    all_players = sorted(list(all_batters.union(all_bowlers)))
    
    players_data = []
    for p in all_players:
        meta = DataTransformer.get_player_metadata(p)
        players_data.append({
            "player_id": p.lower().replace(" ", "_"),
            "name": p,
            "batting_hand": meta.get("hand", "Right-hand bat"),
            "bowling_style": meta.get("style", "Right-arm medium"),
            "player_type": meta.get("role", "Player")
        })
    pd.DataFrame(players_data).to_sql("players", engine, if_exists="append", index=False)
    print(f"  -> Inserted {len(players_data)} players.")

    # Matches
    if not matches_df.empty:
        # Match columns to schema
        match_cols = [c.name for c in Match.__table__.columns]
        valid_m_cols = [c for c in matches_df.columns if c in match_cols]
        matches_df[valid_m_cols].to_sql("matches", engine, if_exists="append", index=False)
        print(f"  -> Inserted {len(matches_df)} matches.")

    # Deliveries
    delivery_cols = [c.name for c in Delivery.__table__.columns if c.name != "id"]
    valid_d_cols = [c for c in deliveries_df.columns if c in delivery_cols]
    
    # Write deliveries in chunks for optimal memory usage
    chunksize = 25000
    deliveries_df[valid_d_cols].to_sql(
        "deliveries", engine, if_exists="append", index=False, chunksize=chunksize
    )
    print(f"  -> Inserted {len(deliveries_df):,} deliveries.")

    # Aggregate and populate player_match_stats
    print("  -> Generating player match statistics rollups...")
    
    # Batting rollup
    bat_stats = deliveries_df.groupby(["match_id", "batter"]).agg(
        bat_runs=("batter_runs", "sum"),
        balls_faced=("is_legal", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
        dots=("is_dot", "sum"),
        venue=("venue", "first"),
        innings=("innings", "first"),
        date=("date", "first") if "date" in deliveries_df.columns else ("innings", lambda x: "Unknown")
    ).reset_index().rename(columns={"batter": "player"})

    # Dismissal check
    dismissals = deliveries_df[deliveries_df["player_dismissed"].notnull()].groupby(
        ["match_id", "player_dismissed"]
    ).size().reset_index().rename(columns={"player_dismissed": "player", 0: "is_out"})
    dismissals["is_out"] = 1

    bat_stats = pd.merge(bat_stats, dismissals, on=["match_id", "player"], how="left")
    bat_stats["is_out"] = bat_stats["is_out"].fillna(0).astype(int)
    bat_stats["strike_rate"] = (bat_stats["bat_runs"] / bat_stats["balls_faced"].replace(0, 1) * 100.0).round(2)

    # Bowling rollup
    bowl_stats = deliveries_df.groupby(["match_id", "bowler"]).agg(
        balls_bowled=("is_legal", "sum"),
        runs_conceded=("total_runs", "sum"),
        wickets=("is_bowler_wicket", "sum"),
        bowling_dots=("is_dot", "sum")
    ).reset_index().rename(columns={"bowler": "player"})
    bowl_stats["overs"] = (bowl_stats["balls_bowled"] / 6.0).round(1)
    bowl_stats["economy"] = (bowl_stats["runs_conceded"] / bowl_stats["overs"].replace(0, 1)).round(2)

    # Merge batting & bowling into player_match_stats
    pms = pd.merge(bat_stats, bowl_stats, on=["match_id", "player"], how="outer")
    for col in ["bat_runs", "balls_faced", "fours", "sixes", "dots", "is_out", "balls_bowled", "runs_conceded", "wickets", "bowling_dots"]:
        if col in pms.columns:
            pms[col] = pms[col].fillna(0).astype(int)
    for col in ["strike_rate", "overs", "economy"]:
        if col in pms.columns:
            pms[col] = pms[col].fillna(0.0).astype(float)

    # Fill venue/date if missing from outer merge
    if "venue" in pms.columns and pms["venue"].isnull().any():
        match_venue_map = deliveries_df.drop_duplicates("match_id").set_index("match_id")["venue"].to_dict()
        pms["venue"] = pms["venue"].fillna(pms["match_id"].map(match_venue_map))

    pms.to_sql("player_match_stats", engine, if_exists="append", index=False)
    print(f"  -> Inserted {len(pms):,} player match performance records.")

    print("\nDATABASE BUILD COMPLETE SUCCESSFULLY!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CricketIQ Database Builder")
    parser.add_argument("--zip", type=str, default=str(RAW_DATA_DIR / "ipl_csv2.zip"), help="Path to zip archive")
    parser.add_argument("--limit", type=int, default=300, help="Number of matches to ingest (default: 300 for high speed & depth; set 0 for all)")
    args = parser.parse_args()

    limit = None if args.limit <= 0 else args.limit
    build_database(Path(args.zip), max_matches=limit)
