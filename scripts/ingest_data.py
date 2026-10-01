"""CLI script for ingesting data archives or flat files into CricketIQ."""

import os
import sys
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import RAW_DATA_DIR, logger
from scripts.build_database import build_database

def main():
    parser = argparse.ArgumentParser(description="CricketIQ Data Ingestion CLI")
    parser.add_argument("--source", type=str, default=str(RAW_DATA_DIR / "ipl_csv2.zip"), help="Path to data file or zip")
    parser.add_argument("--limit", type=int, default=300, help="Max matches to ingest")
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Target file {source_path} does not exist.")
        sys.exit(1)

    limit = None if args.limit <= 0 else args.limit
    build_database(source_path, max_matches=limit)

if __name__ == "__main__":
    main()
