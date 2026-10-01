"""Data Ingestion layer for CricketIQ.

Supports polymorphic adapters for Cricsheet CSV zip archives, flat CSV files,
JSON files, and Parquet archives with automatic schema adaptation.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import zipfile
import io
import pandas as pd
import numpy as np

from src.config import RAW_DATA_DIR
from src.utils.logger import logger
from src.data.validation import DataValidator

class BaseDataAdapter(ABC):
    """Abstract base class for all cricket data adapters."""

    @abstractmethod
    def load(self, source_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads and returns (matches_df, deliveries_df).
        """
        pass

class CricsheetZipAdapter(BaseDataAdapter):
    """
    Ingests official Cricsheet zip format (containing <match_id>.csv and <match_id>_info.csv).
    """

    def __init__(self, max_matches: Optional[int] = None):
        self.max_matches = max_matches

    def load(self, source_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Extracting and ingesting Cricsheet archive from {source_path}")
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found at: {source_path}")

        matches_list = []
        deliveries_dfs = []

        with zipfile.ZipFile(source_path) as z:
            all_files = z.namelist()
            delivery_files = [
                f for f in all_files 
                if f.endswith(".csv") and not f.endswith("_info.csv") and f != "README.txt"
            ]
            
            if self.max_matches:
                delivery_files = delivery_files[:self.max_matches]
                
            logger.info(f"Processing {len(delivery_files)} matches from archive...")

            for i, d_file in enumerate(delivery_files):
                match_id = d_file.replace(".csv", "")
                info_file = f"{match_id}_info.csv"

                # Parse match delivery CSV
                try:
                    df_del = pd.read_csv(z.open(d_file), low_memory=False)
                    deliveries_dfs.append(df_del)
                except Exception as e:
                    logger.warning(f"Failed to read deliveries for match {match_id}: {e}")
                    continue

                # Parse match info CSV if present
                if info_file in all_files:
                    try:
                        info_lines = z.read(info_file).decode("utf-8", errors="replace").splitlines()
                        match_info: Dict[str, Any] = {
                            "match_id": match_id,
                            "season": None,
                            "date": None,
                            "format": "T20",
                            "venue": None,
                            "city": None,
                            "team1": None,
                            "team2": None,
                            "winner": None,
                            "toss_winner": None,
                            "toss_decision": None,
                            "player_of_match": None,
                            "target_runs": None
                        }
                        teams = []
                        for line in info_lines:
                            parts = [p.strip().strip('"') for p in line.split(",")]
                            if len(parts) >= 3 and parts[0] == "info":
                                key = parts[1]
                                val = parts[2]
                                if key == "season":
                                    match_info["season"] = str(val)
                                elif key == "date":
                                    match_info["date"] = str(val)
                                elif key == "team":
                                    teams.append(val)
                                elif key == "venue":
                                    match_info["venue"] = val
                                elif key == "city":
                                    match_info["city"] = val
                                elif key == "winner":
                                    match_info["winner"] = val
                                elif key == "toss_winner":
                                    match_info["toss_winner"] = val
                                elif key == "toss_decision":
                                    match_info["toss_decision"] = val
                                elif key == "player_of_match":
                                    match_info["player_of_match"] = val
                                elif key == "target_runs" and len(parts) >= 4:
                                    try:
                                        match_info["target_runs"] = int(parts[3])
                                    except ValueError:
                                        pass

                        if len(teams) >= 2:
                            match_info["team1"] = teams[0]
                            match_info["team2"] = teams[1]

                        matches_list.append(match_info)
                    except Exception as e:
                        logger.warning(f"Failed to parse info for match {match_id}: {e}")

        if not deliveries_dfs:
            raise ValueError(f"No delivery data found in archive {source_path}")

        deliveries_df = pd.concat(deliveries_dfs, ignore_index=True)
        matches_df = pd.DataFrame(matches_list) if matches_list else pd.DataFrame()

        logger.info(f"Ingested {len(deliveries_df):,} deliveries across {len(matches_df):,} matches.")
        return matches_df, deliveries_df

class FlatCSVAdapter(BaseDataAdapter):
    """Ingests flat deliveries CSV (e.g. Kaggle IPL ball-by-ball dataset)."""

    def load(self, source_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Ingesting flat CSV from {source_path}")
        df = pd.read_csv(source_path, low_memory=False)
        df = DataValidator.map_columns(df)
        
        # Synthesize matches_df from unique match columns if available
        match_cols = [c for c in ["match_id", "date", "season", "venue", "city", "team1", "team2"] if c in df.columns]
        if match_cols:
            matches_df = df[match_cols].drop_duplicates().reset_index(drop=True)
        else:
            matches_df = pd.DataFrame()
            
        return matches_df, df

def get_data_adapter(file_path: Path, max_matches: Optional[int] = None) -> BaseDataAdapter:
    """Factory method to return suitable adapter based on file extension and content."""
    suffix = file_path.suffix.lower()
    if suffix == ".zip":
        return CricsheetZipAdapter(max_matches=max_matches)
    elif suffix == ".csv":
        return FlatCSVAdapter()
    else:
        raise NotImplementedError(f"Unsupported data format: {suffix}")
