"""Data Quality Engine and Schema Validation for CricketIQ."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from src.utils.logger import logger

# Canonical required and optional schema columns
REQUIRED_DELIVERY_COLUMNS = [
    "match_id", "innings", "over", "ball", "batter", "bowler", "batter_runs", "extras", "total_runs"
]

CANONICAL_COLUMN_MAPPINGS = {
    "striker": "batter",
    "batsman": "batter",
    "batsman_name": "batter",
    "bowler_name": "bowler",
    "runs_off_bat": "batter_runs",
    "batsman_runs": "batter_runs",
    "start_date": "date",
    "match_date": "date",
    "wicket_type": "dismissal_kind",
    "dismissal": "dismissal_kind",
    "is_wicket": "wicket",
}

class DataQualityReport:
    """Encapsulates validation metrics and data quality findings."""

    def __init__(self):
        self.total_rows: int = 0
        self.total_matches: int = 0
        self.total_players: int = 0
        self.missing_values: Dict[str, int] = {}
        self.duplicates: int = 0
        self.invalid_records: int = 0
        self.warnings: List[str] = []
        self.is_valid: bool = True

    def __str__(self) -> str:
        report = [
            "============================================================",
            "DATA QUALITY REPORT",
            "============================================================",
            f"Rows:             {self.total_rows:,}",
            f"Matches:          {self.total_matches:,}",
            f"Players:          {self.total_players:,}",
            f"Duplicates:       {self.duplicates:,}",
            f"Invalid records:  {self.invalid_records:,}",
            f"Validation State: {'PASSED' if self.is_valid else 'FAILED'}",
            "------------------------------------------------------------",
            "Missing Values (Top Columns):"
        ]
        top_missing = sorted(self.missing_values.items(), key=lambda x: x[1], reverse=True)[:5]
        for col, count in top_missing:
            if count > 0:
                report.append(f"  - {col}: {count:,} ({count / max(1, self.total_rows) * 100:.2f}%)")
        if not top_missing or all(c == 0 for _, c in top_missing):
            report.append("  - None in critical columns")
            
        report.append("------------------------------------------------------------")
        report.append("Warnings & Anomalies:")
        if self.warnings:
            for w in self.warnings[:8]:
                report.append(f"  ! {w}")
            if len(self.warnings) > 8:
                report.append(f"  ... and {len(self.warnings) - 8} more warnings.")
        else:
            report.append("  - No anomalies detected.")
        report.append("============================================================")
        return "\n".join(report)

class DataValidator:
    """Validates raw and processed cricket datasets for schema consistency and sanity."""

    @staticmethod
    def map_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes non-canonical column names according to known mappings."""
        renamed = {}
        for col in df.columns:
            clean_col = col.strip().lower()
            if clean_col in CANONICAL_COLUMN_MAPPINGS:
                renamed[col] = CANONICAL_COLUMN_MAPPINGS[clean_col]
        if renamed:
            logger.info(f"Mapped {len(renamed)} columns: {renamed}")
            df = df.rename(columns=renamed)
        return df

    @staticmethod
    def validate_deliveries(df: pd.DataFrame) -> Tuple[bool, DataQualityReport]:
        """Runs automated quality checks on delivery data."""
        report = DataQualityReport()
        report.total_rows = len(df)
        
        if df.empty:
            report.is_valid = False
            report.warnings.append("Dataset is completely empty.")
            return False, report

        # Standardize column names
        df = DataValidator.map_columns(df)

        # Check required schema
        missing_required = [col for col in REQUIRED_DELIVERY_COLUMNS if col not in df.columns]
        if missing_required:
            report.is_valid = False
            report.warnings.append(f"Missing required columns: {missing_required}")
            return False, report

        # Missing values check
        report.missing_values = df[REQUIRED_DELIVERY_COLUMNS].isnull().sum().to_dict()

        # Duplicates check
        dup_subset = ["match_id", "innings", "over", "ball"] if "over" in df.columns and "ball" in df.columns else None
        if dup_subset and all(c in df.columns for c in dup_subset):
            report.duplicates = int(df.duplicated(subset=dup_subset).sum())
            if report.duplicates > 0:
                report.warnings.append(f"Found {report.duplicates} duplicate delivery records.")

        # Entities count
        report.total_matches = int(df["match_id"].nunique()) if "match_id" in df.columns else 0
        batters = set(df["batter"].dropna().unique()) if "batter" in df.columns else set()
        bowlers = set(df["bowler"].dropna().unique()) if "bowler" in df.columns else set()
        report.total_players = len(batters.union(bowlers))

        # Numerical integrity checks
        invalid_runs = df[(df["batter_runs"] < 0) | (df["batter_runs"] > 6)]
        if len(invalid_runs) > 0:
            report.invalid_records += len(invalid_runs)
            report.warnings.append(f"{len(invalid_runs)} deliveries with batter_runs outside [0, 6].")

        if "over" in df.columns:
            invalid_overs = df[(df["over"] < 0) | (df["over"] > 50)]
            if len(invalid_overs) > 0:
                report.invalid_records += len(invalid_overs)
                report.warnings.append(f"{len(invalid_overs)} deliveries with over numbers outside [0, 50].")

        # Null check in critical player names
        null_players = df[df["batter"].isnull() | df["bowler"].isnull()]
        if len(null_players) > 0:
            report.invalid_records += len(null_players)
            report.warnings.append(f"{len(null_players)} deliveries missing batter or bowler identifier.")

        if report.invalid_records > (report.total_rows * 0.05):
            report.is_valid = False
            report.warnings.append("Invalid records exceed 5% tolerance threshold.")

        return report.is_valid, report
