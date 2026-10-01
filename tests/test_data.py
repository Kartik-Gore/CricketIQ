"""Unit tests for data validation, schema mapping, and cleaning."""

import pytest
import pandas as pd
from src.data.validation import DataValidator, DataQualityReport
from src.data.cleaning import DataCleaner

def test_data_validator_mapping():
    raw_df = pd.DataFrame({
        "striker": ["V Kohli"],
        "runs_off_bat": [4],
        "wicket_type": ["caught"]
    })
    mapped = DataValidator.map_columns(raw_df)
    assert "batter" in mapped.columns
    assert "batter_runs" in mapped.columns
    assert "dismissal_kind" in mapped.columns

def test_data_validator_quality_check():
    valid_df = pd.DataFrame({
        "match_id": ["1", "1"],
        "innings": [1, 1],
        "over": [0, 0],
        "ball": [1, 2],
        "batter": ["V Kohli", "V Kohli"],
        "bowler": ["JJ Bumrah", "JJ Bumrah"],
        "batter_runs": [0, 4],
        "extras": [0, 0],
        "total_runs": [0, 4]
    })
    is_valid, report = DataValidator.validate_deliveries(valid_df)
    assert is_valid is True
    assert report.total_rows == 2
    assert report.invalid_records == 0

def test_cleaner_legal_deliveries():
    del_df = pd.DataFrame({
        "match_id": ["1", "1"],
        "innings": [1, 1],
        "over": [0, 0],
        "ball": [1, 2],
        "batter": ["V Kohli", "V Kohli"],
        "bowler": ["JJ Bumrah", "JJ Bumrah"],
        "batter_runs": [1, 0],
        "wides": [0, 1],
        "noballs": [0, 0],
        "extras": [0, 1],
        "total_runs": [1, 1]
    })
    cleaned = DataCleaner.clean_deliveries(del_df)
    assert cleaned["is_legal"].tolist() == [1, 0] # Second ball is wide, so not legal
