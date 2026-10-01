"""CricketIQ System Configuration.

Centralized configuration for file paths, database connections,
model hyperparameters, analytics thresholds, and feature settings.
"""

from pathlib import Path
import os
from typing import Dict, Any

# Root Directory of the CricketIQ Project
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

# Ensure runtime directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, SAMPLE_DATA_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Database Configuration
DATABASE_PATH = os.getenv("CRICKETIQ_DB_PATH", str(PROCESSED_DATA_DIR / "cricketiq.db"))
DATABASE_URL = os.getenv("CRICKETIQ_DATABASE_URL", f"sqlite:///{DATABASE_PATH}")

# Models Path
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Analytics Thresholds
MIN_MATCHUP_SAMPLE = int(os.getenv("CRICKETIQ_MIN_MATCHUP_SAMPLE", "12")) # Min balls faced for reliable matchup
MIN_PLAYER_INNINGS = int(os.getenv("CRICKETIQ_MIN_PLAYER_INNINGS", "5")) # Min innings for player stats
FORM_WINDOW = int(os.getenv("CRICKETIQ_FORM_WINDOW", "10")) # Number of recent matches for rolling form
FORM_DECAY = float(os.getenv("CRICKETIQ_FORM_DECAY", "0.05")) # Exponential decay factor for match age
RANDOM_STATE = int(os.getenv("CRICKETIQ_RANDOM_STATE", "42"))

# T20 Over Phase Definitions (0-indexed overs: 0 to 19)
PHASE_POWERPLAY_MAX = 5 # 0.1 to 5.6 (overs 0-5)
PHASE_MIDDLE_MAX = 14 # 6.1 to 14.6 (overs 6-14)
PHASE_DEATH_MAX = 19 # 15.1 to 19.6 (overs 15-19)

# Cache Settings
CACHE_TTL = 3600 # 1 hour default cache for Streamlit computations
