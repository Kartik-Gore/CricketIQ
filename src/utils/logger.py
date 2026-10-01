"""Centralized logging infrastructure for CricketIQ."""

import logging
import sys
from pathlib import Path

def setup_logger(name: str = "CricketIQ", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a structured logger for the platform."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Optional File Handler
    log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "cricketiq.log", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception:
        pass

    return logger

logger = setup_logger()
