"""Database schema initialization and migration management."""

from src.database.connection import engine
from src.database.models import Base
from src.utils.logger import logger

def init_db(drop_existing: bool = False) -> None:
    """Initializes database schema, optionally dropping existing tables."""
    if drop_existing:
        logger.warning("Dropping all existing database tables...")
        Base.metadata.drop_all(bind=engine)
    logger.info("Creating database tables if not exist...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema initialized successfully.")

def verify_schema() -> bool:
    """Verifies that all expected tables exist in the database."""
    from sqlalchemy import inspect
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    expected_tables = ["players", "venues", "matches", "deliveries", "player_match_stats"]
    missing = [t for t in expected_tables if t not in existing_tables]
    if missing:
        logger.warning(f"Missing tables in database: {missing}")
        return False
    return True
