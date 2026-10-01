"""SQLAlchemy ORM models for CricketIQ normalized schema."""

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, Date, ForeignKey, Index, Text
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Player(Base):
    __tablename__ = "players"

    player_id = Column(String(64), primary_key=True)
    name = Column(String(128), nullable=False, index=True)
    batting_hand = Column(String(32), default="Right-hand bat")
    bowling_style = Column(String(64), default="Right-arm medium")
    player_type = Column(String(32), default="Batter")

class Venue(Base):
    __tablename__ = "venues"

    venue_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(256), nullable=False, unique=True, index=True)
    city = Column(String(128), nullable=True)
    total_matches = Column(Integer, default=0)
    avg_first_innings_runs = Column(Float, default=160.0)

class Match(Base):
    __tablename__ = "matches"

    match_id = Column(String(64), primary_key=True)
    date = Column(String(32), nullable=True, index=True)
    season = Column(String(32), nullable=True, index=True)
    format = Column(String(16), default="T20")
    venue = Column(String(256), nullable=True, index=True)
    city = Column(String(128), nullable=True)
    team1 = Column(String(128), nullable=True)
    team2 = Column(String(128), nullable=True)
    winner = Column(String(128), nullable=True)
    toss_winner = Column(String(128), nullable=True)
    toss_decision = Column(String(32), nullable=True)
    player_of_match = Column(String(128), nullable=True)
    target_runs = Column(Integer, nullable=True)

class Delivery(Base):
    __tablename__ = "deliveries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(64), nullable=False, index=True)
    innings = Column(Integer, nullable=False)
    over = Column(Integer, nullable=False)
    ball_number = Column(Integer, nullable=False)
    batter = Column(String(128), nullable=False, index=True)
    bowler = Column(String(128), nullable=False, index=True)
    non_striker = Column(String(128), nullable=True)
    batter_runs = Column(Integer, default=0)
    extras = Column(Integer, default=0)
    total_runs = Column(Integer, default=0)
    wides = Column(Integer, default=0)
    noballs = Column(Integer, default=0)
    byes = Column(Integer, default=0)
    legbyes = Column(Integer, default=0)
    penalty = Column(Integer, default=0)
    is_legal = Column(Integer, default=1)
    is_dot = Column(Integer, default=0)
    is_four = Column(Integer, default=0)
    is_six = Column(Integer, default=0)
    is_wicket = Column(Integer, default=0)
    is_bowler_wicket = Column(Integer, default=0)
    dismissal_kind = Column(String(64), nullable=True)
    player_dismissed = Column(String(128), nullable=True)
    phase = Column(String(32), nullable=False)
    pressure_index = Column(Float, default=50.0)
    pressure_category = Column(String(16), default="Medium")
    batter_hand = Column(String(32), default="Right-hand bat")
    bowler_style = Column(String(64), default="Right-arm medium")
    bowler_type = Column(String(16), default="Pace")
    venue = Column(String(256), nullable=True)

    __table_args__ = (
        Index("idx_batter_match", "batter", "match_id"),
        Index("idx_bowler_match", "bowler", "match_id"),
        Index("idx_matchup", "batter", "bowler"),
        Index("idx_phase", "phase"),
    )

class PlayerMatchStats(Base):
    __tablename__ = "player_match_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(64), nullable=False, index=True)
    player = Column(String(128), nullable=False, index=True)
    date = Column(String(32), nullable=True)
    season = Column(String(32), nullable=True)
    venue = Column(String(256), nullable=True)
    opposition = Column(String(128), nullable=True)
    innings = Column(Integer, default=1)
    
    # Batting stats
    bat_runs = Column(Integer, default=0)
    balls_faced = Column(Integer, default=0)
    fours = Column(Integer, default=0)
    sixes = Column(Integer, default=0)
    dots = Column(Integer, default=0)
    is_out = Column(Integer, default=0)
    strike_rate = Column(Float, default=0.0)
    
    # Bowling stats
    balls_bowled = Column(Integer, default=0)
    overs = Column(Float, default=0.0)
    runs_conceded = Column(Integer, default=0)
    wickets = Column(Integer, default=0)
    economy = Column(Float, default=0.0)
    bowling_dots = Column(Integer, default=0)
