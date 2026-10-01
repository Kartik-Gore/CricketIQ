"""Constants, domain definitions, and mappings for cricket analytics."""

# Match Phases
PHASE_POWERPLAY = "Powerplay"
PHASE_MIDDLE = "Middle"
PHASE_DEATH = "Death"
ALL_PHASES = [PHASE_POWERPLAY, PHASE_MIDDLE, PHASE_DEATH]

# Bowling Styles & Classifications
PACE_STYLES = {
    "Right-arm fast", "Right-arm fast-medium", "Right-arm medium-fast", "Right-arm medium",
    "Left-arm fast", "Left-arm fast-medium", "Left-arm medium-fast", "Left-arm medium"
}
SPIN_STYLES = {
    "Right-arm legbreak", "Right-arm offbreak", "Right-arm legbreak googly",
    "Left-arm orthodox", "Left-arm wrist spin", "Slow left-arm orthodox"
}

# Dismissals attributed to the bowler
BOWLER_DISMISSALS = {
    "bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket"
}

NON_BOWLER_DISMISSALS = {
    "run out", "retired hurt", "obstructing the field", "timed out", "handled the ball"
}

# Trend Categories for Form
TREND_IMPROVING = "Improving"
TREND_STABLE = "Stable"
TREND_DECLINING = "Declining"
TREND_VOLATILE = "Volatile"

# Pressure Categories
PRESSURE_LOW = "Low"
PRESSURE_MEDIUM = "Medium"
PRESSURE_HIGH = "High"

# Standard IPL Team Name Normalization Map
TEAM_NAME_MAP = {
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru"
}
