"""Project configuration. Edit SEASONS to expand or narrow the data pull."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "nba.db"

# Seasons to fetch. NBA season strings are "YYYY-YY".
# Default is 5 recent seasons; expand once you've validated the pipeline.
# Each season takes ~30-40 minutes of box-score fetching at the default throttle.
SEASONS = [
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
]

# Season types to pull. "Regular Season" + "Playoffs" is what you want for
# this project. Other valid values: "Pre Season", "All Star", "PlayIn".
SEASON_TYPES = ["Regular Season", "Playoffs"]

# --- Networking ---

# Seconds between API calls. NBA.com is rate-sensitive; don't go below 0.6.
REQUEST_DELAY = 0.6

# Per-request timeout in seconds.
REQUEST_TIMEOUT = 60

# Retry config for transient failures (timeouts, 5xx, occasional NBA.com hiccups).
MAX_RETRIES = 5
RETRY_BACKOFF_MULTIPLIER = 2  # seconds
RETRY_BACKOFF_MAX = 30        # seconds

# Ensure data directories exist on import.
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
