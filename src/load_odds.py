"""Step 5 of the pipeline: load historical odds from a CSV into the ``odds`` table.

This module is **source-agnostic**. You supply a CSV in a standard schema and
it joins to existing games by ``(game_date, home_team_abbr, away_team_abbr)``.

### Expected CSV schema

Required columns:
    game_date            YYYY-MM-DD
    home_team_abbr       3-letter (e.g. LAL, BOS, GSW)
    away_team_abbr       3-letter
    source               free-text: 'sportsbookreview', 'pinnacle_close', etc.
    line_type            'open' or 'close'

Optional columns (any subset; missing ones become NULL):
    home_ml, away_ml                       American moneyline integers
    home_spread, home_spread_odds, away_spread_odds
    total, over_odds, under_odds

Any extra columns are ignored.

### Where to get historical NBA odds

Free / cheap options:

  - **sportsbookreviewsonline.com** — per-season Excel archives going back
    to the 2007-08 season. Format isn't quite our schema; reshape with
    pandas (one row per game, not per team). Most accessible free source.
  - **Kaggle** — search "NBA betting" or "NBA odds"; several user-uploaded
    historical datasets.
  - **The Odds API** (the-odds-api.com) — free tier covers live odds;
    historical odds require a paid plan.
  - **OddsPortal** — clean per-game pages but scraping is against ToS;
    consider only if you have permission.

Whichever source you pick, reshape it to the schema above and drop it at
``data/odds.csv`` (or pass ``--path`` to point elsewhere).

### Team abbreviation normalization

Different sources use different abbreviations (BKN vs BRK, PHX vs PHO,
CHA vs CHO, NOP vs NO, ...). We normalize known aliases to NBA's official
abbreviations before joining. Update ``ABBR_ALIASES`` if you hit a source
that uses something we don't yet handle.

Run it:
    python -m src.load_odds                       # default: data/odds.csv
    python -m src.load_odds --path mypath.csv
"""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import config
from src.db import get_conn

logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = {"game_date", "home_team_abbr", "away_team_abbr", "source", "line_type"}

OPTIONAL_NUMERIC = [
    "home_ml", "away_ml",
    "home_spread", "home_spread_odds", "away_spread_odds",
    "total", "over_odds", "under_odds",
]

# Maps non-NBA abbreviations -> NBA.com canonical 3-letter codes.
# (NBA.com uses BKN, PHX, CHA, NOP, NYK, UTA, GSW, SAS.)
ABBR_ALIASES = {
    "BRK": "BKN", "NJN": "BKN",
    "PHO": "PHX",
    "CHO": "CHA",
    "NO":  "NOP", "NOH": "NOP",
    "NY":  "NYK",
    "UTH": "UTA",
    "GS":  "GSW",
    "SA":  "SAS",
    "WSH": "WAS",
}


def normalize_abbr(s: pd.Series) -> pd.Series:
    return s.str.upper().str.strip().replace(ABBR_ALIASES)


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Odds CSV {path} is missing required columns: {sorted(missing)}. "
            f"Found: {sorted(df.columns.tolist())}"
        )

    # Normalize types
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    df["home_team_abbr"] = normalize_abbr(df["home_team_abbr"])
    df["away_team_abbr"] = normalize_abbr(df["away_team_abbr"])
    df["line_type"] = df["line_type"].str.lower().str.strip()

    invalid = df[~df["line_type"].isin(["open", "close"])]
    if len(invalid):
        logger.warning("Dropping %d rows with line_type not in {open, close}", len(invalid))
        df = df[df["line_type"].isin(["open", "close"])].copy()

    # Ensure all numeric columns exist (NaN if absent).
    for c in OPTIONAL_NUMERIC:
        if c not in df.columns:
            df[c] = pd.NA

    return df


def match_to_games(conn, odds_df: pd.DataFrame) -> pd.DataFrame:
    """Join odds rows to games by (date, home_abbr, away_abbr) → game_id."""
    games = pd.read_sql(
        """
        SELECT game_id, game_date, home_team_abbr, away_team_abbr
        FROM games
        """,
        conn,
    )
    games["home_team_abbr"] = normalize_abbr(games["home_team_abbr"])
    games["away_team_abbr"] = normalize_abbr(games["away_team_abbr"])

    merged = odds_df.merge(
        games,
        on=["game_date", "home_team_abbr", "away_team_abbr"],
        how="left",
    )

    unmatched = merged[merged["game_id"].isna()]
    if len(unmatched):
        # Show first few examples to help diagnose abbreviation mismatches.
        sample = unmatched[["game_date", "home_team_abbr", "away_team_abbr"]].head(5)
        logger.warning(
            "%d odds rows did not match a game in the database. Sample:\n%s",
            len(unmatched), sample.to_string(index=False),
        )

    return merged.dropna(subset=["game_id"])


def upsert(conn, odds_df: pd.DataFrame) -> int:
    if odds_df.empty:
        return 0

    odds_df = odds_df.copy()
    odds_df["loaded_at"] = datetime.now(timezone.utc).isoformat()

    insert_cols = [
        "game_id", "source", "line_type",
        "home_ml", "away_ml",
        "home_spread", "home_spread_odds", "away_spread_odds",
        "total", "over_odds", "under_odds",
        "loaded_at",
    ]
    rows = odds_df[insert_cols].to_dict(orient="records")
    # Convert pandas NA to None for sqlite.
    cleaned = []
    for r in rows:
        cleaned.append({k: (None if pd.isna(v) else v) for k, v in r.items()})

    cur = conn.cursor()
    cur.executemany(
        """
        INSERT OR REPLACE INTO odds (
            game_id, source, line_type,
            home_ml, away_ml,
            home_spread, home_spread_odds, away_spread_odds,
            total, over_odds, under_odds,
            loaded_at
        ) VALUES (
            :game_id, :source, :line_type,
            :home_ml, :away_ml,
            :home_spread, :home_spread_odds, :away_spread_odds,
            :total, :over_odds, :under_odds,
            :loaded_at
        )
        """,
        cleaned,
    )
    conn.commit()
    return len(cleaned)


def main():
    parser = argparse.ArgumentParser(description="Load odds CSV into the odds table.")
    parser.add_argument(
        "--path",
        type=Path,
        default=config.DATA_DIR / "odds.csv",
        help="Path to odds CSV. Default: data/odds.csv",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not args.path.exists():
        logger.error("Odds CSV not found: %s", args.path)
        logger.error(
            "See src/load_odds.py docstring for the expected schema and "
            "suggested data sources."
        )
        return

    df = load_csv(args.path)
    logger.info("Read %d odds rows from %s", len(df), args.path)

    conn = get_conn()
    try:
        matched = match_to_games(conn, df)
        logger.info("Matched %d / %d rows to games", len(matched), len(df))
        n = upsert(conn, matched)
        logger.info("Upserted %d rows into odds table", n)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
