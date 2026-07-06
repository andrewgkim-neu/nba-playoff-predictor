"""Reshape the Kaggle 'NBA Betting Data 2008-2026' CSV into our odds CSV schema.

That dataset is one row per game with columns:
    season, date, regular, playoffs, away, home, score_*, q*_*,
    whos_favored, spread, total, moneyline_away, moneyline_home, ...

We extract the **moneylines** (the field we need for win-probability comparison)
and emit one row per game in the schema src/load_odds.py expects. Team codes
are lowercase and non-standard (gs, sa, utah, wsh); we map them to NBA.com
3-letter abbreviations so the (date, home_abbr, away_abbr) join lands.

By default we only emit rows for the seasons configured in src/config.SEASONS
(matched on the dataset's end-year season label, e.g. '2024-25' -> 2025), so the
output lines up with what's in the database. Pass --all-seasons to emit every
season the file covers.

Usage:
    python -m src.reshape_kaggle                       # defaults below
    python -m src.reshape_kaggle --input data/kaggle/nba_2008-2026.csv --out data/odds.csv

Then load it:
    python -m src.load_odds
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


# Kaggle dataset's lowercase team code -> NBA.com 3-letter abbreviation.
TEAM_MAP = {
    "atl": "ATL", "bkn": "BKN", "bos": "BOS", "cha": "CHA", "chi": "CHI",
    "cle": "CLE", "dal": "DAL", "den": "DEN", "det": "DET", "gs": "GSW",
    "hou": "HOU", "ind": "IND", "lac": "LAC", "lal": "LAL", "mem": "MEM",
    "mia": "MIA", "mil": "MIL", "min": "MIN", "no": "NOP", "ny": "NYK",
    "okc": "OKC", "orl": "ORL", "phi": "PHI", "phx": "PHX", "por": "POR",
    "sa": "SAS", "sac": "SAC", "tor": "TOR", "utah": "UTA", "wsh": "WAS",
}


def configured_season_end_years() -> set[int]:
    """End-year labels for the seasons in config.SEASONS ('2024-25' -> 2025)."""
    years = set()
    for s in config.SEASONS:
        start, yy = s.split("-")
        start = int(start)
        years.add((start // 100) * 100 + int(yy))
    return years


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Reshape Kaggle NBA betting CSV to our odds CSV.")
    parser.add_argument("--input", type=Path, default=config.DATA_DIR / "kaggle" / "nba_2008-2026.csv")
    parser.add_argument("--out", type=Path, default=config.DATA_DIR / "odds.csv")
    parser.add_argument("--all-seasons", action="store_true",
                        help="Emit every season in the file (default: only config.SEASONS)")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    df = pd.read_csv(args.input)

    if not args.all_seasons:
        years = configured_season_end_years()
        df = df[df["season"].isin(years)]
        logger.info("Filtered to seasons %s -> %d rows", sorted(years), len(df))

    # Map team codes; warn on anything unmapped rather than silently dropping.
    unknown = (set(df["home"]) | set(df["away"])) - set(TEAM_MAP)
    if unknown:
        logger.warning("Unmapped team codes (add to TEAM_MAP): %s", sorted(unknown))
    df = df.copy()
    df["home_team_abbr"] = df["home"].map(TEAM_MAP)
    df["away_team_abbr"] = df["away"].map(TEAM_MAP)

    # Signed point spread from the home team's perspective (negative = home
    # favored). `spread` is a positive magnitude; `whos_favored` gives direction.
    df["home_spread"] = np.where(df["whos_favored"] == "home", -df["spread"], df["spread"])

    # Keep games that have *either* moneylines or a spread (recent seasons in
    # this dataset have spreads but no moneylines).
    has_ml = df["moneyline_home"].notna() & df["moneyline_away"].notna()
    has_spread = df["home_spread"].notna()
    before = len(df)
    df = df[df["home_team_abbr"].notna() & df["away_team_abbr"].notna() & (has_ml | has_spread)]
    logger.info("Dropped %d rows missing team mapping or any betting line", before - len(df))

    out = pd.DataFrame({
        "game_date": pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"),
        "home_team_abbr": df["home_team_abbr"],
        "away_team_abbr": df["away_team_abbr"],
        "source": "kaggle_nba_betting",
        "line_type": "close",
        "home_ml": df["moneyline_home"].astype("Int64"),
        "away_ml": df["moneyline_away"].astype("Int64"),
        "home_spread": df["home_spread"].astype(float),
    })

    out.to_csv(args.out, index=False)
    logger.info("Wrote %d rows -> %s. Next: python -m src.load_odds", len(out), args.out)


if __name__ == "__main__":
    main()
