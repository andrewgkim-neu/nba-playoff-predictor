"""Reshape sportsbookreviewsonline.com NBA odds archives into our odds CSV schema.

SBR publishes one Excel file per season at:
    https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba/nbaoddsarchives.htm
(files named like "nba odds 2023-24.xlsx").

Their format is two rows per game (visitor row, then home row) with columns:
    Date, Rot, VH, Team, 1st, 2nd, 3rd, 4th, Final, Open, Close, ML, 2H
  - Date is an integer MMDD with no year (e.g. 1019 = Oct 19, 201 = Feb 1).
  - Team is a spaceless name ("GoldenState", "LAClippers"), not a 3-letter code.
  - ML is the closing American moneyline for that team.

We extract the **moneyline only** (the field we need for win-probability
comparison) and emit one row per game in the schema src/load_odds.py expects.
Spread/total are intentionally skipped: SBR encodes them ambiguously (spread vs
total inferred by magnitude), and we don't need them for the model-vs-market
comparison.

Usage:
    python -m src.reshape_sbr --input "data/sbr/*.xlsx" --out data/odds.csv
    python -m src.reshape_sbr --input data/sbr/nba_odds_2023-24.xlsx   # single file

Then load it:
    python -m src.load_odds
"""

import argparse
import glob
import logging
import re
from pathlib import Path

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


# SBR team name (spaces/dots/case-insensitive) -> NBA.com 3-letter abbreviation.
SBR_TEAM_MAP = {
    "atlanta": "ATL", "boston": "BOS", "brooklyn": "BKN", "brooklynnets": "BKN",
    "newjersey": "BKN", "charlotte": "CHA", "chicago": "CHI", "cleveland": "CLE",
    "dallas": "DAL", "denver": "DEN", "detroit": "DET", "goldenstate": "GSW",
    "houston": "HOU", "indiana": "IND", "laclippers": "LAC", "losangelesclippers": "LAC",
    "clippers": "LAC", "lalakers": "LAL", "losangeleslakers": "LAL", "lakers": "LAL",
    "memphis": "MEM", "miami": "MIA", "milwaukee": "MIL", "minnesota": "MIN",
    "neworleans": "NOP", "newyork": "NYK", "oklahomacity": "OKC", "oklahoma": "OKC",
    "orlando": "ORL", "philadelphia": "PHI", "phoenix": "PHX", "portland": "POR",
    "sacramento": "SAC", "sanantonio": "SAS", "toronto": "TOR", "utah": "UTA",
    "washington": "WAS",
}


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z]", "", str(name).lower())


def _team_abbr(name: str) -> str | None:
    return SBR_TEAM_MAP.get(_norm_name(name))


def _season_start_year(path: Path, override: int | None) -> int:
    if override is not None:
        return override
    m = re.search(r"(\d{4})-\d{2}", path.name)
    if not m:
        raise SystemExit(
            f"Can't infer season year from filename '{path.name}'. "
            f"Pass --season (e.g. --season 2023 for the 2023-24 season)."
        )
    return int(m.group(1))


def _parse_date(raw, season_start: int) -> str | None:
    """SBR integer MMDD -> 'YYYY-MM-DD', inferring year from the season."""
    try:
        s = str(int(float(raw)))
    except (ValueError, TypeError):
        return None
    if len(s) == 3:        # M DD
        month, day = int(s[0]), int(s[1:])
    elif len(s) == 4:      # MM DD
        month, day = int(s[:2]), int(s[2:])
    else:
        return None
    # Oct-Dec belong to the season's first year; Jan-Sep to the second.
    year = season_start if month >= 10 else season_start + 1
    return f"{year:04d}-{month:02d}-{day:02d}"


def _parse_ml(raw) -> int | None:
    try:
        return int(float(raw))
    except (ValueError, TypeError):
        return None  # 'NL', blank, pick'em, etc.


def _col(df: pd.DataFrame, name: str) -> str | None:
    for c in df.columns:
        if str(c).strip().lower() == name.lower():
            return c
    return None


def reshape_file(path: Path, season_override: int | None) -> list[dict]:
    season_start = _season_start_year(path, season_override)
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]

    date_c, vh_c, team_c, ml_c = (_col(df, "Date"), _col(df, "VH"), _col(df, "Team"), _col(df, "ML"))
    for label, c in [("Date", date_c), ("Team", team_c), ("ML", ml_c)]:
        if c is None:
            raise SystemExit(f"{path.name}: required column '{label}' not found. Columns: {list(df.columns)}")

    rows, unknown = [], set()
    n = len(df)
    if n % 2 != 0:
        logger.warning("%s: odd row count (%d) — last game may be incomplete", path.name, n)

    for i in range(0, n - 1, 2):
        visitor, home = df.iloc[i], df.iloc[i + 1]

        # Sanity-check the visitor/home ordering when a VH column exists.
        if vh_c is not None:
            vh_v, vh_h = str(visitor[vh_c]).strip().upper(), str(home[vh_c]).strip().upper()
            if vh_v == "H" and vh_h in ("V", "N"):
                visitor, home = home, visitor  # rows came home-first; swap

        date = _parse_date(visitor[date_c], season_start)
        away_abbr, home_abbr = _team_abbr(visitor[team_c]), _team_abbr(home[team_c])
        away_ml, home_ml = _parse_ml(visitor[ml_c]), _parse_ml(home[ml_c])

        if away_abbr is None:
            unknown.add(str(visitor[team_c]))
        if home_abbr is None:
            unknown.add(str(home[team_c]))
        if not (date and away_abbr and home_abbr and away_ml is not None and home_ml is not None):
            continue

        rows.append({
            "game_date": date,
            "home_team_abbr": home_abbr,
            "away_team_abbr": away_abbr,
            "source": "sportsbookreviewsonline",
            "line_type": "close",
            "home_ml": home_ml,
            "away_ml": away_ml,
        })

    if unknown:
        logger.warning("%s: unmapped team names (add to SBR_TEAM_MAP): %s", path.name, sorted(unknown))
    logger.info("%s: %d games with moneylines", path.name, len(rows))
    return rows


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Reshape SBR NBA odds archives to our odds CSV.")
    parser.add_argument("--input", required=True, help="Path or glob to SBR .xlsx file(s)")
    parser.add_argument("--out", type=Path, default=config.DATA_DIR / "odds.csv", help="Output CSV path")
    parser.add_argument("--season", type=int, default=None,
                        help="Season start year if not inferable from filename (e.g. 2023 for 2023-24)")
    args = parser.parse_args()

    paths = [Path(p) for p in sorted(glob.glob(args.input))]
    if not paths:
        raise SystemExit(f"No files matched: {args.input}")

    all_rows = []
    for p in paths:
        all_rows.extend(reshape_file(p, args.season))

    if not all_rows:
        raise SystemExit("No moneyline rows extracted. Check the file format / column names.")

    out = pd.DataFrame(all_rows)
    out.to_csv(args.out, index=False)
    logger.info("Wrote %d rows -> %s. Next: python -m src.load_odds", len(out), args.out)


if __name__ == "__main__":
    main()
