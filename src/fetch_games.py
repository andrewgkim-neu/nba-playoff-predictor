"""Step 1 of the pipeline: populate the ``games`` table.

For each season in config.SEASONS and each season type in config.SEASON_TYPES,
calls LeagueGameLog and inserts one row per game into the ``games`` table.

Idempotent: re-running with the same config is a no-op (cached + upserted).

Run it:
    python -m src.fetch_games
"""

import logging
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

from nba_api.stats.endpoints import LeagueGameLog

from src import config
from src.api_client import NBAApiClient
from src.db import get_conn, init_db

logger = logging.getLogger(__name__)


def fetch_season_games(client: NBAApiClient, season: str, season_type: str) -> dict:
    """Fetch the league-wide game log for one (season, season_type)."""
    params = {
        "season": season,
        "season_type_all_star": season_type,
        "player_or_team_abbreviation": "T",
    }

    def _do_fetch():
        ep = LeagueGameLog(
            season=season,
            season_type_all_star=season_type,
            player_or_team_abbreviation="T",
            timeout=config.REQUEST_TIMEOUT,
        )
        return ep.get_normalized_dict()

    return client.fetch(
        endpoint_name="LeagueGameLog",
        fetcher_fn=_do_fetch,
        params=params,
    )


def parse_games(raw: dict, season: str, season_type: str) -> list[dict]:
    """LeagueGameLog returns one row per (team, game); merge into one row per game.

    Home vs. away is encoded in the ``MATCHUP`` field:
        "LAL vs. BOS"  -> LAL is home
        "LAL @ BOS"    -> LAL is away
    """
    rows = raw.get("LeagueGameLog", [])
    if not rows:
        logger.warning("No games returned for %s %s", season, season_type)
        return []

    df = pd.DataFrame(rows)
    # 'vs.' = home game; '@' = away
    df["is_home"] = df["MATCHUP"].str.contains("vs.", regex=False)

    home = df[df["is_home"]].copy()
    away = df[~df["is_home"]].copy()

    if len(home) != len(away):
        logger.warning(
            "%s %s: home rows (%d) != away rows (%d) — some games will be dropped",
            season, season_type, len(home), len(away),
        )

    merged = home.merge(away, on="GAME_ID", suffixes=("_home", "_away"))

    fetched_at = datetime.now(timezone.utc).isoformat()
    games = []
    for _, r in merged.iterrows():
        games.append({
            "game_id":            str(r["GAME_ID"]),
            "season":             season,
            "season_type":        season_type,
            "game_date":          r["GAME_DATE_home"],
            "home_team_id":       int(r["TEAM_ID_home"]),
            "away_team_id":       int(r["TEAM_ID_away"]),
            "home_team_abbr":     r["TEAM_ABBREVIATION_home"],
            "away_team_abbr":     r["TEAM_ABBREVIATION_away"],
            "home_score":         int(r["PTS_home"]) if pd.notna(r["PTS_home"]) else None,
            "away_score":         int(r["PTS_away"]) if pd.notna(r["PTS_away"]) else None,
            "playoff_round":      None,
            "series_id":          None,
            "game_num_in_series": None,
            "fetched_at":         fetched_at,
        })
    return games


def upsert_games(conn, games: list[dict]) -> int:
    if not games:
        return 0
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT OR REPLACE INTO games (
            game_id, season, season_type, game_date,
            home_team_id, away_team_id, home_team_abbr, away_team_abbr,
            home_score, away_score,
            playoff_round, series_id, game_num_in_series, fetched_at
        ) VALUES (
            :game_id, :season, :season_type, :game_date,
            :home_team_id, :away_team_id, :home_team_abbr, :away_team_abbr,
            :home_score, :away_score,
            :playoff_round, :series_id, :game_num_in_series, :fetched_at
        )
        """,
        games,
    )
    conn.commit()
    return len(games)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    init_db()
    client = NBAApiClient()
    conn = get_conn()

    total = 0
    try:
        for season in tqdm(config.SEASONS, desc="Seasons"):
            for season_type in config.SEASON_TYPES:
                try:
                    raw = fetch_season_games(client, season, season_type)
                    games = parse_games(raw, season, season_type)
                    n = upsert_games(conn, games)
                    logger.info("%s %s: %d games", season, season_type, n)
                    total += n
                except Exception as e:
                    logger.error("Failed %s %s: %s", season, season_type, e)
    finally:
        conn.close()

    logger.info("Done. Total games upserted: %d", total)


if __name__ == "__main__":
    main()
