"""Aggregate player_game_stats into per-player season profiles.

One row per (player, season, season_type), so a player's **regular season** and
**playoff** lines are kept separate — which lets the draft game rate players on
either body of work (playoff risers look very different from their RS selves).

Produces the ``player_season_profiles`` table consumed by src/draft.py.

Run it (after build_player_stats):
    python -m src.build_player_profiles
"""

import logging

import pandas as pd

from src.db import get_conn

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    conn = get_conn()
    try:
        df = pd.read_sql("SELECT * FROM player_game_stats", conn)
        if df.empty:
            raise SystemExit("player_game_stats is empty. Run: python -m src.build_player_stats")

        grp = df.groupby(["player_id", "player_name", "season", "season_type"])
        profiles = grp.agg(
            gp=("game_id", "nunique"),
            mpg=("minutes", "mean"),
            ppg=("pts", "mean"),
            net_rating=("net_rating", "mean"),
            off_rating=("off_rating", "mean"),
            def_rating=("def_rating", "mean"),
        ).reset_index()

        # Primary team that season/type = where the player logged the most minutes.
        team = (
            df.groupby(["player_id", "season", "season_type", "team_id"])["minutes"].sum()
              .reset_index().sort_values("minutes")
              .groupby(["player_id", "season", "season_type"]).tail(1)
              [["player_id", "season", "season_type", "team_id"]]
        )
        profiles = profiles.merge(team, on=["player_id", "season", "season_type"], how="left")

        profiles.to_sql("player_season_profiles", conn, if_exists="replace", index=False)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_psp_season "
            "ON player_season_profiles(season, season_type)"
        )
        conn.commit()
        by_type = profiles.groupby("season_type").size().to_dict()
        logger.info("Wrote %d player-season profiles. By type: %s", len(profiles), by_type)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
