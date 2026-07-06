"""Add lineup-adjusted team ratings to ``game_features`` from player stats.

Pipeline position: run AFTER build_features and build_player_stats.

The idea: each team's strength in a game depends on *which* players are on the
floor and *how many minutes* they play. We approximate that with a
minutes-weighted blend of the active players' on-court ratings:

    team_rating = sum_i(proj_minutes_i * player_rating_i) / sum_i(proj_minutes_i)

Design choices (per the "active roster + projected minutes" approach):
  - Active roster = the players who actually appeared in the game. Starting
    lineups / inactives are announced before tip, so this is pregame info.
  - Player rating = season-to-date expanding mean of the player's on-court
    rating (net/off/def), shifted by one game so it uses only prior games.
  - Projected minutes = the player's rolling last-10-game mean minutes (also
    shifted). We do NOT use the player's actual minutes in this game — those
    depend on game flow (blowouts, foul trouble) and aren't knowable pregame.

The result is three new home_/away_/diff_ columns per metric, which the model's
feature_columns() picks up automatically.

Run it:
    python -m src.build_player_features
"""

import logging

import pandas as pd

from src.db import get_conn

logger = logging.getLogger(__name__)

# Per-player on-court ratings to blend into a team rating.
PLAYER_METRICS = ["net_rating", "off_rating", "def_rating"]

MINUTES_WINDOW = 10        # rolling window for projecting a player's minutes
DEFAULT_PROJ_MINUTES = 10.0  # fallback for players with no minutes history yet

# Column names this module manages (dropped & rebuilt each run so re-running is safe).
def player_feature_columns() -> list[str]:
    cols = []
    for m in PLAYER_METRICS:
        cols += [f"home_player_{m}", f"away_player_{m}", f"diff_player_{m}"]
    return cols


def load_player_games(conn) -> pd.DataFrame:
    df = pd.read_sql(
        """
        SELECT p.game_id, p.player_id, p.is_home, p.minutes,
               p.net_rating, p.off_rating, p.def_rating,
               g.season, g.game_date
        FROM player_game_stats p
        JOIN games g ON g.game_id = p.game_id
        """,
        conn,
    )
    if df.empty:
        raise SystemExit("player_game_stats is empty. Run: python -m src.build_player_stats")
    df["game_date_dt"] = pd.to_datetime(df["game_date"])
    return df.sort_values(["player_id", "season", "game_date_dt"]).reset_index(drop=True)


def compute_player_pregame(df: pd.DataFrame) -> pd.DataFrame:
    """Leak-safe per-player pre-game rating (expanding) and projected minutes (rolling)."""
    grp = df.groupby(["player_id", "season"])

    # Season-to-date expanding mean of each rating, excluding the current game.
    for m in PLAYER_METRICS:
        df[f"rate_{m}"] = grp[m].transform(lambda s: s.shift(1).expanding().mean())

    # Projected minutes = rolling mean of recent minutes, excluding current game.
    df["proj_min"] = grp["minutes"].transform(
        lambda s: s.shift(1).rolling(MINUTES_WINDOW, min_periods=2).mean()
    )
    df["proj_min"] = df["proj_min"].fillna(DEFAULT_PROJ_MINUTES)
    for m in PLAYER_METRICS:
        df[f"rate_{m}"] = df[f"rate_{m}"].fillna(0.0)  # neutral for debutants
    return df


def aggregate_team_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Minutes-weighted team rating per (game, side), pivoted to home/away/diff."""
    for m in PLAYER_METRICS:
        df[f"wr_{m}"] = df["proj_min"] * df[f"rate_{m}"]

    agg = df.groupby(["game_id", "is_home"]).agg(
        sum_w=("proj_min", "sum"),
        **{f"sum_wr_{m}": (f"wr_{m}", "sum") for m in PLAYER_METRICS},
    ).reset_index()
    for m in PLAYER_METRICS:
        agg[m] = agg[f"sum_wr_{m}"] / agg["sum_w"]

    home = agg[agg["is_home"] == 1][["game_id"] + PLAYER_METRICS].rename(
        columns={m: f"home_player_{m}" for m in PLAYER_METRICS})
    away = agg[agg["is_home"] == 0][["game_id"] + PLAYER_METRICS].rename(
        columns={m: f"away_player_{m}" for m in PLAYER_METRICS})
    out = home.merge(away, on="game_id", how="inner")
    for m in PLAYER_METRICS:
        out[f"diff_player_{m}"] = out[f"home_player_{m}"] - out[f"away_player_{m}"]
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    conn = get_conn()
    try:
        players = load_player_games(conn)
        logger.info("Loaded %d player-game rows", len(players))
        players = compute_player_pregame(players)
        team_ratings = aggregate_team_ratings(players)
        logger.info("Built lineup-adjusted ratings for %d games", len(team_ratings))

        gf = pd.read_sql("SELECT * FROM game_features", conn)
        # Drop any prior run's player columns so this is idempotent.
        gf = gf.drop(columns=[c for c in player_feature_columns() if c in gf.columns])
        merged = gf.merge(team_ratings, on="game_id", how="left")

        n_new = len(player_feature_columns())
        matched = merged["diff_player_net_rating"].notna().sum()
        merged.to_sql("game_features", conn, if_exists="replace", index=False)
        logger.info("Added %d player columns to game_features; %d/%d games have ratings",
                    n_new, matched, len(merged))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
