"""Step 4 of the pipeline: build the ``game_features`` analytics table.

This is the table you'll train models on. One row per game, with:

  - Targets: ``home_won`` (1/0), ``home_margin``
  - Context: rest days, games played, home/away team IDs
  - Pre-game team stats from three windows:
      ``_rs``   season-to-date regular-season aggregates (or full RS for playoffs)
      ``_l10``  rolling mean of the last 10 games (RS + playoff blended)
      ``_l5``   rolling mean of the last 5 games
  - For each metric and window: ``home_*``, ``away_*``, and ``diff_*``
    (= home − away). Differentials are usually the strongest features.

CRITICAL — no data leakage: every rolling/expanding computation uses
``.shift(1)`` so the value for a given game depends only on games strictly
before it. Verify this before trusting any model.

This script REBUILDS the table from scratch each run (drops and recreates).
That's fine — building 6,500 games takes a few seconds.

Run it:
    python -m src.build_features
"""

import logging
from datetime import datetime, timezone

import pandas as pd

from src.db import get_conn

logger = logging.getLogger(__name__)


# Box-score metrics we'll aggregate. Add or remove as you like — the rest of
# the script picks these up automatically.
METRICS = [
    "off_rating", "def_rating", "net_rating",
    "pace",
    "efg_pct", "tov_pct", "oreb_pct", "ts_pct",
    "fta_rate",
    "opp_efg_pct", "opp_tov_pct", "opp_oreb_pct", "opp_fta_rate",
]

# Rolling window sizes (in games).
ROLLING_WINDOWS = {"l5": 5, "l10": 10}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_long_team_games(conn) -> pd.DataFrame:
    """One row per (game, team) joining games + box score stats."""
    metric_cols = ", ".join(f"t.{m}" for m in METRICS)
    sql = f"""
        SELECT
            g.game_id, g.season, g.season_type, g.game_date,
            g.home_team_id, g.away_team_id,
            g.home_score, g.away_score,
            t.team_id, t.is_home,
            {metric_cols}
        FROM games g
        JOIN team_game_stats t ON t.game_id = g.game_id
        ORDER BY t.team_id, g.game_date
    """
    df = pd.read_sql(sql, conn)
    df["game_date_dt"] = pd.to_datetime(df["game_date"])
    return df


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_season_to_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``<metric>_rs`` columns: regular-season-to-date expanding mean.

    For Regular Season games: expanding mean over this team's RS games strictly
    before the current game.
    For Playoff games: the team's complete RS aggregate for that season.
    """
    df = df.copy().sort_values(["team_id", "season", "game_date_dt"]).reset_index(drop=True)

    rs = df[df["season_type"] == "Regular Season"].copy()

    # Expanding mean per (team, season), shifted by 1 to exclude the current game.
    for m in METRICS:
        rs[f"{m}_rs"] = (
            rs.groupby(["team_id", "season"])[m]
              .transform(lambda s: s.shift(1).expanding().mean())
        )

    # Full RS aggregate per (team, season) — used for playoff games.
    rs_final = (
        rs.groupby(["team_id", "season"])[METRICS]
          .mean()
          .reset_index()
          .rename(columns={m: f"{m}_rs_full" for m in METRICS})
    )

    df = df.merge(
        rs[["game_id", "team_id"] + [f"{m}_rs" for m in METRICS]],
        on=["game_id", "team_id"], how="left",
    )
    df = df.merge(rs_final, on=["team_id", "season"], how="left")

    # For playoff games, ``<m>_rs`` is NaN; fill with the full RS aggregate.
    for m in METRICS:
        df[f"{m}_rs"] = df[f"{m}_rs"].fillna(df[f"{m}_rs_full"])
        df = df.drop(columns=[f"{m}_rs_full"])

    return df


def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``<metric>_l5`` and ``<metric>_l10`` columns: rolling mean over the
    last N games (RS + playoff combined, but reset per season)."""
    df = df.copy().sort_values(["team_id", "season", "game_date_dt"]).reset_index(drop=True)

    for label, window in ROLLING_WINDOWS.items():
        min_periods = max(2, window // 2)
        for m in METRICS:
            df[f"{m}_{label}"] = (
                df.groupby(["team_id", "season"])[m]
                  .transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean())
            )
    return df


def compute_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``rest_days`` and ``games_played`` per (team, season)."""
    df = df.copy().sort_values(["team_id", "season", "game_date_dt"]).reset_index(drop=True)
    df["rest_days"] = df.groupby(["team_id", "season"])["game_date_dt"].diff().dt.days
    df["games_played"] = df.groupby(["team_id", "season"]).cumcount()
    return df


# ---------------------------------------------------------------------------
# Pivot to one row per game
# ---------------------------------------------------------------------------

def feature_columns() -> list[str]:
    """All per-team feature column names produced by the steps above."""
    cols = ["rest_days", "games_played"]
    cols += [f"{m}_rs" for m in METRICS]
    for label in ROLLING_WINDOWS:
        cols += [f"{m}_{label}" for m in METRICS]
    return cols


def pivot_to_game_level(team_df: pd.DataFrame) -> pd.DataFrame:
    """From long (one row per team-game) to wide (one row per game).

    Note: we drop team_id here because game_meta already has
    home_team_id / away_team_id and we don't want duplicate columns.
    """
    feat_cols = feature_columns()

    home = team_df[team_df["is_home"] == 1][["game_id"] + feat_cols].copy()
    away = team_df[team_df["is_home"] == 0][["game_id"] + feat_cols].copy()

    home = home.rename(columns={c: f"home_{c}" for c in feat_cols})
    away = away.rename(columns={c: f"away_{c}" for c in feat_cols})

    return home.merge(away, on="game_id", how="inner")


def add_differentials(features: pd.DataFrame) -> pd.DataFrame:
    """Add ``diff_<feature>`` = home - away for every metric column."""
    # Skip games_played and rest_days from being treated as a "metric diff",
    # but we still want diff_rest_days (it matters!) — just not diff_games_played.
    skip_diff = {"games_played"}
    for c in feature_columns():
        if c in skip_diff:
            continue
        features[f"diff_{c}"] = features[f"home_{c}"] - features[f"away_{c}"]
    return features


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build(conn) -> pd.DataFrame:
    logger.info("Loading team-game data...")
    long_df = load_long_team_games(conn)
    logger.info("  %d team-game rows loaded", len(long_df))

    logger.info("Computing season-to-date (regular season) features...")
    long_df = compute_season_to_date_features(long_df)

    logger.info("Computing rolling window features...")
    long_df = compute_rolling_features(long_df)

    logger.info("Computing context features (rest, games played)...")
    long_df = compute_context_features(long_df)

    logger.info("Pivoting to one row per game...")
    features = pivot_to_game_level(long_df)
    logger.info("  %d games", len(features))

    logger.info("Joining game metadata + targets...")
    game_meta = pd.read_sql(
        """
        SELECT game_id, season, season_type, game_date,
               playoff_round, series_id, game_num_in_series,
               home_team_id, away_team_id,
               home_score, away_score
        FROM games
        """,
        conn,
    )
    features = game_meta.merge(features, on="game_id", how="inner")

    # Targets
    features["home_won"] = (features["home_score"] > features["away_score"]).astype("Int64")
    features["home_margin"] = features["home_score"] - features["away_score"]

    logger.info("Computing differentials...")
    features = add_differentials(features)

    features["built_at"] = datetime.now(timezone.utc).isoformat()

    # Reorder: meta first, targets, then features (home_*, away_*, diff_*).
    meta_cols = [
        "game_id", "season", "season_type", "game_date",
        "playoff_round", "series_id", "game_num_in_series",
        "home_team_id", "away_team_id",
        "home_score", "away_score",
        "home_won", "home_margin",
    ]
    other = [c for c in features.columns if c not in meta_cols and c != "built_at"]
    home_cols = sorted([c for c in other if c.startswith("home_")])
    away_cols = sorted([c for c in other if c.startswith("away_")])
    diff_cols = sorted([c for c in other if c.startswith("diff_")])
    features = features[meta_cols + home_cols + away_cols + diff_cols + ["built_at"]]

    return features


def write(conn, features: pd.DataFrame) -> None:
    """Drop and recreate the game_features table from the dataframe."""
    features.to_sql("game_features", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_game_features_season ON game_features(season)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_game_features_type ON game_features(season_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_game_features_date ON game_features(game_date)")
    conn.commit()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    conn = get_conn()
    try:
        features = build(conn)
        write(conn, features)
        logger.info("Wrote %d rows to game_features (%d columns)",
                    len(features), len(features.columns))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
