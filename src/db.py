"""SQLite schema, connection management, and a status command.

Run directly to initialize the database:
    python -m src.db

Or show what's in it:
    python -m src.db status
"""

import sqlite3
import sys
import logging

from src import config

logger = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id            TEXT    PRIMARY KEY,
    season             TEXT    NOT NULL,
    season_type        TEXT    NOT NULL,        -- 'Regular Season' or 'Playoffs'
    game_date          TEXT    NOT NULL,        -- ISO 'YYYY-MM-DD'
    home_team_id       INTEGER NOT NULL,
    away_team_id       INTEGER NOT NULL,
    home_team_abbr     TEXT,
    away_team_abbr     TEXT,
    home_score         INTEGER,
    away_score         INTEGER,
    -- Playoff context (populate later from CommonPlayoffSeries):
    playoff_round      INTEGER,
    series_id          TEXT,
    game_num_in_series INTEGER,
    fetched_at         TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_games_season       ON games(season);
CREATE INDEX IF NOT EXISTS idx_games_season_type  ON games(season_type);
CREATE INDEX IF NOT EXISTS idx_games_date         ON games(game_date);
CREATE INDEX IF NOT EXISTS idx_games_home_team    ON games(home_team_id);
CREATE INDEX IF NOT EXISTS idx_games_away_team    ON games(away_team_id);


CREATE TABLE IF NOT EXISTS team_game_stats (
    game_id        TEXT    NOT NULL,
    team_id        INTEGER NOT NULL,
    is_home        INTEGER NOT NULL,            -- 1 or 0

    -- Traditional box score
    min            REAL,
    pts            INTEGER,
    fgm            INTEGER, fga INTEGER, fg_pct  REAL,
    fg3m           INTEGER, fg3a INTEGER, fg3_pct REAL,
    ftm            INTEGER, fta INTEGER, ft_pct  REAL,
    oreb           INTEGER, dreb INTEGER, reb    INTEGER,
    ast            INTEGER,
    stl            INTEGER,
    blk            INTEGER,
    tov            INTEGER,
    pf             INTEGER,
    plus_minus     INTEGER,

    -- Advanced
    off_rating     REAL,
    def_rating     REAL,
    net_rating     REAL,
    ast_pct        REAL,
    ast_to         REAL,
    ast_ratio      REAL,
    oreb_pct       REAL,
    dreb_pct       REAL,
    reb_pct        REAL,
    tov_pct        REAL,
    efg_pct        REAL,
    ts_pct         REAL,
    pace           REAL,
    pie            REAL,

    -- Four Factors (opp_* columns capture what the opponent did against this team)
    fta_rate       REAL,
    opp_efg_pct    REAL,
    opp_fta_rate   REAL,
    opp_tov_pct    REAL,
    opp_oreb_pct   REAL,

    fetched_at     TEXT NOT NULL,

    PRIMARY KEY (game_id, team_id),
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);

CREATE INDEX IF NOT EXISTS idx_team_game_stats_team ON team_game_stats(team_id);


CREATE TABLE IF NOT EXISTS odds (
    game_id          TEXT    NOT NULL,
    source           TEXT    NOT NULL,        -- 'sportsbookreview', 'pinnacle', 'consensus', etc.
    line_type        TEXT    NOT NULL,        -- 'open' or 'close'
    home_ml          INTEGER,                 -- American moneyline (e.g. -150 favorite, +130 underdog)
    away_ml          INTEGER,
    home_spread      REAL,                    -- points; negative = home favored
    home_spread_odds INTEGER,
    away_spread_odds INTEGER,
    total            REAL,                    -- over/under points
    over_odds        INTEGER,
    under_odds       INTEGER,
    loaded_at        TEXT    NOT NULL,

    PRIMARY KEY (game_id, source, line_type),
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);

CREATE INDEX IF NOT EXISTS idx_odds_game ON odds(game_id);
"""


def get_conn() -> sqlite3.Connection:
    """Open a connection with sensible defaults."""
    conn = sqlite3.connect(config.DB_PATH)
    # Foreign keys and WAL mode for better concurrent reads.
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = get_conn()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
        logger.info("Database initialized at %s", config.DB_PATH)
    finally:
        conn.close()


def status() -> None:
    """Print row counts and per-season breakdown."""
    conn = get_conn()
    cur = conn.cursor()

    print(f"DB: {config.DB_PATH}")
    print()

    for table in ("games", "team_game_stats", "odds", "game_features"):
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            (n,) = cur.fetchone()
            print(f"  {table:<20} {n:>8,} rows")
        except sqlite3.OperationalError:
            print(f"  {table:<20} (table not created yet)")

    try:
        cur.execute(
            """
            SELECT season, season_type, COUNT(*)
            FROM games
            GROUP BY season, season_type
            ORDER BY season, season_type
            """
        )
        rows = cur.fetchall()
        if rows:
            print()
            print("  Games by season / season type:")
            for season, season_type, count in rows:
                print(f"    {season}  {season_type:<14} {count:>5}")
    except sqlite3.OperationalError:
        pass

    # Coverage: what fraction of games have box scores?
    try:
        cur.execute(
            """
            SELECT
                g.season,
                COUNT(DISTINCT g.game_id) AS total_games,
                COUNT(DISTINCT t.game_id) AS games_with_stats
            FROM games g
            LEFT JOIN team_game_stats t ON t.game_id = g.game_id
            GROUP BY g.season
            ORDER BY g.season
            """
        )
        rows = cur.fetchall()
        if rows:
            print()
            print("  Box score coverage:")
            for season, total, covered in rows:
                pct = (covered / total * 100) if total else 0
                print(f"    {season}  {covered:>5}/{total:<5}  ({pct:5.1f}%)")
    except sqlite3.OperationalError:
        pass

    # Playoff context backfill coverage
    try:
        cur.execute(
            """
            SELECT
                season,
                COUNT(*)                            AS playoff_games,
                SUM(CASE WHEN series_id IS NOT NULL THEN 1 ELSE 0 END) AS with_series
            FROM games
            WHERE season_type = 'Playoffs'
            GROUP BY season
            ORDER BY season
            """
        )
        rows = cur.fetchall()
        if rows:
            print()
            print("  Playoff series context:")
            for season, total, with_series in rows:
                pct = (with_series / total * 100) if total else 0
                print(f"    {season}  {with_series:>4}/{total:<4}  ({pct:5.1f}%)")
    except sqlite3.OperationalError:
        pass

    conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        status()
    else:
        init_db()
