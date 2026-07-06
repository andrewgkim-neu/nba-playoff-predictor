"""Step 2 of the pipeline: populate ``team_game_stats``.

For every game in the ``games`` table that doesn't yet have a row in
``team_game_stats``, fetch three box-score endpoints and combine them into
one wide row per team per game:

  - BoxScoreTraditionalV3  -> traditional box (pts, fg, reb, ast, stl, blk, tov...)
  - BoxScoreAdvancedV3     -> off/def/net rating, pace, ts%, efg%, ...
  - BoxScoreFourFactorsV3  -> fta_rate plus opponent-side four factors

NOTE: the older V2 endpoints were deprecated and stopped publishing data as of
the 2025-26 season (they now return payloads with no result sets, which surfaces
as a ``'resultSet'`` KeyError). This module uses the V3 endpoints, which return
a *nested* shape rather than V2's flat result-set rows:

    {"boxScoreTraditional": {"homeTeam": {"teamId": ..., "statistics": {...}},
                             "awayTeam": {"teamId": ..., "statistics": {...}}}}

Field names are camelCase (``points``, ``reboundsOffensive``) instead of V2's
flat ``PTS``/``OREB``. We map them back to the same ``team_game_stats`` columns
so nothing downstream (build_features, the model) has to change.

Idempotent and resumable: skips games already in ``team_game_stats`` and
serves cached responses from disk when available.

Run it:
    python -m src.fetch_boxscores
"""

import logging
import re
from datetime import datetime, timezone

from tqdm import tqdm

from nba_api.stats.endpoints import (
    BoxScoreTraditionalV3,
    BoxScoreAdvancedV3,
    BoxScoreFourFactorsV3,
)

from src import config
from src.api_client import NBAApiClient
from src.db import get_conn, init_db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def _fetch_boxscore(client, game_id: str, endpoint_class, endpoint_name: str) -> dict:
    def _do_fetch():
        ep = endpoint_class(game_id=game_id, timeout=config.REQUEST_TIMEOUT)
        # V3 endpoints: get_dict() returns the nested {"boxScore*": {...}} payload.
        return ep.get_dict()

    return client.fetch(
        endpoint_name=endpoint_name,
        fetcher_fn=_do_fetch,
        params={"game_id": game_id},
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _teams(payload: dict) -> tuple[dict | None, dict | None]:
    """Pull (homeTeam, awayTeam) objects out of a V3 box-score payload.

    The top-level key differs per endpoint (``boxScoreTraditional`` /
    ``boxScoreAdvanced`` / ``boxScoreFourFactors``), so we find the inner
    object that has both ``homeTeam`` and ``awayTeam`` rather than hard-coding
    the name.
    """
    if not isinstance(payload, dict):
        return None, None
    for value in payload.values():
        if isinstance(value, dict) and "homeTeam" in value and "awayTeam" in value:
            return value.get("homeTeam"), value.get("awayTeam")
    return None, None


def _first(stats: dict, *keys):
    """Return the first present, non-null value among candidate stat keys."""
    for k in keys:
        v = stats.get(k)
        if v is not None:
            return v
    return None


def _parse_minutes(value) -> float | None:
    """Minutes come as ISO-8601 'PT240M00.00S', 'MM:SS', or a plain number."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value)
    if s.startswith("PT"):  # ISO-8601 duration, e.g. 'PT240M00.00S'
        m = re.match(r"PT(?:(\d+)M)?(?:([\d.]+)S)?", s)
        if m:
            mins = float(m.group(1) or 0)
            secs = float(m.group(2) or 0)
            return mins + secs / 60.0
        return None
    if ":" in s:
        mins, secs = s.split(":", 1)
        try:
            return float(mins) + float(secs) / 60.0
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _team_row(trad_team, adv_team, ff_team, game_id, is_home, fetched_at) -> dict:
    """Merge the three endpoints' stats for one team into a team_game_stats row."""
    t = (trad_team or {}).get("statistics", {}) or {}
    a = (adv_team or {}).get("statistics", {}) or {}
    f = (ff_team or {}).get("statistics", {}) or {}

    team_id = (trad_team or {}).get("teamId") or (adv_team or {}).get("teamId")

    return {
        "game_id":  str(game_id),
        "team_id":  int(team_id),
        "is_home":  1 if is_home else 0,

        # Traditional
        "min":        _parse_minutes(_first(t, "minutes")),
        "pts":        _first(t, "points"),
        "fgm":        _first(t, "fieldGoalsMade"),       "fga": _first(t, "fieldGoalsAttempted"),
        "fg_pct":     _first(t, "fieldGoalsPercentage"),
        "fg3m":       _first(t, "threePointersMade"),    "fg3a": _first(t, "threePointersAttempted"),
        "fg3_pct":    _first(t, "threePointersPercentage"),
        "ftm":        _first(t, "freeThrowsMade"),        "fta": _first(t, "freeThrowsAttempted"),
        "ft_pct":     _first(t, "freeThrowsPercentage"),
        "oreb":       _first(t, "reboundsOffensive"),     "dreb": _first(t, "reboundsDefensive"),
        "reb":        _first(t, "reboundsTotal"),
        "ast":        _first(t, "assists"),
        "stl":        _first(t, "steals"),
        "blk":        _first(t, "blocks"),
        "tov":        _first(t, "turnovers"),
        "pf":         _first(t, "foulsPersonal"),
        "plus_minus": _first(t, "plusMinusPoints"),

        # Advanced
        "off_rating": _first(a, "offensiveRating"),
        "def_rating": _first(a, "defensiveRating"),
        "net_rating": _first(a, "netRating"),
        "ast_pct":    _first(a, "assistPercentage"),
        "ast_to":     _first(a, "assistToTurnover"),
        "ast_ratio":  _first(a, "assistRatio"),
        "oreb_pct":   _first(a, "offensiveReboundPercentage"),
        "dreb_pct":   _first(a, "defensiveReboundPercentage"),
        "reb_pct":    _first(a, "reboundPercentage"),
        # Sourced from four factors (0-1 fraction) to match opp_tov_pct's scale
        # and the old V2 TM_TOV_PCT semantics. Advanced's `turnoverRatio` is a
        # different per-100-possessions stat, so we don't use it here.
        "tov_pct":    _first(f, "teamTurnoverPercentage"),
        "efg_pct":    _first(a, "effectiveFieldGoalPercentage"),
        "ts_pct":     _first(a, "trueShootingPercentage"),
        "pace":       _first(a, "pace"),
        "pie":        _first(a, "PIE", "pie"),

        # Four Factors (own EFG/TOV duplicated from advanced; opp_* are unique here)
        "fta_rate":     _first(f, "freeThrowAttemptRate"),
        "opp_efg_pct":  _first(f, "oppEffectiveFieldGoalPercentage"),
        "opp_fta_rate": _first(f, "oppFreeThrowAttemptRate"),
        "opp_tov_pct":  _first(f, "oppTeamTurnoverPercentage"),
        "opp_oreb_pct": _first(f, "oppOffensiveReboundPercentage"),

        "fetched_at": fetched_at,
    }


def parse_team_stats(
    traditional: dict,
    advanced: dict,
    four_factors: dict,
    game_id: str,
    home_team_id: int | None,
) -> list[dict]:
    """Merge the three V3 endpoint payloads into team_game_stats rows.

    V3 tells us directly which team is home vs away (separate ``homeTeam`` /
    ``awayTeam`` objects), so we set ``is_home`` from position rather than by
    comparing IDs. ``home_team_id`` is accepted for API compatibility and used
    only as a sanity check.
    """
    trad_home, trad_away = _teams(traditional)
    adv_home, adv_away = _teams(advanced)
    ff_home, ff_away = _teams(four_factors)

    if trad_home is None or trad_away is None:
        return []

    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = [
        _team_row(trad_home, adv_home, ff_home, game_id, True, fetched_at),
        _team_row(trad_away, adv_away, ff_away, game_id, False, fetched_at),
    ]

    if home_team_id is not None and rows[0]["team_id"] != int(home_team_id):
        logger.warning(
            "Game %s: V3 homeTeam id %s != games.home_team_id %s",
            game_id, rows[0]["team_id"], home_team_id,
        )
    return rows


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def games_needing_boxscores(conn) -> list[tuple[str, int]]:
    """Return (game_id, home_team_id) for games with no rows in team_game_stats yet."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT g.game_id, g.home_team_id
        FROM games g
        LEFT JOIN team_game_stats t ON t.game_id = g.game_id
        WHERE t.game_id IS NULL
        ORDER BY g.game_date
        """
    )
    return cur.fetchall()


def upsert_team_stats(conn, rows: list[dict]) -> int:
    if not rows:
        return 0
    cols = list(rows[0].keys())
    placeholders = ", ".join(f":{c}" for c in cols)
    col_list = ", ".join(cols)
    cur = conn.cursor()
    cur.executemany(
        f"INSERT OR REPLACE INTO team_game_stats ({col_list}) VALUES ({placeholders})",
        rows,
    )
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    init_db()
    client = NBAApiClient()
    conn = get_conn()

    pending = games_needing_boxscores(conn)
    logger.info("%d games need box scores", len(pending))

    total = 0
    failures = 0
    try:
        for game_id, home_team_id in tqdm(pending, desc="Box scores"):
            try:
                trad = _fetch_boxscore(client, game_id, BoxScoreTraditionalV3, "BoxScoreTraditionalV3")
                adv  = _fetch_boxscore(client, game_id, BoxScoreAdvancedV3,    "BoxScoreAdvancedV3")
                ff   = _fetch_boxscore(client, game_id, BoxScoreFourFactorsV3, "BoxScoreFourFactorsV3")

                rows = parse_team_stats(trad, adv, ff, game_id, home_team_id)
                if not rows:
                    logger.warning("No team rows parsed for %s", game_id)
                    continue
                upsert_team_stats(conn, rows)
                total += len(rows)
            except Exception as e:
                logger.error("Failed %s: %s", game_id, e)
                failures += 1
    finally:
        conn.close()

    logger.info("Done. Wrote %d team-game rows (%d game failures)", total, failures)


if __name__ == "__main__":
    main()
