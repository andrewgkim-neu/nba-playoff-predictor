"""Build the ``player_game_stats`` table from cached box-score JSON.

The V3 box-score responses we already fetched carry per-player rows (in each
team's ``players`` array) — both traditional stats and per-player advanced
ratings. This module re-reads those cached files (no network) and writes one
row per player-game for players who actually appeared (minutes > 0).

We pull from two cached endpoints per game and join on personId:
  - BoxScoreTraditionalV3 -> minutes, points, plus/minus
  - BoxScoreAdvancedV3    -> on-court off/def/net rating

Run it (after fetch_boxscores has populated the cache):
    python -m src.build_player_stats
"""

import hashlib
import json
import logging

from tqdm import tqdm

from src import config
from src.db import get_conn
from src.fetch_boxscores import _parse_minutes

logger = logging.getLogger(__name__)


# Full rebuild from cache each run (cheap), so the schema is always current.
SCHEMA = """
DROP TABLE IF EXISTS player_game_stats;
CREATE TABLE player_game_stats (
    game_id     TEXT    NOT NULL,
    season      TEXT,
    season_type TEXT,
    team_id     INTEGER NOT NULL,
    player_id   INTEGER NOT NULL,
    player_name TEXT,
    is_home     INTEGER NOT NULL,
    minutes     REAL,
    pts         INTEGER,
    plus_minus  REAL,
    off_rating  REAL,
    def_rating  REAL,
    net_rating  REAL,
    PRIMARY KEY (game_id, player_id)
);
CREATE INDEX idx_pgs_player ON player_game_stats(player_id);
CREATE INDEX idx_pgs_game   ON player_game_stats(game_id);
CREATE INDEX idx_pgs_type   ON player_game_stats(season, season_type);
"""


def _cache_path(endpoint: str, game_id: str):
    """Mirror api_client's cache key: md5 of the sorted params dict, first 16 hex."""
    payload = json.dumps({"game_id": game_id}, sort_keys=True, default=str)
    key = hashlib.md5(payload.encode()).hexdigest()[:16]
    return config.RAW_DIR / endpoint / f"{key}.json"


def _load_cached(endpoint: str, game_id: str):
    path = _cache_path(endpoint, game_id)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _teams(payload, top_key):
    bs = (payload or {}).get(top_key, {})
    return bs.get("homeTeam"), bs.get("awayTeam")


def parse_game(game_id: str, trad: dict, adv: dict) -> list[dict]:
    """One row per player who logged minutes, traditional joined to advanced."""
    trad_home, trad_away = _teams(trad, "boxScoreTraditional")
    adv_home, adv_away = _teams(adv, "boxScoreAdvanced")

    rows = []
    for team_obj, adv_obj, is_home in [(trad_home, adv_home, 1), (trad_away, adv_away, 0)]:
        if not team_obj:
            continue
        team_id = team_obj.get("teamId")
        adv_by_id = {
            p.get("personId"): p.get("statistics", {})
            for p in (adv_obj or {}).get("players", [])
        }
        for p in team_obj.get("players", []):
            st = p.get("statistics", {}) or {}
            mins = _parse_minutes(st.get("minutes"))
            if not mins or mins <= 0:
                continue  # DNP — not part of the active rotation
            a = adv_by_id.get(p.get("personId"), {})
            name = (f"{p.get('firstName', '')} {p.get('familyName', '')}").strip() or p.get("nameI")
            rows.append({
                "game_id": str(game_id),
                "team_id": int(team_id),
                "player_id": int(p.get("personId")),
                "player_name": name,
                "is_home": is_home,
                "minutes": mins,
                "pts": st.get("points"),
                "plus_minus": st.get("plusMinusPoints"),
                "off_rating": a.get("offensiveRating"),
                "def_rating": a.get("defensiveRating"),
                "net_rating": a.get("netRating"),
            })
    return rows


def upsert(conn, rows: list[dict]) -> int:
    if not rows:
        return 0
    cols = list(rows[0].keys())
    placeholders = ", ".join(f":{c}" for c in cols)
    conn.executemany(
        f"INSERT OR REPLACE INTO player_game_stats ({', '.join(cols)}) VALUES ({placeholders})",
        rows,
    )
    conn.commit()
    return len(rows)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    conn = get_conn()
    conn.executescript(SCHEMA)

    meta = {
        gid: (season, stype)
        for gid, season, stype in conn.execute("SELECT game_id, season, season_type FROM games")
    }
    logger.info("%d games to parse from cache", len(meta))

    total, missing = 0, 0
    try:
        for gid in tqdm(list(meta), desc="Players"):
            trad = _load_cached("BoxScoreTraditionalV3", gid)
            adv = _load_cached("BoxScoreAdvancedV3", gid)
            if not trad:
                missing += 1
                continue
            rows = parse_game(gid, trad, adv or {})
            season, stype = meta[gid]
            for r in rows:
                r["season"], r["season_type"] = season, stype
            total += upsert(conn, rows)
    finally:
        conn.close()

    logger.info("Wrote %d player-game rows (%d games missing from cache)", total, missing)


if __name__ == "__main__":
    main()
