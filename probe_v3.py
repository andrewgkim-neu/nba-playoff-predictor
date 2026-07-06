"""One-game V3 box-score probe. Run BEFORE the full fetch to confirm the V3
field mapping in src/fetch_boxscores.py is correct.

    python probe_v3.py

It fetches a single game with the V3 endpoints, prints the real stat field
names returned by each endpoint, then runs them through parse_team_stats and
prints the resulting rows. If the mapping is right, the printed rows should
have real numbers (not None) for pts / off_rating / fta_rate / opp_efg_pct etc.

Throwaway diagnostic — delete it once the fetch is validated.
"""

import json
import sqlite3

from nba_api.stats.endpoints import (
    BoxScoreTraditionalV3,
    BoxScoreAdvancedV3,
    BoxScoreFourFactorsV3,
)

from src import config, fetch_boxscores as fb

conn = sqlite3.connect(config.DB_PATH)
row = conn.execute(
    "SELECT game_id, home_team_id FROM games ORDER BY game_date LIMIT 1"
).fetchone()
conn.close()
if not row:
    raise SystemExit("No games in DB. Run `python -m src.fetch_games` first.")

game_id, home_team_id = row
print(f"Probing game_id={game_id} (home_team_id={home_team_id})\n")

trad = BoxScoreTraditionalV3(game_id=game_id, timeout=config.REQUEST_TIMEOUT).get_dict()
adv  = BoxScoreAdvancedV3(game_id=game_id,    timeout=config.REQUEST_TIMEOUT).get_dict()
ff   = BoxScoreFourFactorsV3(game_id=game_id, timeout=config.REQUEST_TIMEOUT).get_dict()

for label, payload in [("TRADITIONAL", trad), ("ADVANCED", adv), ("FOUR FACTORS", ff)]:
    print(f"===== {label}: top-level keys = {list(payload.keys())}")
    home, _ = fb._teams(payload)
    stats = (home or {}).get("statistics", {}) or {}
    print(f"  homeTeam.teamId = {(home or {}).get('teamId')}")
    print(f"  statistics field names: {sorted(stats.keys())}")
    print(f"  statistics values:\n{json.dumps(stats, indent=2)[:1800]}\n")

print("===== PARSED ROWS (None values mean a field name didn't match) =====")
rows = fb.parse_team_stats(trad, adv, ff, game_id, home_team_id)
for r in rows:
    print(json.dumps(r, indent=2, default=str))
    nones = [k for k, v in r.items() if v is None]
    print(f"  -> NULL columns: {nones}\n")
