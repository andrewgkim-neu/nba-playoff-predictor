"""Step 3 of the pipeline: backfill playoff context into the ``games`` table.

For each season, calls ``CommonPlayoffSeries`` and updates the existing
``games`` rows with:

  - ``series_id``         (the series this game belongs to)
  - ``game_num_in_series``(1..7)
  - ``playoff_round``     (1=first round, 2=conference semis, 3=conference
                           finals, 4=NBA Finals)

The endpoint doesn't return round number directly, so we derive it from
the chronological ordering of series start dates: in any modern NBA
playoffs there are exactly 15 series in a 16-team bracket (8 + 4 + 2 + 1),
and round N series always start later than round N-1 series. We sort
series by their first game date and bucket them accordingly.

Idempotent: re-running re-applies the same updates. Resumable: previously
updated rows just get the same values written again.

Run it:
    python -m src.fetch_playoff_series
"""

import logging
from collections import defaultdict

from tqdm import tqdm
from nba_api.stats.endpoints import CommonPlayoffSeries

from src import config
from src.api_client import NBAApiClient
from src.db import get_conn

logger = logging.getLogger(__name__)


EXPECTED_SERIES_PER_ROUND = [8, 4, 2, 1]   # round 1, 2, 3, 4
TOTAL_SERIES = sum(EXPECTED_SERIES_PER_ROUND)


def fetch_playoff_series(client: NBAApiClient, season: str) -> dict:
    params = {"season": season}

    def _do_fetch():
        ep = CommonPlayoffSeries(season=season, timeout=config.REQUEST_TIMEOUT)
        return ep.get_normalized_dict()

    return client.fetch(
        endpoint_name="CommonPlayoffSeries",
        fetcher_fn=_do_fetch,
        params=params,
    )


def _extract_rows(raw: dict) -> list:
    """The result set name has varied across nba_api versions."""
    for key in ("PlayoffSeries", "CommonPlayoffSeries"):
        if key in raw and raw[key]:
            return raw[key]
    return []


def assign_rounds(series_first_dates: dict) -> dict:
    """Map series_id -> round number (1..4).

    Args:
        series_first_dates: {series_id: earliest_game_date_iso}.

    Returns:
        {series_id: round_number} for the modern 16-team bracket.
    """
    # Sort series by chronological start date.
    sorted_series = sorted(series_first_dates.items(), key=lambda kv: kv[1])

    if len(sorted_series) != TOTAL_SERIES:
        logger.warning(
            "Expected %d playoff series, got %d. Round assignment may be off.",
            TOTAL_SERIES, len(sorted_series),
        )

    series_round = {}
    # Cumulative bracket positions: round 1 = series 0..7, round 2 = 8..11, etc.
    cutoffs = []
    cumulative = 0
    for round_num, count in enumerate(EXPECTED_SERIES_PER_ROUND, start=1):
        cumulative += count
        cutoffs.append((cumulative, round_num))

    for idx, (sid, _) in enumerate(sorted_series):
        round_num = None
        for cutoff, rnd in cutoffs:
            if idx < cutoff:
                round_num = rnd
                break
        if round_num is None:
            # More series than expected; assign to the final round bucket.
            round_num = EXPECTED_SERIES_PER_ROUND.__len__()
        series_round[sid] = round_num
    return series_round


def update_games(conn, season: str, series_rows: list, series_round_map: dict) -> int:
    """Apply per-game updates. ``series_rows`` are the raw CommonPlayoffSeries rows."""
    updates = []
    for row in series_rows:
        sid = str(row["SERIES_ID"])
        updates.append({
            "playoff_round":      series_round_map.get(sid),
            "series_id":          sid,
            "game_num_in_series": int(row["GAME_NUM"]),
            "game_id":            str(row["GAME_ID"]),
        })

    if not updates:
        return 0

    cur = conn.cursor()
    cur.executemany(
        """
        UPDATE games
           SET playoff_round = :playoff_round,
               series_id = :series_id,
               game_num_in_series = :game_num_in_series
         WHERE game_id = :game_id
        """,
        updates,
    )
    conn.commit()
    return cur.rowcount


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    client = NBAApiClient()
    conn = get_conn()

    try:
        # Pull each game's date into a lookup so we can compute series start dates.
        cur = conn.cursor()
        cur.execute("SELECT game_id, game_date FROM games")
        game_date_lookup = {gid: gdate for gid, gdate in cur.fetchall()}

        for season in tqdm(config.SEASONS, desc="Seasons"):
            try:
                raw = fetch_playoff_series(client, season)
                rows = _extract_rows(raw)
                if not rows:
                    logger.warning("No playoff series found for %s", season)
                    continue

                # Group by series; find each series' earliest game date.
                series_first_dates = {}
                for r in rows:
                    sid = str(r["SERIES_ID"])
                    gid = str(r["GAME_ID"])
                    gdate = game_date_lookup.get(gid)
                    if gdate is None:
                        continue
                    if sid not in series_first_dates or gdate < series_first_dates[sid]:
                        series_first_dates[sid] = gdate

                series_round_map = assign_rounds(series_first_dates)

                n = update_games(conn, season, rows, series_round_map)
                logger.info("%s: updated %d playoff games across %d series",
                            season, n, len(series_first_dates))
            except Exception as e:
                logger.error("Failed %s: %s", season, e)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
