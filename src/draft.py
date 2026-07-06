"""Draft-a-team game.

For a chosen year, every eligible player gets a value rating and a salary cost.
You draft a roster under a fixed cap, and the team is scored two ways:

  - **Rating**: a realistic minutes allocation (each player up to his historical
    minutes, capping the team at 240 player-minutes; any shortfall is filled at
    replacement level), giving a minutes-weighted on-court net rating.
  - **Sim**: that team rating is turned into a per-game win probability and run
    against the chosen year's real teams — an expected record, plus a Monte
    Carlo playoff bracket for championship odds.

Player value can be based on **regular season**, **playoffs**, or a **blended**
body of work (--basis), so you can reward postseason performers specifically.

Commands:
    python -m src.draft pool   --year 2023-24 --basis playoffs
    python -m src.draft score  --year 2023-24 --basis regular --budget 100 \
        --players "Nikola Jokic, Jamal Murray, Aaron Gordon, ..."
    python -m src.draft play   --year 2023-24 --basis regular --budget 100
"""

import argparse
import logging
import unicodedata

import numpy as np
import pandas as pd

from src.db import get_conn

logger = logging.getLogger(__name__)

# --- Game economy / scoring constants ---
TEAM_MINUTES = 240.0          # 5 players * 48 minutes
MAX_PLAYER_MIN = 42.0         # nobody realistically plays more
REPLACEMENT_NR = -6.0         # net rating of the filler when a roster is too thin
DEFAULT_CAP = 100
ROSTER_MAX = 15

# Value weights (z-scored within the pool) and eligibility filters.
VALUE_WEIGHTS = {"net_rating": 0.40, "ppg": 0.35, "mpg": 0.25}
MIN_GP = {"regular": 20, "playoffs": 3, "blended": 20}
MIN_MPG = 12.0

SEASON_TYPE = {"regular": "Regular Season", "playoffs": "Playoffs", "blended": "blended"}


# ---------------------------------------------------------------------------
# Player pool: value + cost
# ---------------------------------------------------------------------------

def build_pool(conn, year: str, basis: str) -> pd.DataFrame:
    stype = SEASON_TYPE[basis]
    psp = pd.read_sql(
        "SELECT * FROM player_season_profiles WHERE season = ?", conn, params=(year,)
    )
    if psp.empty:
        raise SystemExit(f"No player profiles for {year}. Available years: "
                         f"{sorted(pd.read_sql('SELECT DISTINCT season FROM player_season_profiles', conn)['season'])}")

    if basis == "blended":
        # Minutes/production blended across RS + PO, weighted by games played.
        def blend(g):
            w = g["gp"]
            tot = w.sum()
            out = {"player_name": g["player_name"].iloc[0], "gp": int(tot),
                   "team_id": int(g.sort_values("gp").iloc[-1]["team_id"])}
            for m in ["mpg", "ppg", "net_rating", "off_rating", "def_rating"]:
                out[m] = float((g[m] * w).sum() / tot) if tot else float("nan")
            return pd.Series(out)
        pool = psp.groupby("player_id").apply(blend, include_groups=False).reset_index()
    else:
        pool = psp[psp["season_type"] == stype].copy()

    pool = pool[(pool["gp"] >= MIN_GP[basis]) & (pool["mpg"] >= MIN_MPG)].copy()
    if pool.empty:
        raise SystemExit(f"No eligible players for {year} / {basis}.")

    # Value = weighted blend of z-scored impact, scoring, and role.
    def z(s):
        sd = s.std(ddof=0)
        return (s - s.mean()) / sd if sd else s * 0.0
    pool["value"] = sum(w * z(pool[m]) for m, w in VALUE_WEIGHTS.items())
    pool["ovr"] = (60 + 12 * pool["value"]).clip(40, 99).round().astype(int)

    # Cost: convex in value percentile so stars are disproportionately pricey.
    pct = pool["value"].rank(pct=True)
    pool["cost"] = (1 + 49 * pct ** 2).round().astype(int)

    return pool.sort_values("value", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Snake draft (3rd-round reversal)
# ---------------------------------------------------------------------------

def snake_order(n_teams: int, n_rounds: int) -> list[int]:
    """Seat order across all picks under 3rd-round reversal (seats are 0-indexed).

    Round 1: forward (0..N-1). Round 2: reverse (N-1..0). Round 3: reverse AGAIN
    (the reversal). Round 4 onward: normal snake. Concretely the per-round
    direction is forward, reverse, reverse, forward, reverse, forward, ...
    """
    order = []
    for r in range(1, n_rounds + 1):
        seats = list(range(n_teams))
        if r == 1:
            seq = seats                                   # forward
        elif r == 2:
            seq = seats[::-1]                             # reverse (normal snake)
        else:
            seq = seats[::-1] if r % 2 == 1 else seats    # R3 reversed, then alternate
        order.extend(seq)
    return order


def best_available(pool: pd.DataFrame, taken: set) -> pd.Series:
    """The highest-value undrafted player (the AI's 'best player available')."""
    avail = pool[~pool["player_id"].isin(taken)]
    return avail.iloc[0]


def score_rosters(pool: pd.DataFrame, rosters: dict) -> pd.DataFrame:
    """Team rating per seat, ranked best-first."""
    rows = []
    for seat, ids in rosters.items():
        picks = pool[pool["player_id"].isin(ids)]
        rows.append({"seat": seat, "net_rating": team_rating(picks)["net_rating"] if len(picks) else 0.0})
    out = pd.DataFrame(rows).sort_values("net_rating", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out


def autodraft(pool: pd.DataFrame, n_teams: int, n_rounds: int,
              your_seat: int = None, your_picks: list = None) -> dict:
    """Run a full snake draft. AI seats take best-available; your_seat takes from
    your_picks (by player name) in order, falling back to best-available."""
    pool = pool.sort_values("value", ascending=False).reset_index(drop=True)
    taken, rosters = set(), {s: [] for s in range(n_teams)}
    queue = list(your_picks or [])
    for seat in snake_order(n_teams, n_rounds):
        if seat == your_seat and queue:
            want = _norm(queue.pop(0))
            hit = pool[(pool["player_name"].map(_norm) == want) & (~pool["player_id"].isin(taken))]
            pick = hit.iloc[0] if len(hit) else best_available(pool, taken)
        else:
            pick = best_available(pool, taken)
        taken.add(int(pick["player_id"]))
        rosters[seat].append(int(pick["player_id"]))
    return rosters


# ---------------------------------------------------------------------------
# Team rating (realistic minutes allocation)
# ---------------------------------------------------------------------------

def team_rating(picks: pd.DataFrame) -> dict:
    """Allocate up to 240 minutes by net rating; fill any shortfall at replacement."""
    p = picks.sort_values("net_rating", ascending=False)
    mins = np.minimum(p["mpg"].to_numpy(), MAX_PLAYER_MIN)
    total = mins.sum()
    if total >= TEAM_MINUTES:
        alloc = mins * (TEAM_MINUTES / total)
        rep = 0.0
    else:
        alloc = mins
        rep = TEAM_MINUTES - total

    def weighted(metric, rep_val):
        return (float((alloc * p[metric].to_numpy()).sum()) + rep * rep_val) / TEAM_MINUTES

    return {
        "net_rating": weighted("net_rating", REPLACEMENT_NR),
        "off_rating": weighted("off_rating", 108.0),   # replacement ~ below-avg offense
        "def_rating": weighted("def_rating", 115.0),   # replacement ~ below-avg defense
        "replacement_minutes": rep,
        "n_players": len(p),
    }


# ---------------------------------------------------------------------------
# Simulation: net-rating-diff -> win probability, record, title odds
# ---------------------------------------------------------------------------

def fit_winprob(conn) -> tuple[float, float]:
    """Logistic home_won ~ diff_net_rating_rs. Returns (intercept=home edge, slope)."""
    from sklearn.linear_model import LogisticRegression
    gf = pd.read_sql(
        "SELECT home_won, diff_net_rating_rs FROM game_features "
        "WHERE home_won IS NOT NULL AND diff_net_rating_rs IS NOT NULL", conn
    )
    clf = LogisticRegression()
    clf.fit(gf[["diff_net_rating_rs"]], gf["home_won"].astype(int))
    return float(clf.intercept_[0]), float(clf.coef_[0][0])


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def field_net_ratings(conn, year: str) -> pd.DataFrame:
    """Each real team's regular-season net rating that year (abbr, net_rating)."""
    df = pd.read_sql(
        """
        SELECT g.home_team_abbr AS abbr, t.team_id, t.net_rating
        FROM team_game_stats t JOIN games g ON g.game_id = t.game_id
        WHERE g.season = ? AND g.season_type = 'Regular Season' AND t.is_home = 1
        UNION ALL
        SELECT g.away_team_abbr AS abbr, t.team_id, t.net_rating
        FROM team_game_stats t JOIN games g ON g.game_id = t.game_id
        WHERE g.season = ? AND g.season_type = 'Regular Season' AND t.is_home = 0
        """,
        conn, params=(year, year),
    )
    return df.groupby(["team_id", "abbr"], as_index=False)["net_rating"].mean()


GAMES_PER_TEAM = 82
PLAYOFF_TEAMS = 16
ROUND_NAMES = {16: "Round 1", 8: "Round 2", 4: "Semifinals", 2: "Finals"}


def adjust_field(field: pd.DataFrame, picks: pd.DataFrame = None) -> pd.DataFrame:
    """Model every opponent at league-average strength.

    Your drafted players came out of the real league, so it's not fair to also
    make you beat the stars you took. Instead of carrying each real team's actual
    (player-inflated) net rating, we set every opponent to the league-average net
    rating — your drafted team is measured against an average league, so a strong
    roster is rewarded with the title odds it deserves. (`picks` is accepted for
    interface stability but isn't needed under this model.)
    """
    field = field.copy()
    field["net_rating"] = float(field["net_rating"].mean())
    return field


def _league(your_nr, field) -> tuple[list[str], "np.ndarray"]:
    """Your team plus the year's real teams: (names, net_ratings)."""
    names = ["My Team"] + list(field["abbr"])
    nrs = np.array([your_nr] + list(field["net_rating"]), dtype=float)
    return names, nrs


def _season_wins(nrs, b0, b1, rng, games_per_team=GAMES_PER_TEAM):
    """Simulate a balanced schedule (each team plays ~games_per_team); return wins/played."""
    n = len(nrs)
    # Each team appears exactly games_per_team times, then we pair the slots up,
    # so every team plays an equal-length schedule.
    slots = np.repeat(np.arange(n), games_per_team)
    rng.shuffle(slots)
    if len(slots) % 2:                       # drop one slot if odd so pairs are even
        slots = slots[:-1]
    home, away = slots[0::2].copy(), slots[1::2].copy()

    # Resolve self-matches by swapping the away assignment with a compatible game
    # (preserves each team's game count exactly).
    for i in np.where(home == away)[0]:
        for j in range(len(away)):
            if away[j] != home[i] and home[j] != home[i]:
                away[i], away[j] = away[j], away[i]
                break

    p_home = _sigmoid(b0 + b1 * (nrs[home] - nrs[away]))
    home_win = rng.random(home.size) < p_home

    wins = np.zeros(n)
    played = np.zeros(n)
    np.add.at(wins, home, home_win)
    np.add.at(wins, away, ~home_win)
    np.add.at(played, home, 1)
    np.add.at(played, away, 1)
    return wins, played


def _bracket(seed_nrs, seed_names, b0, b1, rng):
    """Single-elimination best-of-7 bracket, seeded best-first. Returns (rounds, champion)."""
    idx = list(range(len(seed_nrs)))
    rounds = []
    while len(idx) > 1:
        results = []
        nxt = []
        for i in range(len(idx) // 2):
            hi, lo = idx[i], idx[len(idx) - 1 - i]          # 1v16, 2v15, ...
            hi_wins = _sim_series(seed_nrs[hi], seed_nrs[lo], b0, b1, rng)  # higher seed gets home court
            winner = hi if hi_wins else lo
            results.append({"high": seed_names[hi], "low": seed_names[lo], "winner": seed_names[winner]})
            nxt.append(winner)
        rounds.append((len(idx), results))
        idx = sorted(nxt)                                    # reseed by original seed
    return rounds, seed_names[idx[0]]


def run_season(your_nr, field, b0, b1, seed=7) -> dict:
    """One full regular season + playoff bracket. Reproducible for a given seed."""
    rng = np.random.default_rng(seed)
    names, nrs = _league(your_nr, field)

    wins, played = _season_wins(nrs, b0, b1, rng)
    standings = pd.DataFrame({"team": names, "net_rating": nrs,
                              "wins": wins.astype(int), "losses": (played - wins).astype(int)})
    standings["win_pct"] = standings["wins"] / standings[["wins", "losses"]].sum(axis=1).clip(lower=1)
    standings = standings.sort_values("win_pct", ascending=False).reset_index(drop=True)
    standings["seed"] = standings.index + 1

    me = standings[standings["team"] == "My Team"].iloc[0]
    made = int(me["seed"]) <= PLAYOFF_TEAMS

    rounds, champion = ([], None)
    if made:
        top = standings.head(PLAYOFF_TEAMS)
        rounds, champion = _bracket(top["net_rating"].tolist(), top["team"].tolist(), b0, b1, rng)

    return {
        "standings": standings,
        "record": (int(me["wins"]), int(me["losses"])),
        "seed": int(me["seed"]),
        "made_playoffs": made,
        "rounds": rounds,
        "champion": champion,
        "won_title": champion == "My Team",
    }


def my_playoff_path(result: dict) -> list[dict]:
    """Round-by-round results involving your team, until eliminated or champion."""
    path = []
    for size, results in result["rounds"]:
        series = next((s for s in results if "My Team" in (s["high"], s["low"])), None)
        if series is None:
            break
        opp = series["low"] if series["high"] == "My Team" else series["high"]
        won = series["winner"] == "My Team"
        path.append({"round": ROUND_NAMES.get(size, f"Round of {size}"), "opponent": opp, "won": won})
        if not won:
            break
    return path


def championship_odds(your_nr, field, b0, b1, n_sims=300, base_seed=1000) -> float:
    """Title probability over many full season+playoff simulations."""
    names, nrs = _league(your_nr, field)
    titles = 0
    for k in range(n_sims):
        rng = np.random.default_rng(base_seed + k)
        wins, played = _season_wins(nrs, b0, b1, rng)
        win_pct = wins / np.clip(played, 1, None)
        order = np.argsort(-win_pct)[:PLAYOFF_TEAMS]
        if 0 not in order:          # index 0 is "My Team"; missed the playoffs
            continue
        _, champ = _bracket(nrs[order].tolist(), [names[i] for i in order], b0, b1, rng)
        titles += champ == "My Team"
    return titles / n_sims


def _series_winprob_high(nr_high, nr_low, b0, b1) -> tuple[float, float]:
    """Per-game P(higher seed wins) at home and on the road."""
    p_home = _sigmoid(b0 + b1 * (nr_high - nr_low))
    p_road = _sigmoid(-b0 + b1 * (nr_high - nr_low))
    return p_home, p_road


HOME_PATTERN = [True, True, False, False, True, False, True]  # 2-2-1-1-1 for higher seed


def _sim_series(nr_a, nr_b, b0, b1, rng) -> bool:
    """True if team A (treated as higher seed / home-court) wins the best-of-7."""
    hi, lo = (nr_a, nr_b)
    p_home, p_road = _series_winprob_high(hi, lo, b0, b1)
    a = bbb = 0
    for at_home in HOME_PATTERN:
        p = p_home if at_home else p_road
        if rng.random() < p:
            a += 1
        else:
            bbb += 1
        if a == 4 or bbb == 4:
            break
    return a == 4


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    """Lowercase and strip accents so 'Jokic' matches 'Jokić'."""
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()


def _match_players(pool: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    norm_names = pool["player_name"].map(_norm)
    picks, missing = [], []
    for raw in names:
        q = _norm(raw)
        hit = pool[norm_names == q]
        if hit.empty:
            hit = pool[norm_names.str.contains(q, regex=False)]
        if hit.empty:
            missing.append(raw)
        else:
            picks.append(hit.iloc[0])
    if missing:
        raise SystemExit(f"Couldn't find (in this year/basis pool): {missing}")
    return pd.DataFrame(picks)


def report_team(conn, pool, picks, budget, b0, b1, sim=True):
    spent = int(picks["cost"].sum())
    print(f"\nRoster ({len(picks)} players) — spent {spent}/{budget}"
          + ("  OVER CAP!" if spent > budget else ""))
    for _, p in picks.sort_values("cost", ascending=False).iterrows():
        print(f"  ${p['cost']:>2}  OVR {p['ovr']:>2}  {p['player_name']:<24} "
              f"{p['mpg']:4.1f} mpg  {p['ppg']:4.1f} ppg  NR {p['net_rating']:+5.1f}")

    r = team_rating(picks)
    print(f"\n  Team rating: net {r['net_rating']:+.1f}  "
          f"(off {r['off_rating']:.1f} / def {r['def_rating']:.1f})")
    if r["replacement_minutes"] > 0:
        print(f"  ⚠ roster only covers {TEAM_MINUTES - r['replacement_minutes']:.0f}/240 minutes; "
              f"{r['replacement_minutes']:.0f} filled at replacement level")

    if sim and spent <= budget:
        field = adjust_field(field_net_ratings(conn, picks.attrs.get("year")), picks)
        res = run_season(r["net_rating"], field, b0, b1)
        w, l = res["record"]
        print(f"\n  Regular season: {w}-{l}  →  #{res['seed']} seed of {len(res['standings'])}")
        if not res["made_playoffs"]:
            print("  Missed the playoffs.")
        else:
            print("  Playoff run:")
            for leg in my_playoff_path(res):
                verdict = "beat" if leg["won"] else "lost to"
                print(f"    {leg['round']}: {verdict} {leg['opponent']}")
            if res["won_title"]:
                print("    🏆 CHAMPIONS!")
        odds = championship_odds(r["net_rating"], field, b0, b1)
        print(f"  Championship odds (over many sims): {odds:.1%}")
    if spent > budget:
        print("\n  NOTE: roster is over the salary cap — not a legal team (no sim).")
    print()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_pool(conn, year, basis, top):
    pool = build_pool(conn, year, basis)
    print(f"\n{year} player pool — basis: {basis}  ({len(pool)} eligible)\n")
    print(f"{'cost':>4}  {'OVR':>3}  {'player':<24}{'mpg':>6}{'ppg':>6}{'NR':>7}")
    for _, p in pool.head(top).iterrows():
        print(f"{p['cost']:>4}  {p['ovr']:>3}  {p['player_name']:<24}"
              f"{p['mpg']:>6.1f}{p['ppg']:>6.1f}{p['net_rating']:>7.1f}")
    print()


def cmd_score(conn, year, basis, budget, players):
    pool = build_pool(conn, year, basis)
    pool.attrs["year"] = year
    picks = _match_players(pool, players.split(","))
    picks.attrs["year"] = year
    b0, b1 = fit_winprob(conn)
    report_team(conn, pool, picks, budget, b0, b1, sim=True)


def cmd_play(conn, year, basis, budget):
    pool = build_pool(conn, year, basis)
    pool.attrs["year"] = year
    b0, b1 = fit_winprob(conn)
    chosen, spent = [], 0
    print(f"\n=== Draft your {year} team — basis: {basis}, cap: {budget} ===")
    print("Type a player name to draft, 'pool' to list affordable players, 'done' to finish.\n")
    while True:
        remaining = budget - spent
        try:
            cmd = input(f"[${remaining} left, {len(chosen)} drafted] > ").strip()
        except EOFError:
            break
        if not cmd:
            continue
        if cmd.lower() == "done":
            break
        if cmd.lower() == "pool":
            aff = pool[(pool["cost"] <= remaining) & (~pool["player_name"].isin(chosen))].head(20)
            for _, p in aff.iterrows():
                print(f"  ${p['cost']:>2}  OVR {p['ovr']:>2}  {p['player_name']:<24} NR {p['net_rating']:+5.1f}")
            continue
        try:
            pick = _match_players(pool, [cmd]).iloc[0]
        except SystemExit as e:
            print(f"  {e}")
            continue
        if pick["player_name"] in chosen:
            print("  already drafted.")
            continue
        if pick["cost"] > remaining:
            print(f"  can't afford {pick['player_name']} (${pick['cost']}, ${remaining} left).")
            continue
        if len(chosen) >= ROSTER_MAX:
            print(f"  roster full ({ROSTER_MAX}).")
            continue
        chosen.append(pick["player_name"])
        spent += int(pick["cost"])
        print(f"  drafted {pick['player_name']} for ${pick['cost']}.")

    if not chosen:
        print("No players drafted.")
        return
    picks = _match_players(pool, chosen)
    picks.attrs["year"] = year
    report_team(conn, pool, picks, budget, b0, b1, sim=True)


def main():
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Draft-a-team game.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("pool", "score", "play"):
        sp = sub.add_parser(name)
        sp.add_argument("--year", required=True, help="Season, e.g. 2023-24")
        sp.add_argument("--basis", default="regular", choices=["regular", "playoffs", "blended"])
        sp.add_argument("--budget", type=int, default=DEFAULT_CAP)
        if name == "pool":
            sp.add_argument("--top", type=int, default=40)
        if name == "score":
            sp.add_argument("--players", required=True, help="Comma-separated player names")
    args = parser.parse_args()

    conn = get_conn()
    try:
        if args.command == "pool":
            cmd_pool(conn, args.year, args.basis, args.top)
        elif args.command == "score":
            cmd_score(conn, args.year, args.basis, args.budget, args.players)
        elif args.command == "play":
            cmd_play(conn, args.year, args.basis, args.budget)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
