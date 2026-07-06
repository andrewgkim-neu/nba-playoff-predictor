"""Browser GUI: an 8-team snake draft (3rd-round reversal) you play against AI.

Run it:
    streamlit run app.py

Pick your draft slot, then draft a team turn-by-turn against 7 AI teams that
take the best player available. When the board is full, your team runs through
a full regular-season + playoff simulation.

All the logic lives in src/draft.py; this file is the UI / draft-room state.
"""

import pandas as pd
import streamlit as st

from src import draft
from src.db import get_conn

st.set_page_config(page_title="NBA Snake Draft", page_icon="🏀", layout="wide")

N_TEAMS = 8
N_ROUNDS = 8
TOTAL_PICKS = N_TEAMS * N_ROUNDS


# --- cached data access ---

@st.cache_data
def get_seasons():
    conn = get_conn()
    try:
        return sorted(pd.read_sql("SELECT DISTINCT season FROM player_season_profiles", conn)["season"])
    finally:
        conn.close()


@st.cache_data
def load_pool(year, basis):
    conn = get_conn()
    try:
        return draft.build_pool(conn, year, basis).sort_values("value", ascending=False).reset_index(drop=True)
    finally:
        conn.close()


@st.cache_data
def get_field(year):
    conn = get_conn()
    try:
        return draft.field_net_ratings(conn, year)
    finally:
        conn.close()


@st.cache_resource
def winprob_params():
    conn = get_conn()
    try:
        return draft.fit_winprob(conn)
    finally:
        conn.close()


def _reroll_season():
    st.session_state["season_seed"] = st.session_state.get("season_seed", 7) + 1


PLAYER_COLS = {
    "player_name": "Player", "ovr": "OVR",
    "mpg": st.column_config.NumberColumn("MPG", format="%.1f"),
    "ppg": st.column_config.NumberColumn("PPG", format="%.1f"),
    "net_rating": st.column_config.NumberColumn("Net Rtg", format="%+.1f"),
}


# --- setup / sidebar ---

st.title("🏀 NBA Snake Draft")
st.caption("8 teams · 8 rounds · 3rd-round-reversal order. Draft against the AI, then simulate your season.")

seasons = get_seasons()
with st.sidebar:
    st.header("Setup")
    year = st.selectbox("Season", seasons, index=len(seasons) - 1)
    basis = st.radio("Rate players by", ["regular", "playoffs", "blended"],
                     help="Playoffs rewards postseason risers.")
    your_seat = st.selectbox("Your draft slot", list(range(1, N_TEAMS + 1)),
                             help="Seat 1 picks first; 3rd-round reversal softens that edge.")
    start = st.button("🏀 Start / reset draft", type="primary")
    st.markdown("---")
    st.caption("Snake order: R1 forward, R2 reverse, **R3 reverse again** (the reversal), "
               "then normal snake. AI teams take the best player available.")

if start:
    st.session_state.draft = {
        "year": year, "basis": basis, "your_seat": your_seat - 1,
        "order": draft.snake_order(N_TEAMS, N_ROUNDS),
        "pick": 0, "taken": set(), "rosters": {s: [] for s in range(N_TEAMS)}, "log": [],
    }
    st.session_state.pop("season_seed", None)

if "draft" not in st.session_state:
    st.info("Choose your season, player basis, and draft slot in the sidebar, then click **Start / reset draft**.")
    st.stop()

D = st.session_state.draft
pool = load_pool(D["year"], D["basis"])

# --- advance AI picks until it's your turn (or the draft is done) ---

while D["pick"] < TOTAL_PICKS and D["order"][D["pick"]] != D["your_seat"]:
    seat = D["order"][D["pick"]]
    pick = draft.best_available(pool, D["taken"])
    D["taken"].add(int(pick["player_id"]))
    D["rosters"][seat].append(int(pick["player_id"]))
    D["log"].append({"round": D["pick"] // N_TEAMS + 1, "seat": seat + 1,
                     "player": pick["player_name"], "you": False})
    D["pick"] += 1

done = D["pick"] >= TOTAL_PICKS
st.progress(
    (D["pick"]) / TOTAL_PICKS,
    text=(f"Pick {D['pick'] + 1}/{TOTAL_PICKS} · Round {D['pick'] // N_TEAMS + 1}" if not done
          else "Draft complete"),
)

main, side = st.columns([2, 1])

# --- side panel: your roster + recent picks ---

with side:
    st.markdown(f"**Your roster · seat {D['your_seat'] + 1}**")
    your_picks = pool[pool["player_id"].isin(D["rosters"][D["your_seat"]])]
    if len(your_picks):
        st.dataframe(your_picks[["player_name", "ovr", "mpg", "net_rating"]],
                     hide_index=True, width="stretch", column_config=PLAYER_COLS)
    else:
        st.caption("No picks yet — you're up soon.")
    st.markdown("**Recent picks**")
    for e in D["log"][-10:][::-1]:
        who = "🟢 **You**" if e["you"] else f"Team {e['seat']}"
        st.caption(f"R{e['round']} · {who}: {e['player']}")

# --- main panel: your pick, or the results ---

with main:
    if not done:
        avail = pool[~pool["player_id"].isin(D["taken"])].reset_index(drop=True)
        st.subheader(f"You're on the clock — Round {D['pick'] // N_TEAMS + 1}")
        labels = [f"{r.player_name} · OVR {r.ovr} · {r.mpg:.0f} mpg · NR {r.net_rating:+.1f}"
                  for r in avail.itertuples()]
        idx = st.selectbox("Pick a player (type to search)", range(len(labels)),
                           format_func=lambda i: labels[i])
        if st.button("Draft player", type="primary"):
            pick = avail.iloc[idx]
            D["taken"].add(int(pick["player_id"]))
            D["rosters"][D["your_seat"]].append(int(pick["player_id"]))
            D["log"].append({"round": D["pick"] // N_TEAMS + 1, "seat": D["your_seat"] + 1,
                             "player": pick["player_name"], "you": True})
            D["pick"] += 1
            st.rerun()
        with st.expander("Best available", expanded=True):
            st.dataframe(avail.head(15)[["player_name", "ovr", "mpg", "ppg", "net_rating"]],
                         hide_index=True, width="stretch", column_config=PLAYER_COLS)
    else:
        b0, b1 = winprob_params()
        standings = draft.score_rosters(pool, D["rosters"])
        your_rank = int(standings.loc[standings["seat"] == D["your_seat"], "rank"].iloc[0])
        st.subheader(f"Draft complete — your team ranks #{your_rank} of {N_TEAMS} by roster strength")

        disp = standings.copy()
        disp["team"] = disp["seat"].map(lambda s: "🟢 You" if s == D["your_seat"] else f"Team {s + 1}")
        st.dataframe(
            disp[["rank", "team", "net_rating"]], hide_index=True, width="stretch",
            column_config={"rank": "Rank", "team": "Team",
                           "net_rating": st.column_config.NumberColumn("Team Net Rtg", format="%+.1f")},
        )

        # --- season + playoff simulation for your team ---
        your_picks = pool[pool["player_id"].isin(D["rosters"][D["your_seat"]])]
        r = draft.team_rating(your_picks)
        field = draft.adjust_field(get_field(D["year"]), your_picks)  # opponents at league average
        seed = st.session_state.get("season_seed", 7)
        res = draft.run_season(r["net_rating"], field, b0, b1, seed=seed)
        odds = draft.championship_odds(r["net_rating"], field, b0, b1)
        w, l = res["record"]

        st.markdown("---")
        head, btn = st.columns([4, 1])
        head.subheader("Season simulation")
        btn.button("🎲 Re-sim", on_click=_reroll_season, help="Roll a new season with this roster")

        m1, m2, m3 = st.columns(3)
        m1.metric("Regular season", f"{w}-{l}", f"#{res['seed']} seed", delta_color="off")
        m2.metric("Result", "🏆 Champions!" if res["won_title"]
                  else ("Made playoffs" if res["made_playoffs"] else "Missed playoffs"))
        m3.metric("Championship odds", f"{odds:.1%}")

        if r["replacement_minutes"] > 0:
            st.caption(f"⚠ roster covers {240 - r['replacement_minutes']:.0f}/240 minutes; "
                       f"the rest is replacement level.")
        if not res["made_playoffs"]:
            st.info(f"Finished #{res['seed']} and missed the 16-team field this run — try Re-sim.")
        else:
            st.markdown("**Your playoff run**")
            for leg in draft.my_playoff_path(res):
                icon = "✅" if leg["won"] else "❌"
                st.write(f"{icon} **{leg['round']}** — {'beat' if leg['won'] else 'lost to'} {leg['opponent']}")
            if res["won_title"]:
                st.success(f"🏆 Your team wins the {D['year']} championship!")
            else:
                st.info(f"Champion this run: **{res['champion']}**")
