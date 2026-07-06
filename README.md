# NBA Playoff Predictor

A pipeline for predicting NBA playoff outcomes from regular-season and playoff
statistics. Five stages, all idempotent:

1. **`fetch_games`** — list of every game (regular season + playoffs) per season
2. **`fetch_boxscores`** — traditional + advanced + four-factors box scores
3. **`fetch_playoff_series`** — round / series ID / game-in-series for playoff games
4. **`build_features`** — analytics-ready `game_features` table with pre-game rolling stats
5. **`load_odds`** — historical betting lines (user-supplied CSV)

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate    # or: .venv\Scripts\activate on Windows
make install
make pipeline          # stages 1–4 (odds is opt-in; see below)
make status            # row counts + coverage
```

If you don't have `make`, every target is just `python -m src.<module>`:

```bash
python -m src.db                      # init schema
python -m src.fetch_games
python -m src.fetch_boxscores
python -m src.fetch_playoff_series
python -m src.build_features
python -m src.load_odds               # after you've placed data/odds.csv
```

Default config fetches 5 seasons (2020-21 → 2024-25). Edit `SEASONS` in
`src/config.py` to widen the window. The first run of `fetch_boxscores` is the
slow step (~3 hours); it's resumable.

## Architecture

```
nba-playoff-predictor/
├── Makefile
├── requirements.txt
├── data/
│   ├── nba.db                  # SQLite database (created on first run)
│   ├── sample_odds.csv         # example of the odds CSV format
│   └── raw/                    # cached raw JSON from the NBA API
└── src/
    ├── config.py               # SEASONS, paths, throttle
    ├── db.py                   # schema + status helpers
    ├── api_client.py           # throttled, retrying, disk-cached client
    ├── fetch_games.py          # stage 1
    ├── fetch_boxscores.py      # stage 2
    ├── fetch_playoff_series.py # stage 3
    ├── build_features.py       # stage 4
    └── load_odds.py            # stage 5
```

## Database schema

### `games`
One row per game with date, season, season type, home/away team IDs, scores,
and (after stage 3) playoff round / series ID / game-in-series.

### `team_game_stats`
One row per team per game. Traditional + advanced + four factors combined.
Primary key: `(game_id, team_id)`.

### `game_features` (rebuilt every `build_features` run)
One row per game, with both teams' pre-game features and prediction targets.
Most useful columns:

- **Targets**: `home_won` (1/0), `home_margin` (signed point differential)
- **Context**: `home_rest_days`, `away_rest_days`, `diff_rest_days`,
  `home_games_played`, `away_games_played`
- **For each metric and window** (`_rs`, `_l5`, `_l10`):
  `home_<metric>_<window>`, `away_<metric>_<window>`,
  `diff_<metric>_<window>` (= home − away)

Where `<metric>` is one of `net_rating`, `off_rating`, `def_rating`, `pace`,
`efg_pct`, `tov_pct`, `oreb_pct`, `ts_pct`, `fta_rate`, and four
opponent-side four factors.

**Windows:**

- `_rs` — regular-season-to-date expanding mean (a team's season aggregate
  through the day before this game). For playoff games, this is the team's
  full regular-season aggregate.
- `_l5`, `_l10` — rolling mean of the last 5 / 10 games (regular and playoff
  combined), reset per season.

**Leakage check.** Every rolling and expanding computation uses `.shift(1)`
so a game's feature value is computed strictly from games before it. Sanity
check this yourself before trusting any model — pick a known game, manually
compute one feature, and compare.

### `odds`
Historical betting lines joined to games. Primary key:
`(game_id, source, line_type)`, where `line_type` is `open` or `close`.

## Stage details

### Stage 3 — playoff series backfill

`CommonPlayoffSeries` gives `SERIES_ID` and `GAME_NUM` per game but not round
number. We derive round by sorting series by their earliest game date, then
bucketing chronologically: first 8 series → round 1, next 4 → round 2,
next 2 → round 3, final 1 → round 4. This works because round N series can't
start before round N-1 begins.

The play-in tournament (2020-21 onward) uses `season_type = "PlayIn"` and is
not pulled by default — only `Regular Season` and `Playoffs` are in `config.SEASON_TYPES`.

### Stage 4 — feature engineering

`build_features.py` does the long → wide pivot and produces ~100 numeric
feature columns per game. The differentials (`diff_*`) are usually the
strongest single predictors — start with `diff_net_rating_rs` as a baseline.

To rebuild after pulling new data:

```bash
make build-features         # rebuilds in seconds; safe to re-run anytime
```

### Stage 5 — odds

This stage is **opt-in** and **source-agnostic**. You supply a CSV at
`data/odds.csv` matching the schema in `data/sample_odds.csv`:

| column | required | notes |
|---|---|---|
| `game_date` | yes | `YYYY-MM-DD` |
| `home_team_abbr` | yes | 3-letter |
| `away_team_abbr` | yes | 3-letter |
| `source` | yes | free-text label |
| `line_type` | yes | `open` or `close` |
| `home_ml`, `away_ml` | no | American moneyline |
| `home_spread`, `home_spread_odds`, `away_spread_odds` | no | |
| `total`, `over_odds`, `under_odds` | no | |

Joining is by `(game_date, home_team_abbr, away_team_abbr)`. The loader
normalizes common abbreviation aliases (BRK→BKN, PHO→PHX, etc.); add new
ones to `ABBR_ALIASES` in `src/load_odds.py` if a source uses something
unfamiliar.

**Where to find historical odds** (open question for you to decide):

- `sportsbookreviewsonline.com` — free per-season Excel archives, going
  back to 2007-08. Most accessible free option. You'll need to reshape
  from their two-rows-per-game format into our schema (a small pandas
  exercise — happy to script it if you pick this source).
- Kaggle — several user-uploaded NBA odds datasets of varying completeness.
- The Odds API — paid for historical data.

## Modeling roadmap

Now that the data is in shape, the natural next steps are:

1. **Sanity-check the features.** Open a notebook, load `game_features`,
   verify a known game (e.g. 2024 Finals Game 1) against Basketball Reference.
2. **Baseline logistic regression** on `diff_net_rating_rs`,
   `diff_rest_days`, and a home-court intercept. Filter to
   `season_type = 'Playoffs'`. Walk-forward validation by season.
3. **Compare against odds.** If you have moneylines loaded, compute the
   market's implied probability and use it as either a feature or a
   baseline to beat.
4. **Gradient boosting** (XGBoost / LightGBM) using the full feature set.
5. **Series simulation** — if you want series-win probability, simulate the
   7-game series via Monte Carlo using your per-game model, accounting for
   the 2-2-1-1-1 home/away pattern.

## Gotchas (still relevant)

- **Rate limits**: NBA.com throttles aggressively. Default 0.6s delay; don't
  lower it.
- **Era effects**: pace and 3PA jumped sharply post-2014. Don't train
  cross-era without normalization.
- **2019-20 bubble**: neutral-site playoffs, shortened. Flag or exclude.
- **`game_id` as TEXT**: zero-padded (`"0042300401"`). Don't cast to int.
- **Abbreviation drift**: if odds don't match games, check that the
  abbreviations agree (see stage 5).
