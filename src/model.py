"""Step 6 (modeling): train and evaluate playoff prediction models.

This module sits on top of the ``game_features`` table produced by
``build_features``. It provides three things:

  1. A **baseline** logistic regression on a handful of interpretable
     features (``diff_net_rating_rs``, ``diff_rest_days``, plus a home-court
     intercept).
  2. A **full** gradient-boosting model (sklearn ``HistGradientBoostingClassifier``)
     over every numeric feature column. It handles NaNs natively, so early-season
     games with incomplete rolling windows are fine. (Swap in XGBoost/LightGBM
     here if you prefer — the interface is the same ``.fit``/``.predict_proba``.)
  3. **Walk-forward validation by season**: to score season S's playoffs we train
     only on games that happened before them — every prior season, plus season S's
     regular season. No future information leaks into a prediction.

It also includes a **Monte Carlo series simulator**: given two teams it builds
each team's end-of-season feature profile, predicts per-game home-win
probability with the trained model, and simulates best-of-7 series under the
standard 2-2-1-1-1 home/away pattern.

Prerequisite: run the data pipeline first (``make pipeline`` then
``make build-features``) so ``game_features`` exists. Until then every command
here will tell you the table is missing.

Usage:
    python -m src.model evaluate                 # walk-forward validation report
    python -m src.model train                    # fit final model on all data -> data/model.pkl
    python -m src.model simulate --home BOS --away DAL [--season 2023-24] [--sims 20000]
"""

import argparse
import logging
import pickle
import sys

import numpy as np
import pandas as pd

from src import config
from src.db import get_conn

logger = logging.getLogger(__name__)

# Where the trained final model is saved by `train`.
MODEL_PATH = config.DATA_DIR / "model.pkl"

# Reproducibility. Series simulation and model training both use this.
RANDOM_SEED = 17

# Std of (actual home margin − spread-implied margin), used to convert a point
# spread into a win probability via a normal model. Calibrated empirically from
# ~6k NBA regular-season games in the odds dataset (residual std ≈ 13.4, mean ≈ 0).
SPREAD_SIGMA = 13.4

# Interpretable baseline features. Differentials + a home-court intercept
# (the intercept is captured automatically because the target is `home_won`
# and all features are home-minus-away differentials).
BASELINE_FEATURES = ["diff_net_rating_rs", "diff_rest_days"]

# Columns that look like features (home_/away_/diff_ prefix) but are NOT.
NON_FEATURE_COLUMNS = {
    "home_team_id", "away_team_id",
    "home_score", "away_score",
    "home_won", "home_margin",
}

TARGET = "home_won"

# Default rest days assumed for both teams in a hypothetical playoff matchup.
# (Real playoff games typically have 1-3 days between them.)
DEFAULT_SERIES_REST_DAYS = 2

# Best-of-7 home/away pattern for the higher seed: home games are 1, 2, 5, 7.
# Index = game number (1-based); value = True if the higher seed is at home.
HIGHER_SEED_HOME = {1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: True}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _table_exists(conn, name: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    )
    return cur.fetchone() is not None


def load_features(conn) -> pd.DataFrame:
    """Load the whole ``game_features`` table, sorted by date."""
    if not _table_exists(conn, "game_features"):
        raise SystemExit(
            "game_features table not found. Run the pipeline first:\n"
            "    make pipeline && make build-features"
        )
    df = pd.read_sql("SELECT * FROM game_features", conn)
    if df.empty:
        raise SystemExit("game_features is empty. Did the box-score fetch finish?")
    df["game_date_dt"] = pd.to_datetime(df["game_date"])
    return df.sort_values("game_date_dt").reset_index(drop=True)


def team_abbr_maps(conn) -> tuple[dict, dict]:
    """Return (abbr -> team_id, team_id -> abbr) from the games table."""
    rows = conn.execute(
        """
        SELECT home_team_id, home_team_abbr FROM games
        WHERE home_team_abbr IS NOT NULL
        UNION
        SELECT away_team_id, away_team_abbr FROM games
        WHERE away_team_abbr IS NOT NULL
        """
    ).fetchall()
    abbr_to_id = {abbr: tid for tid, abbr in rows}
    id_to_abbr = {tid: abbr for tid, abbr in rows}
    return abbr_to_id, id_to_abbr


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Every numeric predictor column (home_/away_/diff_) minus IDs and targets."""
    return [
        c for c in df.columns
        if c.startswith(("home_", "away_", "diff_")) and c not in NON_FEATURE_COLUMNS
    ]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def make_baseline_model():
    """Median-impute -> standardize -> logistic regression."""
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def make_full_model():
    """Gradient-boosted trees over the full feature set. Handles NaNs natively."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=RANDOM_SEED,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def score(y_true: np.ndarray, p_pred: np.ndarray) -> dict:
    """Accuracy / log-loss / Brier / AUC for predicted home-win probabilities."""
    from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score

    y_true = np.asarray(y_true, dtype=int)
    p_pred = np.clip(np.asarray(p_pred, dtype=float), 1e-6, 1 - 1e-6)
    preds = (p_pred >= 0.5).astype(int)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "log_loss": float(log_loss(y_true, p_pred, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, p_pred)),
    }
    # AUC is undefined if the test fold is all one class.
    out["auc"] = float(roc_auc_score(y_true, p_pred)) if len(np.unique(y_true)) > 1 else float("nan")
    return out


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def walk_forward_predictions(df: pd.DataFrame, make_model, features: list[str]) -> pd.DataFrame:
    """Per-game walk-forward predictions for playoff games.

    For each season S that has playoff games, train on every game before S's
    playoffs (all earlier seasons + S's regular season) and predict S's
    playoffs. Returns one row per playoff game: game_id, season, y_true, p_pred.
    """
    out = []
    for s in sorted(df["season"].unique()):
        test = df[(df["season"] == s) & (df["season_type"] == "Playoffs")].dropna(subset=[TARGET])
        if test.empty:
            continue

        train = df[
            (df["season"] < s)
            | ((df["season"] == s) & (df["season_type"] == "Regular Season"))
        ].dropna(subset=[TARGET])
        if len(train) < 200:
            logger.info("  %s: only %d training games, skipping", s, len(train))
            continue

        model = make_model()
        model.fit(train[features], train[TARGET].astype(int))
        p = model.predict_proba(test[features])[:, 1]
        out.append(pd.DataFrame({
            "game_id": test["game_id"].to_numpy(),
            "season": s,
            "y_true": test[TARGET].astype(int).to_numpy(),
            "p_pred": p,
        }))

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["game_id", "season", "y_true", "p_pred"]
    )


def _score_by_season(preds: pd.DataFrame) -> pd.DataFrame:
    """One metrics row per season plus a pooled 'ALL' row."""
    rows = [{"season": s, **score(g["y_true"], g["p_pred"])} for s, g in preds.groupby("season")]
    rows.append({"season": "ALL", **score(preds["y_true"], preds["p_pred"])})
    return pd.DataFrame(rows)


def walk_forward(df: pd.DataFrame, make_model, features: list[str]) -> pd.DataFrame:
    """Walk-forward validation, summarized as a per-season metrics table."""
    preds = walk_forward_predictions(df, make_model, features)
    return _score_by_season(preds) if not preds.empty else pd.DataFrame()


def naive_baselines(df: pd.DataFrame) -> dict:
    """Reference points to beat, scored on all playoff games with a target."""
    po = df[(df["season_type"] == "Playoffs")].dropna(subset=[TARGET])
    if po.empty:
        return {}
    y = po[TARGET].astype(int).to_numpy()
    out = {"always_home": score(y, np.full(len(y), 0.5 + 1e-6))}
    out["always_home"]["accuracy"] = float((y == 1).mean())  # pick home every time
    if "diff_net_rating_rs" in po.columns:
        higher_net = (po["diff_net_rating_rs"].fillna(0) > 0).astype(int).to_numpy()
        out["higher_net_rating"] = {
            "n": int(len(y)),
            "accuracy": float((higher_net == y).mean()),
        }
    return out


def evaluate(conn) -> None:
    df = load_features(conn)
    feats = feature_columns(df)
    logger.info("Loaded %d games, %d feature columns", len(df), len(feats))
    # (kept below; see compare() for the head-to-head against sportsbook odds)

    print("\n=== Naive baselines (all playoff games) ===")
    for name, m in naive_baselines(df).items():
        print(f"  {name:<20} acc={m['accuracy']:.3f}  (n={m['n']})")

    print("\n=== Baseline logistic regression (walk-forward by season) ===")
    print(walk_forward(df, make_baseline_model, BASELINE_FEATURES).to_string(index=False))

    print("\n=== Full gradient-boosting model (walk-forward by season) ===")
    print(walk_forward(df, make_full_model, feats).to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# Market (sportsbook) comparison
# ---------------------------------------------------------------------------

def american_to_prob(ml) -> float:
    """American moneyline -> implied win probability (includes the vig)."""
    if ml is None or pd.isna(ml):
        return np.nan
    ml = float(ml)
    return (-ml) / (-ml + 100.0) if ml < 0 else 100.0 / (ml + 100.0)


def spread_to_prob(home_spread, sigma: float = SPREAD_SIGMA):
    """Signed home point spread (negative = home favored) -> P(home win).

    Models the actual home margin as Normal(-home_spread, sigma) and returns
    P(margin > 0). Vectorized; NaN spreads pass through as NaN.
    """
    from scipy.stats import norm
    hs = np.asarray(home_spread, dtype=float)
    return norm.cdf(-hs / sigma)


def load_market_probs(conn, line_type: str = "close", basis: str = "auto") -> pd.DataFrame:
    """Market home-win probability per game.

    - Moneyline -> de-vigged probability (home_ml, away_ml implied probs are
      normalized to sum to 1, removing the bookmaker margin).
    - Spread    -> normal-model probability via spread_to_prob.

    `basis`: 'moneyline' (ML only), 'spread' (spread only), or 'auto' (default:
    use the moneyline when present, else fall back to the spread). Returns
    columns [game_id, p_market, basis].
    """
    if not _table_exists(conn, "odds"):
        return pd.DataFrame(columns=["game_id", "p_market", "basis"])
    o = pd.read_sql(
        "SELECT game_id, home_ml, away_ml, home_spread FROM odds WHERE line_type = ?",
        conn, params=(line_type,),
    )
    if o.empty:
        return pd.DataFrame(columns=["game_id", "p_market", "basis"])

    ph, pa = o["home_ml"].map(american_to_prob), o["away_ml"].map(american_to_prob)
    p_ml = ph / (ph + pa)                       # de-vig
    p_spread = pd.Series(spread_to_prob(o["home_spread"]), index=o.index)

    if basis == "moneyline":
        o["p_market"], o["basis"] = p_ml, "moneyline"
    elif basis == "spread":
        o["p_market"], o["basis"] = p_spread, "spread"
    else:  # auto: prefer moneyline, fall back to spread
        o["p_market"] = p_ml.where(p_ml.notna(), p_spread)
        o["basis"] = np.where(p_ml.notna(), "moneyline", "spread")

    o = o.dropna(subset=["p_market"])
    # If multiple sources priced a game, average their probabilities.
    agg = o.groupby("game_id", as_index=False)["p_market"].mean()
    label = o.groupby("game_id", as_index=False)["basis"].first()
    return agg.merge(label, on="game_id")


def compare(conn, line_type: str = "close", basis: str = "auto") -> None:
    """Score the sportsbook vs both models on the playoff games that have lines."""
    df = load_features(conn)
    feats = feature_columns(df)

    market = load_market_probs(conn, line_type, basis)
    if market.empty:
        raise SystemExit(
            "No usable betting lines in the odds table. Load odds first:\n"
            "    python -m src.reshape_kaggle && python -m src.load_odds"
        )

    full = walk_forward_predictions(df, make_full_model, feats).rename(columns={"p_pred": "p_full"})
    base = walk_forward_predictions(df, make_baseline_model, BASELINE_FEATURES).rename(columns={"p_pred": "p_base"})

    merged = (
        full.merge(base[["game_id", "p_base"]], on="game_id")
            .merge(market, on="game_id", how="inner")
    )
    if merged.empty:
        raise SystemExit(
            "No overlap between playoff test games and games with betting lines. "
            "Check that the odds CSV covers these seasons' playoffs and that "
            "team abbreviations match (see src/load_odds.py)."
        )

    y = merged["y_true"].to_numpy()
    n_ml = int((merged["basis"] == "moneyline").sum())
    n_sp = int((merged["basis"] == "spread").sum())
    print(f"\n=== Model vs. market: {len(merged)} playoff games "
          f"({n_ml} via moneyline, {n_sp} via spread @ sigma={SPREAD_SIGMA}) ===")
    print(f"{'predictor':<22}{'acc':>8}{'log_loss':>10}{'brier':>9}{'auc':>8}")
    for name, col in [
        ("Market (sportsbook)", "p_market"),
        ("Baseline logreg", "p_base"),
        ("Full GBM model", "p_full"),
    ]:
        m = score(y, merged[col].to_numpy())
        print(f"{name:<22}{m['accuracy']:>8.3f}{m['log_loss']:>10.3f}{m['brier']:>9.3f}{m['auc']:>8.3f}")

    # How often each model agrees with the market's pick, and how the model's
    # pick fares when it disagrees with the book (the only place it can add value).
    for name, col in [("Baseline", "p_base"), ("Full GBM", "p_full")]:
        disagree = merged[(merged[col] >= 0.5) != (merged["p_market"] >= 0.5)]
        agree_rate = 1 - len(disagree) / len(merged)
        edge = (
            ((disagree[col] >= 0.5).astype(int) == disagree["y_true"]).mean()
            if len(disagree) else float("nan")
        )
        print(
            f"  {name}: agrees with book on {agree_rate:.0%} of games; "
            f"when it disagrees ({len(disagree)} games) it is right {edge:.0%} of the time"
        )

    # Accuracy by season — shows how the model fares against the book on its
    # stronger recent folds (which only have spreads, not moneylines).
    print("\n  accuracy by season (market vs full GBM):")
    for s, g in merged.groupby("season"):
        mkt_acc = (g["p_market"] >= 0.5).eq(g["y_true"] == 1).mean()
        mdl_acc = (g["p_full"] >= 0.5).eq(g["y_true"] == 1).mean()
        print(f"    {s}: market {mkt_acc:.3f}  model {mdl_acc:.3f}  (n={len(g)})")

    # Validate the spread->prob conversion against the true moneyline-implied
    # probability on games that have both.
    ov = pd.read_sql(
        "SELECT home_ml, away_ml, home_spread FROM odds WHERE line_type = ? "
        "AND home_ml IS NOT NULL AND away_ml IS NOT NULL AND home_spread IS NOT NULL",
        conn, params=(line_type,),
    )
    if len(ov):
        ph, pa = ov["home_ml"].map(american_to_prob), ov["away_ml"].map(american_to_prob)
        p_ml = (ph / (ph + pa)).to_numpy()
        p_sp = spread_to_prob(ov["home_spread"])
        mad = float(np.mean(np.abs(p_ml - p_sp)))
        corr = float(np.corrcoef(p_ml, p_sp)[0, 1])
        print(f"\n  spread->prob check on {len(ov)} games with both lines: "
              f"mean|diff|={mad:.3f}, corr={corr:.3f}")
    print()


# ---------------------------------------------------------------------------
# Market line as a model feature
# ---------------------------------------------------------------------------

def add_market_feature(conn, df: pd.DataFrame, line_type: str = "close",
                       basis: str = "auto") -> pd.DataFrame:
    """Join the market-implied probability onto the feature frame.

    Adds two columns: `p_market` (the probability) and `market_logit`
    (= ln(p/(1-p)), its log-odds — the scale the linear model works on, where a
    weight of 1 exactly reproduces the market). Games without a line get NaN,
    which the GBM handles natively and the logistic pipeline median-imputes.
    """
    market = load_market_probs(conn, line_type, basis)
    if market.empty:
        raise SystemExit(
            "No betting lines in the odds table. Load them first:\n"
            "    python -m src.reshape_kaggle && python -m src.load_odds"
        )
    df = df.merge(market[["game_id", "p_market"]], on="game_id", how="left")
    p = df["p_market"].clip(1e-4, 1 - 1e-4)
    df["market_logit"] = np.log(p / (1 - p))
    return df


def edge(conn, line_type: str = "close", basis: str = "auto") -> None:
    """Does adding the market line as a feature beat the line alone?

    Runs walk-forward for each model family twice — features only, and
    features + market_logit — and scores all of them, plus the market by
    itself, on the same playoff games. Then prints the logistic weights so you
    can see how much influence the line gets relative to the box-score features.
    """
    df = load_features(conn)
    df = add_market_feature(conn, df, line_type, basis)
    feats = feature_columns(df)  # excludes market_logit (not a home_/away_/diff_ column)

    runs = {
        "Baseline (features only)": (make_baseline_model, BASELINE_FEATURES),
        "Baseline + market":        (make_baseline_model, ["market_logit"] + BASELINE_FEATURES),
        "Full GBM (features only)": (make_full_model, feats),
        "Full GBM + market":        (make_full_model, feats + ["market_logit"]),
    }
    preds = {
        name: walk_forward_predictions(df, mk, fl).rename(columns={"p_pred": name})
        for name, (mk, fl) in runs.items()
    }

    # Common subset: playoff games that have a market line.
    merged = preds["Full GBM (features only)"][["game_id", "y_true"]].merge(
        df[["game_id", "p_market"]], on="game_id"
    ).dropna(subset=["p_market"])
    for name in runs:
        merged = merged.merge(preds[name][["game_id", name]], on="game_id")

    y = merged["y_true"].to_numpy()
    print(f"\n=== Does the market line as a feature help? {len(merged)} playoff games ===")
    print(f"{'predictor':<28}{'acc':>8}{'log_loss':>10}{'brier':>9}{'auc':>8}")

    def row(label, col):
        m = score(y, merged[col].to_numpy())
        print(f"{label:<28}{m['accuracy']:>8.3f}{m['log_loss']:>10.3f}{m['brier']:>9.3f}{m['auc']:>8.3f}")

    row("Market (sportsbook)", "p_market")
    for name in runs:
        row(name, name)

    # Illustrative: a logistic model on [market + a couple of diffs], fit on all
    # games, to show how much weight the line gets vs. the fundamentals.
    feat_list = ["market_logit"] + BASELINE_FEATURES
    train = df.dropna(subset=[TARGET, "market_logit"])
    pipe = make_baseline_model()
    pipe.fit(train[feat_list], train[TARGET].astype(int))
    coefs = pipe.named_steps["clf"].coef_[0]
    print("\n  logistic weights (standardized — larger |w| = more influence):")
    for f, c in sorted(zip(feat_list, coefs), key=lambda kv: -abs(kv[1])):
        print(f"    {f:<22} {c:+.3f}")
    print()


# ---------------------------------------------------------------------------
# Final model training + persistence
# ---------------------------------------------------------------------------

def train(conn) -> None:
    """Fit the full model on ALL games and save it for the simulator."""
    df = load_features(conn)
    feats = feature_columns(df)
    train_df = df.dropna(subset=[TARGET])

    model = make_full_model()
    model.fit(train_df[feats], train_df[TARGET].astype(int))

    profiles = build_team_profiles(df)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(
            {"model": model, "features": feats, "profiles": profiles},
            fh,
        )
    logger.info("Trained on %d games, saved model -> %s", len(train_df), MODEL_PATH)


def load_trained():
    if not MODEL_PATH.exists():
        raise SystemExit(f"No trained model at {MODEL_PATH}. Run: python -m src.model train")
    with open(MODEL_PATH, "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# Team feature profiles (for hypothetical matchups)
# ---------------------------------------------------------------------------

def build_team_profiles(df: pd.DataFrame) -> dict:
    """Each team's most recent feature profile, per season.

    In game_features a team appears as 'home_*' or 'away_*' depending on the
    game. We reconstruct a per-team view by taking, for the team's latest game
    in a season, whichever side they were on. Returns
    {season: {team_id: {base_feature: value}}} where base_feature has the
    home_/away_ prefix stripped (e.g. 'net_rating_rs', 'rest_days').
    """
    home_cols = [c for c in df.columns if c.startswith("home_") and c not in NON_FEATURE_COLUMNS]
    base = [c[len("home_"):] for c in home_cols]

    def side(prefix, team_col):
        cols = [f"{prefix}{b}" for b in base]
        part = df[["season", "game_date_dt", team_col] + cols].copy()
        part = part.rename(columns={team_col: "team_id"})
        part = part.rename(columns={f"{prefix}{b}": b for b in base})
        return part

    long = pd.concat(
        [side("home_", "home_team_id"), side("away_", "away_team_id")],
        ignore_index=True,
    )
    # Latest game per (season, team).
    long = long.sort_values("game_date_dt")
    latest = long.groupby(["season", "team_id"], as_index=False).tail(1)

    profiles: dict = {}
    for _, r in latest.iterrows():
        profiles.setdefault(r["season"], {})[int(r["team_id"])] = {
            b: r[b] for b in base
        }
    return profiles


def matchup_row(home_profile: dict, away_profile: dict, features: list[str],
                rest_home: float, rest_away: float) -> pd.DataFrame:
    """Build a single-game feature row for home vs away from team profiles."""
    home_profile = dict(home_profile)
    away_profile = dict(away_profile)
    if "rest_days" in home_profile:
        home_profile["rest_days"] = rest_home
        away_profile["rest_days"] = rest_away

    row = {}
    for b in home_profile:
        hv, av = home_profile[b], away_profile.get(b)
        row[f"home_{b}"] = hv
        row[f"away_{b}"] = av
        if hv is not None and av is not None:
            row[f"diff_{b}"] = hv - av
    # Align to exactly the columns the model was trained on.
    return pd.DataFrame([row]).reindex(columns=features)


def home_win_prob(model, features, profiles, season, home_id, away_id,
                  rest_home=DEFAULT_SERIES_REST_DAYS, rest_away=DEFAULT_SERIES_REST_DAYS) -> float:
    season_profiles = profiles.get(season)
    if season_profiles is None:
        raise SystemExit(f"No team profiles for season {season}. Available: {sorted(profiles)}")
    for tid in (home_id, away_id):
        if tid not in season_profiles:
            raise SystemExit(f"No profile for team_id {tid} in {season}.")
    x = matchup_row(season_profiles[home_id], season_profiles[away_id], features, rest_home, rest_away)
    return float(model.predict_proba(x)[:, 1][0])


# ---------------------------------------------------------------------------
# Monte Carlo series simulation
# ---------------------------------------------------------------------------

def simulate_series(model, features, profiles, season, higher_seed_id, lower_seed_id,
                    n_sims=20000, seed=RANDOM_SEED) -> dict:
    """Simulate a best-of-7 series under the 2-2-1-1-1 pattern.

    The higher seed hosts games 1, 2, 5, 7. We precompute the home-win
    probability for each hosting arrangement once, then vectorize the sims.
    """
    rng = np.random.default_rng(seed)

    # P(higher seed wins) when the higher seed is home vs. away.
    p_high_at_home = home_win_prob(model, features, profiles, season, higher_seed_id, lower_seed_id)
    p_high_on_road = 1.0 - home_win_prob(model, features, profiles, season, lower_seed_id, higher_seed_id)

    # Per game-number probability that the HIGHER seed wins that game.
    p_high_by_game = np.array([
        p_high_at_home if HIGHER_SEED_HOME[g] else p_high_on_road
        for g in range(1, 8)
    ])

    high_series_wins = 0
    length_counts = {4: 0, 5: 0, 6: 0, 7: 0}
    for _ in range(n_sims):
        hw = lw = 0
        for g in range(7):
            if rng.random() < p_high_by_game[g]:
                hw += 1
            else:
                lw += 1
            if hw == 4 or lw == 4:
                length_counts[g + 1] += 1
                break
        if hw == 4:
            high_series_wins += 1

    return {
        "p_high_seed_series": high_series_wins / n_sims,
        "p_high_game_home": p_high_at_home,
        "p_high_game_road": p_high_on_road,
        "length_distribution": {k: v / n_sims for k, v in length_counts.items()},
        "n_sims": n_sims,
    }


def simulate(conn, home_abbr, away_abbr, season, n_sims) -> None:
    bundle = load_trained()
    model, features, profiles = bundle["model"], bundle["features"], bundle["profiles"]

    abbr_to_id, id_to_abbr = team_abbr_maps(conn)
    if season is None:
        season = max(profiles)  # most recent season we have profiles for
    for ab in (home_abbr, away_abbr):
        if ab not in abbr_to_id:
            raise SystemExit(f"Unknown team abbreviation '{ab}'. Known: {sorted(abbr_to_id)}")

    home_id, away_id = abbr_to_id[home_abbr], abbr_to_id[away_abbr]

    p_home = home_win_prob(model, features, profiles, season, home_id, away_id)
    print(f"\nSeason {season}")
    print(f"Single game, {home_abbr} hosting {away_abbr}:")
    print(f"  P({home_abbr} win) = {p_home:.3f}   P({away_abbr} win) = {1 - p_home:.3f}")

    # Treat the named home team as the higher seed (home-court advantage in the series).
    res = simulate_series(model, features, profiles, season, home_id, away_id, n_sims=n_sims)
    print(f"\nBest-of-7 series, {home_abbr} as higher seed (2-2-1-1-1):")
    print(f"  P({home_abbr} win series) = {res['p_high_seed_series']:.3f}")
    print(f"  P({away_abbr} win series) = {1 - res['p_high_seed_series']:.3f}")
    print("  Series length distribution:")
    for length, prob in res["length_distribution"].items():
        print(f"    {length} games: {prob:.3f}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="NBA playoff prediction models.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("evaluate", help="Walk-forward validation report")
    sub.add_parser("train", help="Fit final model on all data and save it")

    cmp = sub.add_parser("compare", help="Score the model against sportsbook odds")
    cmp.add_argument("--line-type", default="close", choices=["open", "close"],
                     help="Which line to use (default: close)")
    cmp.add_argument("--basis", default="auto", choices=["auto", "moneyline", "spread"],
                     help="Market probability source (default: auto = moneyline, else spread)")

    edg = sub.add_parser("edge", help="Test whether the market line as a feature beats the line alone")
    edg.add_argument("--line-type", default="close", choices=["open", "close"])
    edg.add_argument("--basis", default="auto", choices=["auto", "moneyline", "spread"])

    sim = sub.add_parser("simulate", help="Predict a matchup / simulate a series")
    sim.add_argument("--home", required=True, help="Home / higher-seed team abbr (e.g. BOS)")
    sim.add_argument("--away", required=True, help="Away / lower-seed team abbr (e.g. DAL)")
    sim.add_argument("--season", default=None, help="Season string, e.g. 2024-25 (default: latest)")
    sim.add_argument("--sims", type=int, default=20000, help="Monte Carlo iterations")

    args = parser.parse_args()
    conn = get_conn()
    try:
        if args.command == "evaluate":
            evaluate(conn)
        elif args.command == "train":
            train(conn)
        elif args.command == "compare":
            compare(conn, args.line_type, args.basis)
        elif args.command == "edge":
            edge(conn, args.line_type, args.basis)
        elif args.command == "simulate":
            simulate(conn, args.home, args.away, args.season, args.sims)
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
