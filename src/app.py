"""
F1 Prediction Engine — Web Dashboard.

Local Flask app for browsing predictions, results, standings, and news.

Usage:
    python -m src.app              # Start on http://localhost:5050
    python -m src.app --port 8080  # Custom port
"""

from __future__ import annotations

import threading
import time
from datetime import datetime

import feedparser
import pandas as pd
from flask import Flask, render_template, request, jsonify

from src.shared import (
    DATA_DIR,
    available_predictions,
    available_rounds,
    driver_name,
    get_event_name,
    load_prediction,
    prediction_mtime,
    team_color_hex,
    team_name,
)

import math

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Data loaders (parquet files — cached by mtime)
# ---------------------------------------------------------------------------

_cache: dict = {}
_cache_ts: dict = {}


def _load(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.parquet"
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return pd.DataFrame()
    if name in _cache and _cache_ts.get(name) == mtime:
        return _cache[name]
    df = pd.read_parquet(path)
    _cache[name] = df
    _cache_ts[name] = mtime
    return df


# Make helpers available in templates
app.jinja_env.globals.update(
    driver_name=driver_name,
    team_color=team_color_hex,
    team_name=team_name,
)
app.jinja_env.filters["notnan"] = lambda v: v is not None and (not isinstance(v, float) or not math.isnan(v))


def _pos_class(pos) -> str:
    """Return CSS class for a finishing position."""
    try:
        p = int(pos)
    except (TypeError, ValueError):
        return "dim"
    return {1: "p1", 2: "p2", 3: "p3"}.get(p, "dim")


def _delta_class(val) -> str:
    """Return CSS class for a +/- delta value."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "dim"
    if v > 0:
        return "gain"
    if v < 0:
        return "drop"
    return "dim"


def _na(val, default=""):
    """Return default if val is NaN/None, else val."""
    try:
        if val != val:  # NaN check
            return default
    except (TypeError, ValueError):
        pass
    if val is None:
        return default
    return val


app.jinja_env.filters["pos_class"] = _pos_class
app.jinja_env.filters["delta_class"] = _delta_class
app.jinja_env.filters["na"] = _na


def _filter_round(df: pd.DataFrame, season: int, race_round: int) -> pd.DataFrame:
    """Filter a results DataFrame to a specific season/round."""
    if df.empty or "season" not in df.columns or "round" not in df.columns:
        return pd.DataFrame()
    return df[(df["season"] == season) & (df["round"] == race_round)].sort_values("position")


def _latest_standings(df: pd.DataFrame, season: int, max_round: int | None = None) -> pd.DataFrame:
    """Get standings for a season up to max_round, sorted by points."""
    if df.empty or "season" not in df.columns:
        return pd.DataFrame()
    s = df[df["season"] == season]
    if max_round is not None:
        s = s[s["round"] <= max_round]
    if s.empty:
        return pd.DataFrame()
    latest_round = s["round"].max()
    s = s[s["round"] == latest_round].sort_values("points", ascending=False)
    s["position"] = range(1, len(s) + 1)
    return s


def _event_name_from(rr: pd.DataFrame, season: int, race_round: int) -> str:
    """Extract event name from already-loaded race_results, falling back to shared helper."""
    if not rr.empty and "race_name" in rr.columns:
        match = rr[(rr["season"] == season) & (rr["round"] == race_round)]
        if not match.empty:
            name = match.iloc[0]["race_name"]
            if pd.notna(name):
                return name
    return get_event_name(season, race_round)


def _build_sparklines(
    df: pd.DataFrame, current: pd.DataFrame, season: int, race_round: int,
    col: str, n: int = 10, as_int: bool = False,
) -> dict:
    """Build per-driver sparkline data from a historical DataFrame.

    Filters to drivers in ``current``, excludes future rounds, and returns
    the last ``n`` values of ``col`` per driver.
    """
    if df.empty or current.empty or "driver_id" not in current.columns:
        return {}

    driver_ids = set(current["driver_id"].unique())
    recent = df[df["driver_id"].isin(driver_ids)]
    recent = recent[~((recent["season"] == season) & (recent["round"] > race_round))]
    recent = recent.sort_values(["season", "round"])

    history: dict = {}
    for did, group in recent.groupby("driver_id"):
        vals = group[col].dropna().tail(n).tolist()
        if len(vals) >= 2:
            history[did] = [int(v) for v in vals] if as_int else vals
    return history


def _build_elo_data(fm: pd.DataFrame, current: pd.DataFrame, season: int, race_round: int) -> dict:
    """Extract ELO rankings and sparkline history from feature matrix."""
    if fm.empty or "elo_overall" not in fm.columns:
        return {"driver_elo": pd.DataFrame(), "constructor_elo_map": {}, "elo_history": {}, "circuit_type": "mixed"}

    # Fall back to latest available race if current slice is empty
    if current.empty:
        current = fm[fm["season"] == season]
        if not current.empty:
            current = current[current["round"] == current["round"].max()]

    # Driver ELO table
    elo_cols = ["driver_id", "constructor_id", "elo_overall", "elo_qualifying", "elo_circuit_type", "elo_constructor"]
    available_cols = [c for c in elo_cols if c in current.columns]
    driver_elo = current[available_cols].copy() if not current.empty else pd.DataFrame()
    if not driver_elo.empty:
        driver_elo = driver_elo.sort_values("elo_overall", ascending=False).reset_index(drop=True)
        driver_elo["elo_rank"] = range(1, len(driver_elo) + 1)

    # Constructor ELO as dict for O(1) template lookups
    constructor_elo_map: dict[str, int] = {}
    if not current.empty and "elo_constructor" in current.columns:
        constructor_elo_map = (
            current.groupby("constructor_id")["elo_constructor"]
            .max()
            .round()
            .astype(int)
            .to_dict()
        )

    elo_history = _build_sparklines(fm, current, season, race_round, "elo_overall", n=10)

    # Circuit type for this race
    circuit_type = "mixed"
    if not current.empty and "circuit_type" in current.columns:
        ct = current.iloc[0].get("circuit_type", "mixed")
        circuit_type = ct if pd.notna(ct) else "mixed"

    return {
        "driver_elo": driver_elo,
        "constructor_elo_map": constructor_elo_map,
        "elo_history": elo_history,
        "circuit_type": circuit_type,
    }


def _build_race_context(current: pd.DataFrame) -> dict:
    """Extract circuit-specific context for the current race."""
    if current.empty:
        return {"circuit_stats": {}, "circuit_drivers": pd.DataFrame()}

    stats = {}
    stat_cols = {
        "circuit_grid_correlation": "grid_correlation",
        "circuit_overtaking_rate": "overtaking_rate",
        "circuit_attrition_rate": "attrition_rate",
        "grid_importance_score": "grid_importance",
        "circuit_front_row_win_rate": "front_row_win_rate",
    }
    for col, label in stat_cols.items():
        if col in current.columns:
            v = current[col].mean()
            if pd.notna(v):
                stats[label] = round(float(v), 2)

    driver_cols = [
        "driver_id", "constructor_id",
        "circuit_avg_pos", "circuit_best_pos", "circuit_races",
        "circuit_podium_rate", "circuit_quali_avg", "circuit_win_streak",
    ]
    avail = [c for c in driver_cols if c in current.columns]
    drivers = current[avail].copy() if len(avail) > 2 else pd.DataFrame()
    if not drivers.empty and "circuit_avg_pos" in drivers.columns:
        drivers = drivers.sort_values("circuit_avg_pos").reset_index(drop=True)

    return {"circuit_stats": stats, "circuit_drivers": drivers}


def _build_driver_form(current: pd.DataFrame) -> pd.DataFrame:
    """Extract driver form, momentum, teammate comparison, and pace data."""
    if current.empty:
        return pd.DataFrame()

    cols = [
        "driver_id", "constructor_id",
        "pos_last3_mean", "pos_last5_mean", "pos_last10_mean",
        "momentum_score", "form_vs_season_avg", "season_avg_pos",
        "h2h_quali_rate", "teammate_elo_diff",
        "dnf_rate_last5", "dnf_rate_last10", "dnf_streak",
        "quali_delta_vs_field", "quali_improvement_pct",
        "fp_delta_vs_field", "fp_total_laps",
    ]
    avail = [c for c in cols if c in current.columns]
    if len(avail) <= 2:
        return pd.DataFrame()
    df = current[avail].copy()
    if "momentum_score" in df.columns:
        df = df.sort_values("momentum_score", ascending=False)
    return df.reset_index(drop=True)


_FORM_SORT_SPECS = {
    "form_by_quali": ("quali_delta_vs_field", True),
    "form_by_pos": ("pos_last5_mean", True),
    "form_by_h2h": ("h2h_quali_rate", False),
    "form_by_dnf": ("dnf_rate_last5", False),
}


def _presort_driver_form(df: pd.DataFrame) -> dict:
    """Pre-sort driver form for template use (avoids sort_values in Jinja)."""
    if df.empty:
        return {}
    views = {}
    for key, (col, ascending) in _FORM_SORT_SPECS.items():
        if col in df.columns:
            views[key] = df.sort_values(col, ascending=ascending).reset_index(drop=True)
    return views


def _build_constructor_trends(current: pd.DataFrame) -> pd.DataFrame:
    """Extract constructor development trajectory data."""
    if current.empty:
        return pd.DataFrame()

    cols = [
        "constructor_id",
        "constructor_pace_jump", "constructor_pace_jump_magnitude",
        "constructor_season_trend", "constructor_season_avg",
        "elo_constructor",
    ]
    avail = [c for c in cols if c in current.columns]
    if "constructor_id" not in avail:
        return pd.DataFrame()
    df = current[avail].drop_duplicates("constructor_id").copy()
    if "elo_constructor" in df.columns:
        df = df.sort_values("elo_constructor", ascending=False)
    return df.reset_index(drop=True)


def _build_prediction_accuracy(pred, race: pd.DataFrame):
    """Compare prediction vs actual race result."""
    if pred is None or pred.empty or race.empty:
        return None
    if "driver_id" not in race.columns:
        return None

    pred_cols = ["driver_id", "predicted_position"]
    for c in ["sim_win_pct", "sim_podium_pct", "sim_points_pct"]:
        if c in pred.columns:
            pred_cols.append(c)

    merged = pred[pred_cols].merge(
        race[["driver_id", "position"]].rename(columns={"position": "actual"}),
        on="driver_id", how="inner",
    )
    if merged.empty:
        return None

    merged["delta"] = merged["actual"] - merged["predicted_position"]
    return merged.sort_values("actual").reset_index(drop=True)


def _build_position_history(
    rr: pd.DataFrame, current: pd.DataFrame, season: int, race_round: int,
) -> dict:
    """Build last-8 race positions per driver for sparklines."""
    if rr.empty:
        return {}
    # Filter to recent seasons to avoid scanning full history
    recent_rr = rr[rr["season"] >= season - 2]
    return _build_sparklines(recent_rr, current, season, race_round, "position", n=8, as_int=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Single-page terminal dashboard — all data in one view."""
    # Results (needed for round list + race display)
    rr = _load("race_results")

    # Build full round list from results + prediction CSVs
    pred_set = available_predictions()
    all_rounds = available_rounds(rr, pred_set=pred_set)

    season = request.args.get("season", type=int)
    race_round = request.args.get("round", type=int)

    if (not season or not race_round) and all_rounds:
        (season, race_round), _ = all_rounds[0]
    elif not season:
        season, race_round = 2026, 1

    # Feature matrix (used for predictions context + ELO)
    fm = _load("feature_matrix")

    # Pre-compute the current race slice (used by both prediction merge and ELO)
    if not fm.empty:
        current = fm[(fm["season"] == season) & (fm["round"] == race_round)]
    else:
        current = pd.DataFrame()

    # Prediction
    pred = load_prediction(season, race_round)

    if pred is not None and not current.empty:
        ctx = current[["driver_id", "constructor_id", "grid"]]
        pred = pred.merge(ctx, on="driver_id", how="left")

    # Pre-sort DNF for chart (avoid sort_values in Jinja template)
    dnf_sorted = None
    if pred is not None and "sim_dnf_pct" in pred.columns:
        dnf_sorted = pred.sort_values("sim_dnf_pct", ascending=False)

    # Results (qualifying, sprint, race) — rr already loaded above
    q = _load("qualifying")
    sp = _load("sprints")

    race = _filter_round(rr, season, race_round)
    quali = _filter_round(q, season, race_round)
    sprint = _filter_round(sp, season, race_round)

    # Event name from already-loaded race_results
    event = _event_name_from(rr, season, race_round)

    # Standings — scoped to selected round
    driver_s = _latest_standings(_load("driver_standings"), season, race_round)
    constructor_s = _latest_standings(_load("constructor_standings"), season, race_round)

    # Prediction freshness
    pred_generated = prediction_mtime(season, race_round)

    # ELO data
    elo_data = _build_elo_data(fm, current, season, race_round)

    # New data layers
    race_context = _build_race_context(current)
    driver_form = _build_driver_form(current)
    form_views = _presort_driver_form(driver_form)
    constructor_trends = _build_constructor_trends(current)
    pred_accuracy = _build_prediction_accuracy(pred, race)
    position_history = _build_position_history(rr, current, season, race_round)

    return render_template(
        "terminal.html",
        pred=pred,
        dnf_sorted=dnf_sorted,
        event_name=event,
        season=season,
        race_round=race_round,
        all_rounds=all_rounds,
        pred_set=pred_set,
        pred_generated=pred_generated,
        race=race,
        quali=quali,
        sprint=sprint,
        drivers=driver_s,
        constructors=constructor_s,
        now=datetime.now().strftime("%Y-%m-%d %H:%M"),
        race_context=race_context,
        driver_form=driver_form,
        constructor_trends=constructor_trends,
        pred_accuracy=pred_accuracy,
        position_history=position_history,
        **form_views,
        **elo_data,
    )


# RSS cache — avoid re-fetching on every page load
_news_cache: list = []
_news_cache_time: float = 0
NEWS_CACHE_TTL = 300  # 5 minutes


@app.route("/api/news")
def api_news():
    """Fetch RSS feeds (called via JS to avoid blocking page load)."""
    global _news_cache, _news_cache_time

    if _news_cache and (time.time() - _news_cache_time) < NEWS_CACHE_TTL:
        return jsonify(_news_cache)

    feeds = [
        ("Formula 1", "https://www.formula1.com/content/fom-website/en/latest/all.xml"),
        ("Autosport", "https://www.autosport.com/rss/f1/news/"),
        ("Motorsport.com", "https://www.motorsport.com/rss/f1/news/"),
    ]

    articles = []
    for source_name, url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                articles.append({
                    "source": source_name,
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "summary": entry.get("summary", "")[:200],
                })
        except Exception:
            continue

    _news_cache = articles
    _news_cache_time = time.time()
    return jsonify(articles)


# ---------------------------------------------------------------------------
# Prediction runner
# ---------------------------------------------------------------------------

_predict_lock = threading.Lock()
_predict_status: dict = {"running": False, "result": None, "error": None}


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Trigger prediction for a given season/round in a background thread."""
    data = request.json or {}
    season = data.get("season")
    race_round = data.get("round")
    mode = data.get("mode", "predict")

    if mode not in ("predict", "full"):
        return jsonify({"error": "mode must be 'predict' or 'full'"}), 400
    if not season or not race_round:
        return jsonify({"error": "season and round required"}), 400

    with _predict_lock:
        if _predict_status["running"]:
            return jsonify({"status": "already_running"}), 409
        _predict_status["running"] = True
        _predict_status["result"] = None
        _predict_status["error"] = None

    def run():
        try:
            if mode == "full":
                from data.auto_update import run_update
                run_update(force=True)
            else:
                from data.predict_weekend import run_weekend_prediction
                run_weekend_prediction(season, race_round)
            with _predict_lock:
                _predict_status["result"] = "done"
        except Exception as e:
            with _predict_lock:
                _predict_status["error"] = str(e)
        finally:
            with _predict_lock:
                _predict_status["running"] = False
            _cache_ts.clear()

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "started", "season": season, "round": race_round, "mode": mode})


@app.route("/api/predict/status")
def api_predict_status():
    """Poll prediction progress."""
    with _predict_lock:
        snapshot = _predict_status.copy()
    return jsonify(snapshot)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def start(port: int = 5050, open_browser: bool = True):
    """Start the dashboard server."""
    if open_browser:
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="F1 Dashboard")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    start(port=args.port, open_browser=not args.no_browser)
