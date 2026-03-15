"""
F1 Prediction Engine — Web Dashboard.

Local Flask app for browsing predictions, results, standings, and news.

Usage:
    python -m src.app              # Start on http://localhost:5050
    python -m src.app --port 8080  # Custom port
"""

from __future__ import annotations

import time
from datetime import datetime

import feedparser
import pandas as pd
from flask import Flask, render_template, request, jsonify

from src.shared import (
    DATA_DIR,
    available_predictions,
    driver_name,
    get_event_name,
    load_prediction,
    team_color_hex,
    team_name,
)

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


def _latest_standings(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Get latest standings for a season, sorted by points."""
    if df.empty or "season" not in df.columns:
        return pd.DataFrame()
    s = df[df["season"] == season]
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

    # Sparkline history: last 10 races per driver (single groupby, no N+1)
    elo_history = {}
    if not current.empty:
        driver_ids = set(current["driver_id"].unique())
        recent = fm[fm["season"] <= season]
        recent = recent[recent["driver_id"].isin(driver_ids)]
        if season == fm["season"].max():
            recent = recent[~((recent["season"] == season) & (recent["round"] > race_round))]
        recent = recent.sort_values(["season", "round"])
        for did, group in recent.groupby("driver_id"):
            vals = group["elo_overall"].dropna().tail(10).tolist()
            if len(vals) >= 2:
                elo_history[did] = vals

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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Single-page terminal dashboard — all data in one view."""
    preds = available_predictions()

    season = request.args.get("season", type=int)
    race_round = request.args.get("round", type=int)

    if (not season or not race_round) and preds:
        season, race_round, _ = preds[0]
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

    # Results (qualifying, sprint, race)
    rr = _load("race_results")
    q = _load("qualifying")
    sp = _load("sprints")

    race = _filter_round(rr, season, race_round)
    quali = _filter_round(q, season, race_round)
    sprint = _filter_round(sp, season, race_round)

    # Event name from already-loaded race_results
    event = _event_name_from(rr, season, race_round)

    # Standings
    driver_s = _latest_standings(_load("driver_standings"), season)
    constructor_s = _latest_standings(_load("constructor_standings"), season)

    # ELO data
    elo_data = _build_elo_data(fm, current, season, race_round)

    return render_template(
        "terminal.html",
        pred=pred,
        dnf_sorted=dnf_sorted,
        event_name=event,
        season=season,
        race_round=race_round,
        available=preds,
        race=race,
        quali=quali,
        sprint=sprint,
        drivers=driver_s,
        constructors=constructor_s,
        now=datetime.now().strftime("%Y-%m-%d %H:%M"),
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
# Entry point
# ---------------------------------------------------------------------------

def start(port: int = 5050, open_browser: bool = True):
    """Start the dashboard server."""
    if open_browser:
        import webbrowser
        import threading
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="F1 Dashboard")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    start(port=args.port, open_browser=not args.no_browser)
