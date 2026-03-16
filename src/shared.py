"""
Shared constants and helpers for F1 dashboards (terminal + web).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "cache" / "processed"

# ---------------------------------------------------------------------------
# Driver display names
# ---------------------------------------------------------------------------

DRIVER_NAMES = {
    "max_verstappen": "Verstappen", "hamilton": "Hamilton",
    "leclerc": "Leclerc", "norris": "Norris", "piastri": "Piastri",
    "russell": "Russell", "sainz": "Sainz", "alonso": "Alonso",
    "stroll": "Stroll", "gasly": "Gasly", "ocon": "Ocon",
    "albon": "Albon", "bottas": "Bottas", "hulkenberg": "Hülkenberg",
    "bearman": "Bearman", "antonelli": "Antonelli", "hadjar": "Hadjar",
    "lawson": "Lawson", "colapinto": "Colapinto", "perez": "Pérez",
    "bortoleto": "Bortoleto", "arvid_lindblad": "Lindblad",
}

# ---------------------------------------------------------------------------
# Team colors — hex for web, Rich names derived in dashboard.py
# ---------------------------------------------------------------------------

TEAM_COLORS_HEX = {
    "mercedes": "#00D2BE", "ferrari": "#DC0000", "red_bull": "#3671C6",
    "mclaren": "#FF8700", "alpine": "#0090FF", "aston_martin": "#006F62",
    "williams": "#005AFF", "haas": "#B6BABD", "rb": "#6692FF",
    "audi": "#00E701", "cadillac": "#C0C0C0",
}

TEAM_COLORS_RICH = {
    "mercedes": "cyan", "ferrari": "red", "red_bull": "blue",
    "mclaren": "bright_yellow", "alpine": "bright_cyan",
    "aston_martin": "green", "williams": "bright_blue",
    "haas": "white", "rb": "bright_blue",
    "audi": "bright_green", "cadillac": "bright_white",
}

TEAM_SHORT = {
    "mercedes": "Mercedes", "ferrari": "Ferrari", "red_bull": "Red Bull",
    "mclaren": "McLaren", "alpine": "Alpine", "aston_martin": "Aston Martin",
    "williams": "Williams", "haas": "Haas", "rb": "RB",
    "audi": "Audi", "cadillac": "Cadillac",
}

# Circuit type display names (distinct from data/features/elo.py CIRCUIT_TYPES which maps circuit_id → type)
CIRCUIT_TYPE_LABELS = {
    "street": "Street", "high_speed": "High-Speed",
    "technical": "Technical", "mixed": "Mixed",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def driver_name(did: str) -> str:
    return DRIVER_NAMES.get(did, did.replace("_", " ").title())


def team_color_hex(cid: str) -> str:
    return TEAM_COLORS_HEX.get(cid, "#888888")


def team_color_rich(cid: str) -> str:
    return TEAM_COLORS_RICH.get(cid, "white")


def team_name(cid: str) -> str:
    return TEAM_SHORT.get(cid, cid.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_prediction(season: int, race_round: int) -> pd.DataFrame | None:
    path = DATA_DIR / f"prediction_{season}_R{race_round:02d}.csv"
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None


def prediction_mtime(season: int, race_round: int) -> str | None:
    """Return human-readable mtime of the prediction CSV, or None."""
    from datetime import datetime
    path = DATA_DIR / f"prediction_{season}_R{race_round:02d}.csv"
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except FileNotFoundError:
        return None


def available_predictions() -> set[tuple[int, int]]:
    """Return set of (season, round) tuples that have prediction CSVs."""
    files = DATA_DIR.glob("prediction_*_R*.csv")
    results = set()
    for f in files:
        parts = f.stem.split("_")
        results.add((int(parts[1]), int(parts[2][1:])))
    return results


def available_rounds(
    race_results: pd.DataFrame | None = None,
    pred_set: set[tuple[int, int]] | None = None,
) -> list:
    """Build list of rounds for the current season from race_results + prediction CSVs.

    Returns ``[((season, round), event_name), ...]`` sorted newest-first.
    """
    rounds: dict[tuple[int, int], str] = {}

    if pred_set is None:
        pred_set = available_predictions()

    # From race_results — current season only
    if race_results is not None and not race_results.empty:
        max_season = int(race_results["season"].max())
        current = race_results[race_results["season"] == max_season]
        if "race_name" in current.columns:
            unique = current[["season", "round", "race_name"]].drop_duplicates(["season", "round"])
            for _, row in unique.iterrows():
                s, r = int(row["season"]), int(row["round"])
                name = row["race_name"]
                rounds[(s, r)] = name if pd.notna(name) else f"Round {r}"
        else:
            for _, row in current[["season", "round"]].drop_duplicates().iterrows():
                rounds[(int(row["season"]), int(row["round"]))] = f"Round {row['round']}"

    # From prediction CSVs (may include future races not yet in results)
    for s, r in pred_set:
        if (s, r) not in rounds:
            rounds[(s, r)] = f"Round {r}"

    return sorted(rounds.items(), key=lambda x: (x[0][0], x[0][1]), reverse=True)


def get_event_name(season: int, race_round: int) -> str:
    """Get event name — tries race_results parquet first, then FastF1."""
    rr_path = DATA_DIR / "race_results.parquet"
    try:
        rr = pd.read_parquet(rr_path)
        match = rr[(rr["season"] == season) & (rr["round"] == race_round)]
        if not match.empty and "race_name" in match.columns:
            name = match.iloc[0]["race_name"]
            if pd.notna(name):
                return name
    except (FileNotFoundError, ImportError, OSError):
        pass

    try:
        import fastf1
        fastf1.Cache.enable_cache(str(DATA_DIR.parent / "fastf1"))
        schedule = fastf1.get_event_schedule(season, include_testing=False)
        event = schedule[schedule["RoundNumber"] == race_round]
        if not event.empty:
            return event.iloc[0]["EventName"]
    except Exception:
        pass

    return f"Round {race_round}"
