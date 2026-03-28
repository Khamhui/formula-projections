"""Upload pipeline results to Supabase via REST API (works from GitHub Actions)."""

import os
import sys
import json
import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.shared import DRIVER_NAMES, TEAM_SHORT as TEAM_NAMES

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://krfhvkbavtfbhsadzhee.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", os.environ.get("SUPABASE_SERVICE_ROLE_KEY", ""))

DRIVER_TEAMS = {
    "max_verstappen": "red_bull", "hadjar": "red_bull",
    "norris": "mclaren", "piastri": "mclaren",
    "leclerc": "ferrari", "hamilton": "ferrari",
    "russell": "mercedes", "antonelli": "mercedes",
    "alonso": "aston_martin", "stroll": "aston_martin",
    "gasly": "alpine", "colapinto": "alpine",
    "albon": "williams", "sainz": "williams",
    "hulkenberg": "audi", "bortoleto": "audi",
    "ocon": "haas", "bearman": "haas",
    "perez": "cadillac", "bottas": "cadillac",
    "lawson": "rb", "arvid_lindblad": "rb",
}

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}

UPSERT_HEADERS = {**HEADERS, "Prefer": "resolution=merge-duplicates"}


def upsert(table: str, rows: list[dict], on_conflict: str = ""):
    """Upsert rows to a Supabase table via REST API."""
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    if on_conflict:
        url += f"?on_conflict={on_conflict}"
    resp = requests.post(url, headers=UPSERT_HEADERS, json=rows)
    if resp.status_code not in (200, 201):
        print(f"  ERROR uploading to {table}: {resp.status_code} {resp.text[:200]}")
    return resp.status_code in (200, 201)


def upload_race(season: int, round_num: int, name: str, circuit_id: str,
                circuit_name: str, circuit_type: str = "mixed", country: str = ""):
    race_id = f"{season}-r{round_num:02d}-{circuit_id}"
    upsert("races", [{
        "id": race_id, "season": season, "round": round_num, "name": name,
        "circuit_id": circuit_id, "circuit_name": circuit_name,
        "circuit_type": circuit_type, "country": country,
    }], on_conflict="id")
    print(f"Race uploaded: {race_id}")
    return race_id


def upload_predictions(race_id: str, csv_path: str):
    df = pd.read_csv(csv_path)
    rows = []
    for _, row in df.iterrows():
        did = row["driver_id"]
        confidence = "high" if row["prob_winner"] >= 0.10 else \
                     "moderate" if row["prob_winner"] >= 0.03 else \
                     "volatile" if row["prob_winner"] >= 0.005 else "coin_flip"
        rows.append({
            "race_id": race_id, "driver_id": did,
            "driver_name": DRIVER_NAMES.get(did, did),
            "team_id": DRIVER_TEAMS.get(did, "unknown"),
            "predicted_position": float(row["predicted_position"]),
            "prob_winner": float(row["prob_winner"]),
            "prob_podium": float(row["prob_podium"]),
            "prob_points": float(row["prob_points"]),
            "prob_dnf": float(row["prob_dnf"]),
            "expected_points": float(row["sim_expected_points"]),
            "sim_median_position": int(row["sim_median_position"]),
            "sim_position_25": int(row["sim_position_25"]),
            "sim_position_75": int(row["sim_position_75"]),
            "confidence": confidence,
        })
    upsert("predictions", rows, on_conflict="race_id,driver_id")
    print(f"Predictions uploaded: {len(rows)} drivers for {race_id}")


def upload_standings(season: int):
    ds = pd.read_parquet("data/cache/processed/driver_standings.parquet")
    latest = ds[ds["season"] == season]
    latest = latest[latest["round"] == latest["round"].max()]

    rows = []
    for _, row in latest.iterrows():
        did = row["driver_id"]
        rows.append({
            "season": season, "driver_id": did,
            "driver_name": DRIVER_NAMES.get(did, did),
            "team_id": DRIVER_TEAMS.get(did, row.get("constructor_id", "unknown")),
            "position": int(row["position"]),
            "points": float(row["points"]),
            "wins": int(row["wins"]),
        })
    upsert("standings", rows, on_conflict="season,driver_id")

    cs = pd.read_parquet("data/cache/processed/constructor_standings.parquet")
    latest_c = cs[cs["season"] == season]
    latest_c = latest_c[latest_c["round"] == latest_c["round"].max()]

    c_rows = []
    for _, row in latest_c.iterrows():
        cid = row["constructor_id"]
        c_rows.append({
            "season": season, "team_id": cid,
            "team_name": TEAM_NAMES.get(cid, cid),
            "position": int(row["position"]),
            "points": float(row["points"]),
        })
    upsert("constructor_standings", c_rows, on_conflict="season,team_id")

    print(f"Standings uploaded: {len(rows)} drivers, {len(c_rows)} constructors")


if __name__ == "__main__":
    if not SUPABASE_KEY:
        print("ERROR: SUPABASE_SERVICE_KEY not set")
        sys.exit(1)

    races = [
        (2026, 1, "Australian Grand Prix", "albert_park", "Albert Park", "mixed", "Australia"),
        (2026, 2, "Chinese Grand Prix", "shanghai", "Shanghai International Circuit", "mixed", "China"),
        (2026, 3, "Japanese Grand Prix", "suzuka", "Suzuka International Racing Course", "technical", "Japan"),
    ]
    for r in races:
        upload_race(*r)

    for round_num in [2, 3]:
        csv_path = f"data/cache/processed/prediction_2026_R{round_num:02d}.csv"
        race_id = f"2026-r{round_num:02d}-{'shanghai' if round_num == 2 else 'suzuka'}"
        if os.path.exists(csv_path):
            upload_predictions(race_id, csv_path)

    upload_standings(2026)

    print("\nDone! All data uploaded to Supabase.")
