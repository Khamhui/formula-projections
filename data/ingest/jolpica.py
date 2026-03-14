"""
Jolpica-F1 API client — ingests historical F1 data from 1950 to present.
Successor to the Ergast API with full feature parity.
Base URL: https://api.jolpi.ca/ergast/f1/
"""

import time
import json
import logging
from pathlib import Path
from typing import Optional

import requests
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

BASE_URL = "https://api.jolpi.ca/ergast/f1"
CACHE_DIR = Path(__file__).parent.parent / "cache" / "jolpica"
RATE_LIMIT_DELAY = 0.5  # seconds between requests


class JolpicaClient:
    """Client for the Jolpica-F1 (Ergast successor) API."""

    def __init__(self, cache: bool = True):
        self.session = requests.Session()
        self.cache = cache
        if cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _get(self, endpoint: str, limit: int = 1000, offset: int = 0) -> dict:
        """Make a paginated API request with caching."""
        cache_key = endpoint.replace("/", "_").strip("_") + f"_L{limit}_O{offset}"
        cache_file = CACHE_DIR / f"{cache_key}.json"

        if self.cache and cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)

        url = f"{BASE_URL}/{endpoint}.json?limit={limit}&offset={offset}"
        logger.debug(f"GET {url}")

        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if self.cache:
            with open(cache_file, "w") as f:
                json.dump(data, f)

        time.sleep(RATE_LIMIT_DELAY)
        return data

    def _get_all(self, endpoint: str, table_key: str) -> list:
        """Paginate through all results for an endpoint."""
        all_items = []
        offset = 0
        limit = 1000

        first = self._get(endpoint, limit=limit, offset=0)
        mr = first["MRData"]
        total = int(mr["total"])

        # Find the table data — Ergast nests it under a *Table key
        table_data = None
        for key, val in mr.items():
            if key.endswith("Table"):
                table_data = val
                break

        if table_data is None:
            return []

        items = table_data.get(table_key, [])
        all_items.extend(items)
        offset += limit

        # Cache the table wrapper key from first response
        table_wrapper_key = next((k for k in mr if k.endswith("Table")), None)

        if total > limit and table_wrapper_key:
            pbar = tqdm(total=total, initial=len(items), desc=endpoint)
            while offset < total:
                page = self._get(endpoint, limit=limit, offset=offset)
                new_items = page["MRData"][table_wrapper_key].get(table_key, [])
                all_items.extend(new_items)
                pbar.update(len(new_items))
                offset += limit
            pbar.close()

        return all_items

    # ── Core Data Endpoints ──────────────────────────────────────────

    def get_seasons(self) -> pd.DataFrame:
        """All F1 seasons (1950–present)."""
        items = self._get_all("seasons", "Seasons")
        return pd.DataFrame(items)

    def get_circuits(self) -> pd.DataFrame:
        """All circuits ever used in F1."""
        items = self._get_all("circuits", "Circuits")
        rows = []
        for c in items:
            loc = c.get("Location", {})
            rows.append({
                "circuit_id": c["circuitId"],
                "circuit_name": c["circuitName"],
                "locality": loc.get("locality"),
                "country": loc.get("country"),
                "lat": float(loc.get("lat", 0)),
                "lng": float(loc.get("long", 0)),
                "url": c.get("url"),
            })
        return pd.DataFrame(rows)

    def get_drivers(self) -> pd.DataFrame:
        """All drivers in F1 history."""
        items = self._get_all("drivers", "Drivers")
        rows = []
        for d in items:
            rows.append({
                "driver_id": d["driverId"],
                "number": d.get("permanentNumber"),
                "code": d.get("code"),
                "first_name": d.get("givenName"),
                "last_name": d.get("familyName"),
                "dob": d.get("dateOfBirth"),
                "nationality": d.get("nationality"),
                "url": d.get("url"),
            })
        return pd.DataFrame(rows)

    def get_constructors(self) -> pd.DataFrame:
        """All constructors in F1 history."""
        items = self._get_all("constructors", "Constructors")
        rows = []
        for c in items:
            rows.append({
                "constructor_id": c["constructorId"],
                "name": c["name"],
                "nationality": c.get("nationality"),
                "url": c.get("url"),
            })
        return pd.DataFrame(rows)

    def get_race_results(self, season: int) -> pd.DataFrame:
        """Full race results for a season — positions, times, status, points."""
        items = self._get_all(f"{season}/results", "Races")
        rows = []
        for race in items:
            for result in race.get("Results", []):
                driver = result.get("Driver", {})
                constructor = result.get("Constructor", {})
                time_data = result.get("Time", {})
                fastest = result.get("FastestLap", {})
                fastest_time = fastest.get("Time", {})
                avg_speed = fastest.get("AverageSpeed", {})

                rows.append({
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "race_name": race["raceName"],
                    "circuit_id": race["Circuit"]["circuitId"],
                    "date": race["date"],
                    "driver_id": driver.get("driverId"),
                    "driver_code": driver.get("code"),
                    "constructor_id": constructor.get("constructorId"),
                    "grid": int(result.get("grid", 0)),
                    "position": int(result["position"]) if result.get("position", "").isdigit() else None,
                    "position_text": result.get("positionText"),
                    "points": float(result.get("points", 0)),
                    "laps": int(result.get("laps", 0)),
                    "status": result.get("status"),
                    "time_millis": int(time_data["millis"]) if "millis" in time_data else None,
                    "time_text": time_data.get("time"),
                    "fastest_lap": int(fastest.get("lap", 0)) if fastest.get("lap") else None,
                    "fastest_lap_rank": int(fastest.get("rank", 0)) if fastest.get("rank") else None,
                    "fastest_lap_time": fastest_time.get("time"),
                    "fastest_lap_avg_speed": float(avg_speed.get("speed", 0)) if avg_speed.get("speed") else None,
                })
        return pd.DataFrame(rows)

    def get_qualifying(self, season: int) -> pd.DataFrame:
        """Qualifying results for a season (available from ~2003+)."""
        items = self._get_all(f"{season}/qualifying", "Races")
        rows = []
        for race in items:
            for q in race.get("QualifyingResults", []):
                driver = q.get("Driver", {})
                constructor = q.get("Constructor", {})
                rows.append({
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "circuit_id": race["Circuit"]["circuitId"],
                    "driver_id": driver.get("driverId"),
                    "constructor_id": constructor.get("constructorId"),
                    "position": int(q.get("position", 0)),
                    "q1": q.get("Q1"),
                    "q2": q.get("Q2"),
                    "q3": q.get("Q3"),
                })
        return pd.DataFrame(rows)

    def get_sprint_results(self, season: int) -> pd.DataFrame:
        """Sprint race results (2021+)."""
        items = self._get_all(f"{season}/sprint", "Races")
        rows = []
        for race in items:
            for result in race.get("SprintResults", []):
                driver = result.get("Driver", {})
                constructor = result.get("Constructor", {})
                time_data = result.get("Time", {})
                rows.append({
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "circuit_id": race["Circuit"]["circuitId"],
                    "driver_id": driver.get("driverId"),
                    "constructor_id": constructor.get("constructorId"),
                    "grid": int(result.get("grid", 0)),
                    "position": int(result["position"]) if result.get("position", "").isdigit() else None,
                    "points": float(result.get("points", 0)),
                    "laps": int(result.get("laps", 0)),
                    "status": result.get("status"),
                    "time_millis": int(time_data["millis"]) if "millis" in time_data else None,
                })
        return pd.DataFrame(rows)

    def get_driver_standings(self, season: int) -> pd.DataFrame:
        """Driver championship standings after each round."""
        items = self._get_all(f"{season}/driverStandings", "StandingsLists")
        rows = []
        for standings_list in items:
            rnd = standings_list.get("round")
            for s in standings_list.get("DriverStandings", []):
                driver = s.get("Driver", {})
                constructors = s.get("Constructors", [{}])
                rows.append({
                    "season": int(standings_list["season"]),
                    "round": int(rnd) if rnd else None,
                    "driver_id": driver.get("driverId"),
                    "position": int(s.get("position", 0)),
                    "points": float(s.get("points", 0)),
                    "wins": int(s.get("wins", 0)),
                    "constructor_id": constructors[0].get("constructorId") if constructors else None,
                })
        return pd.DataFrame(rows)

    def get_constructor_standings(self, season: int) -> pd.DataFrame:
        """Constructor championship standings after each round."""
        items = self._get_all(f"{season}/constructorStandings", "StandingsLists")
        rows = []
        for standings_list in items:
            rnd = standings_list.get("round")
            for s in standings_list.get("ConstructorStandings", []):
                constructor = s.get("Constructor", {})
                rows.append({
                    "season": int(standings_list["season"]),
                    "round": int(rnd) if rnd else None,
                    "constructor_id": constructor.get("constructorId"),
                    "position": int(s.get("position", 0)),
                    "points": float(s.get("points", 0)),
                    "wins": int(s.get("wins", 0)),
                })
        return pd.DataFrame(rows)

    def get_pit_stops(self, season: int, race_round: int) -> pd.DataFrame:
        """Pit stop data for a specific race (2012+)."""
        items = self._get_all(f"{season}/{race_round}/pitstops", "Races")
        rows = []
        for race in items:
            for pit in race.get("PitStops", []):
                rows.append({
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "driver_id": pit.get("driverId"),
                    "stop": int(pit.get("stop", 0)),
                    "lap": int(pit.get("lap", 0)),
                    "time": pit.get("time"),
                    "duration": pit.get("duration"),
                })
        return pd.DataFrame(rows)

    def get_lap_times(self, season: int, race_round: int) -> pd.DataFrame:
        """Lap-by-lap times for a specific race (1996+ partial, 2011+ full)."""
        items = self._get_all(f"{season}/{race_round}/laps", "Races")
        rows = []
        for race in items:
            for lap_data in race.get("Laps", []):
                lap_num = int(lap_data["number"])
                for timing in lap_data.get("Timings", []):
                    rows.append({
                        "season": int(race["season"]),
                        "round": int(race["round"]),
                        "lap": lap_num,
                        "driver_id": timing.get("driverId"),
                        "position": int(timing.get("position", 0)),
                        "time": timing.get("time"),
                    })
        return pd.DataFrame(rows)

    # ── Bulk Ingestion ───────────────────────────────────────────────

    def ingest_all_seasons(
        self,
        start_year: int = 1950,
        end_year: int = 2025,
        output_dir: Optional[Path] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Download all historical data and save as parquet files.
        Returns dict of DataFrames.
        """
        out = output_dir or (Path(__file__).parent.parent / "cache" / "processed")
        out.mkdir(parents=True, exist_ok=True)

        logger.info("Fetching static reference data...")
        circuits = self.get_circuits()
        drivers = self.get_drivers()
        constructors = self.get_constructors()

        circuits.to_parquet(out / "circuits.parquet", index=False)
        drivers.to_parquet(out / "drivers.parquet", index=False)
        constructors.to_parquet(out / "constructors.parquet", index=False)

        all_results = []
        all_qualifying = []
        all_driver_standings = []
        all_constructor_standings = []
        all_sprints = []

        for year in tqdm(range(start_year, end_year + 1), desc="Seasons"):
            logger.info(f"Ingesting season {year}...")

            results = self.get_race_results(year)
            if not results.empty:
                all_results.append(results)

            if year >= 2003:
                quali = self.get_qualifying(year)
                if not quali.empty:
                    all_qualifying.append(quali)

            if year >= 2021:
                sprints = self.get_sprint_results(year)
                if not sprints.empty:
                    all_sprints.append(sprints)

            standings = self.get_driver_standings(year)
            if not standings.empty:
                all_driver_standings.append(standings)

            cstandings = self.get_constructor_standings(year)
            if not cstandings.empty:
                all_constructor_standings.append(cstandings)

        datasets = {}

        if all_results:
            df = pd.concat(all_results, ignore_index=True)
            df.to_parquet(out / "race_results.parquet", index=False)
            datasets["race_results"] = df
            logger.info(f"Race results: {len(df)} rows ({df['season'].min()}-{df['season'].max()})")

        if all_qualifying:
            df = pd.concat(all_qualifying, ignore_index=True)
            df.to_parquet(out / "qualifying.parquet", index=False)
            datasets["qualifying"] = df
            logger.info(f"Qualifying: {len(df)} rows")

        if all_sprints:
            df = pd.concat(all_sprints, ignore_index=True)
            df.to_parquet(out / "sprints.parquet", index=False)
            datasets["sprints"] = df
            logger.info(f"Sprints: {len(df)} rows")

        if all_driver_standings:
            df = pd.concat(all_driver_standings, ignore_index=True)
            df.to_parquet(out / "driver_standings.parquet", index=False)
            datasets["driver_standings"] = df

        if all_constructor_standings:
            df = pd.concat(all_constructor_standings, ignore_index=True)
            df.to_parquet(out / "constructor_standings.parquet", index=False)
            datasets["constructor_standings"] = df

        datasets["circuits"] = circuits
        datasets["drivers"] = drivers
        datasets["constructors"] = constructors

        logger.info(f"Ingestion complete. {len(datasets)} datasets saved to {out}")
        return datasets


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = JolpicaClient()
    datasets = client.ingest_all_seasons(start_year=1950, end_year=2025)
    for name, df in datasets.items():
        print(f"  {name}: {len(df)} rows, {len(df.columns)} columns")
