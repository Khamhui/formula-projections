"""
OpenF1 API client — real-time and recent historical F1 data (2023+).
18 endpoints covering telemetry, positions, overtakes, weather, pit stops.
Used for live dashboard and enriching recent prediction features.
"""

import logging
from typing import Optional, Any

import requests
import pandas as pd

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openf1.org/v1"


class OpenF1Client:
    """Client for the OpenF1 real-time API."""

    def __init__(self):
        self.session = requests.Session()

    def _get(self, endpoint: str, params: Optional[dict] = None) -> list[dict]:
        """Make an API request."""
        url = f"{BASE_URL}/{endpoint}"
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _get_df(self, endpoint: str, params: Optional[dict] = None) -> pd.DataFrame:
        """Get endpoint data as DataFrame."""
        data = self._get(endpoint, params)
        return pd.DataFrame(data) if data else pd.DataFrame()

    # ── Session Discovery ────────────────────────────────────────────

    def get_meetings(self, year: Optional[int] = None) -> pd.DataFrame:
        """Get all race meetings (weekends)."""
        params = {"year": year} if year else {}
        return self._get_df("meetings", params)

    def get_sessions(self, year: Optional[int] = None, session_type: Optional[str] = None) -> pd.DataFrame:
        """Get sessions. session_type: Race, Qualifying, Practice, Sprint."""
        params = {}
        if year:
            params["year"] = year
        if session_type:
            params["session_type"] = session_type
        return self._get_df("sessions", params)

    def get_drivers(self, session_key: int) -> pd.DataFrame:
        """Get drivers for a specific session."""
        return self._get_df("drivers", {"session_key": session_key})

    # ── Race Data ────────────────────────────────────────────────────

    def get_laps(self, session_key: int, driver_number: Optional[int] = None) -> pd.DataFrame:
        """Lap times with sector splits and speed traps."""
        params: dict[str, Any] = {"session_key": session_key}
        if driver_number:
            params["driver_number"] = driver_number
        return self._get_df("laps", params)

    def get_positions(self, session_key: int, driver_number: Optional[int] = None) -> pd.DataFrame:
        """Real-time position changes during session."""
        params: dict[str, Any] = {"session_key": session_key}
        if driver_number:
            params["driver_number"] = driver_number
        return self._get_df("position", params)

    def get_intervals(self, session_key: int) -> pd.DataFrame:
        """Time intervals and gaps to leader."""
        return self._get_df("intervals", {"session_key": session_key})

    def get_overtakes(self, session_key: int) -> pd.DataFrame:
        """Overtake events during session."""
        return self._get_df("overtakes", {"session_key": session_key})

    def get_pit_stops(self, session_key: int) -> pd.DataFrame:
        """Pit stop data with durations."""
        return self._get_df("pit", {"session_key": session_key})

    def get_stints(self, session_key: int) -> pd.DataFrame:
        """Tire stints — compound, start/end laps, tire age."""
        return self._get_df("stints", {"session_key": session_key})

    def get_starting_grid(self, session_key: int) -> pd.DataFrame:
        """Starting grid positions."""
        return self._get_df("starting_grid", {"session_key": session_key})

    def get_session_results(self, session_key: int) -> pd.DataFrame:
        """Final session results — positions, gaps, DNFs."""
        return self._get_df("session_result", {"session_key": session_key})

    # ── Telemetry ────────────────────────────────────────────────────

    def get_car_data(self, session_key: int, driver_number: int) -> pd.DataFrame:
        """
        Raw car telemetry at ~3.7Hz.
        Fields: speed, rpm, n_gear, throttle, brake, drs.
        WARNING: Returns large datasets. Use for specific analyses.
        """
        return self._get_df("car_data", {
            "session_key": session_key,
            "driver_number": driver_number,
        })

    def get_location(self, session_key: int, driver_number: int) -> pd.DataFrame:
        """Car position on track (x, y, z coordinates)."""
        return self._get_df("location", {
            "session_key": session_key,
            "driver_number": driver_number,
        })

    # ── Conditions ───────────────────────────────────────────────────

    def get_weather(self, session_key: int) -> pd.DataFrame:
        """
        Weather data: air/track temp, humidity, pressure, rainfall, wind.
        """
        return self._get_df("weather", {"session_key": session_key})

    def get_race_control(self, session_key: int) -> pd.DataFrame:
        """Race control messages — flags, safety car, incidents."""
        return self._get_df("race_control", {"session_key": session_key})

    # ── Championships ────────────────────────────────────────────────

    def get_championship_drivers(self, session_key: int) -> pd.DataFrame:
        """Driver championship standings at session."""
        return self._get_df("championship_drivers", {"session_key": session_key})

    def get_championship_teams(self, session_key: int) -> pd.DataFrame:
        """Constructor championship standings at session."""
        return self._get_df("championship_teams", {"session_key": session_key})

    # ── Audio ────────────────────────────────────────────────────────

    def get_team_radio(self, session_key: int, driver_number: Optional[int] = None) -> pd.DataFrame:
        """Team radio recordings."""
        params: dict[str, Any] = {"session_key": session_key}
        if driver_number:
            params["driver_number"] = driver_number
        return self._get_df("team_radio", params)


if __name__ == "__main__":
    client = OpenF1Client()

    # Example: get 2024 meetings
    meetings = client.get_meetings(year=2024)
    print(f"2024 meetings: {len(meetings)}")
    if not meetings.empty:
        print(meetings[["meeting_name", "country_name", "date_start"]].head())

    # Get a race session
    sessions = client.get_sessions(year=2024, session_type="Race")
    if not sessions.empty:
        sk = sessions.iloc[0]["session_key"]
        print(f"\nSession {sk}:")
        results = client.get_session_results(sk)
        print(results[["driver_number", "position"]].head(10))
