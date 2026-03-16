"""
OpenF1 Live Data Feed — polls real-time race data and builds RaceState objects.

Wraps the OpenF1 REST API to produce structured RaceState snapshots
for the LiveRacePredictor.

Usage:
    feed = LiveFeed()
    feed.start_polling()
    state = feed.get_current_state()
"""

from __future__ import annotations

import logging
import time
import threading
from typing import Dict, List, Optional

import pandas as pd

from data.models.live import RaceState, DriverState

logger = logging.getLogger(__name__)

# Default polling interval in seconds
DEFAULT_POLL_INTERVAL = 5.0

# OpenF1 driver numbers -> internal driver_id (2024-2025 grid)
# Must be updated when drivers change numbers or new drivers join
DRIVER_NUMBER_MAP: Dict[int, str] = {
    1: "max_verstappen", 4: "norris", 16: "leclerc", 55: "sainz",
    44: "hamilton", 63: "russell", 14: "alonso", 18: "stroll",
    81: "piastri", 10: "gasly", 31: "ocon", 22: "tsunoda",
    3: "ricciardo", 11: "perez", 27: "hulkenberg", 20: "magnussen",
    77: "bottas", 24: "zhou", 23: "albon", 2: "sargeant",
    43: "colapinto", 30: "lawson", 87: "bearman", 12: "doohan",
    38: "drugovich", 61: "antonelli",
}


class LiveFeed:
    """
    Polls OpenF1 API and builds RaceState snapshots.

    OpenF1 is a REST API (not streaming), so we poll at regular intervals
    and aggregate the data into a structured race state.
    """

    def __init__(
        self,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        session_key: Optional[int] = None,
    ):
        self.poll_interval = poll_interval
        self.session_key = session_key
        self._current_state: Optional[RaceState] = None
        self._polling = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._client = None

    def _get_client(self):
        """Lazy-load OpenF1 client."""
        if self._client is None:
            try:
                from data.ingest.openf1_client import OpenF1Client
                self._client = OpenF1Client()
            except ImportError:
                logger.error("OpenF1Client not available")
                raise
        return self._client

    def discover_session(self) -> Optional[int]:
        """
        Find the currently active race session.

        Returns:
            Session key if a live session is found, None otherwise
        """
        client = self._get_client()
        try:
            sessions = client.get_sessions(
                year=pd.Timestamp.now().year,
                session_type="Race",
            )
            if isinstance(sessions, pd.DataFrame) and not sessions.empty:
                latest = sessions.iloc[-1]
                key = latest.get("session_key")
                if key is not None:
                    self.session_key = int(key)
                    logger.info("Discovered session key: %d", self.session_key)
                    return self.session_key
        except Exception as e:
            logger.warning("Session discovery failed: %s", e)
        return None

    def poll_once(self) -> Optional[RaceState]:
        """
        Poll OpenF1 once and build a RaceState snapshot.

        Returns:
            RaceState if data is available, None otherwise
        """
        if not self.session_key:
            self.discover_session()
        if not self.session_key:
            return None

        client = self._get_client()
        state = RaceState()

        try:
            # Fetch positions (returns DataFrame)
            positions_df = client.get_positions(session_key=self.session_key)
            if positions_df.empty:
                return None

            # Get latest position per driver
            latest_positions: Dict[int, pd.Series] = {}
            if "driver_number" in positions_df.columns:
                for driver_num, group in positions_df.groupby("driver_number"):
                    latest_positions[int(driver_num)] = group.iloc[-1]

            # Fetch intervals (gaps)
            intervals_df = client.get_intervals(session_key=self.session_key)
            latest_intervals: Dict[int, pd.Series] = {}
            if not intervals_df.empty and "driver_number" in intervals_df.columns:
                for driver_num, group in intervals_df.groupby("driver_number"):
                    latest_intervals[int(driver_num)] = group.iloc[-1]

            # Fetch laps for lap count
            laps_df = client.get_laps(session_key=self.session_key)
            if not laps_df.empty and "lap_number" in laps_df.columns:
                state.lap = int(laps_df["lap_number"].max())

            # Fetch stints for tire info
            stints_df = client.get_stints(session_key=self.session_key)
            latest_stints: Dict[int, pd.Series] = {}
            if not stints_df.empty and "driver_number" in stints_df.columns:
                for driver_num, group in stints_df.groupby("driver_number"):
                    latest_stints[int(driver_num)] = group.iloc[-1]

            # Fetch race control for track status
            race_control_df = client.get_race_control(session_key=self.session_key)
            if not race_control_df.empty:
                latest_rc = race_control_df.iloc[-1]
                flag = str(latest_rc.get("flag", "")).lower()
                if "safety" in flag:
                    state.track_status = "sc"
                elif "virtual" in flag:
                    state.track_status = "vsc"
                elif "red" in flag:
                    state.track_status = "red"
                else:
                    state.track_status = "clear"

            # Build driver states
            for driver_num, pos_data in latest_positions.items():
                driver_id = DRIVER_NUMBER_MAP.get(driver_num, str(driver_num))

                ds = DriverState(driver_id)
                ds.position = int(pos_data.get("position", 0))

                # Interval data
                if driver_num in latest_intervals:
                    iv = latest_intervals[driver_num]
                    gap = iv.get("gap_to_leader")
                    if gap is not None:
                        try:
                            ds.gap_to_leader = float(gap) if gap != "" else 0.0
                        except (ValueError, TypeError):
                            ds.gap_to_leader = 0.0
                    interval = iv.get("interval")
                    if interval is not None:
                        try:
                            ds.gap_to_ahead = float(interval) if interval != "" else 0.0
                        except (ValueError, TypeError):
                            ds.gap_to_ahead = 0.0

                # Stint / tire data
                if driver_num in latest_stints:
                    stint = latest_stints[driver_num]
                    ds.tire_compound = str(stint.get("compound", "unknown")).lower()
                    ds.tire_age = int(stint.get("tyre_age_at_start", 0) or 0)
                    if state.lap > 0:
                        stint_start_lap = int(stint.get("lap_start", 0) or 0)
                        ds.tire_age += max(0, state.lap - stint_start_lap)
                    ds.pits_completed = int(stint.get("stint_number", 1) or 1) - 1

                state.drivers[driver_id] = ds

            with self._lock:
                self._current_state = state

            return state

        except Exception as e:
            logger.error("Poll failed: %s", e)
            return None

    def start_polling(self):
        """Start background polling thread."""
        if self._polling:
            return

        self._polling = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("Live feed polling started (interval=%.1fs)", self.poll_interval)

    def stop_polling(self):
        """Stop background polling."""
        self._polling = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Live feed polling stopped")

    def _poll_loop(self):
        """Background polling loop."""
        while self._polling:
            self.poll_once()
            time.sleep(self.poll_interval)

    def get_current_state(self) -> Optional[RaceState]:
        """Thread-safe access to the latest race state."""
        with self._lock:
            return self._current_state


def create_race_state_from_dict(data: Dict) -> RaceState:
    """
    Create a RaceState from a flat dict (for testing / manual input).

    Args:
        data: Dict with keys:
            lap, total_laps, track_status,
            drivers: list of dicts with driver_id, position, gap_to_leader, etc.
    """
    state = RaceState()
    state.lap = data.get("lap", 0)
    state.total_laps = data.get("total_laps", 57)
    state.track_status = data.get("track_status", "clear")

    for driver_data in data.get("drivers", []):
        ds = DriverState(driver_data["driver_id"])
        ds.position = driver_data.get("position", 0)
        ds.gap_to_leader = driver_data.get("gap_to_leader", 0.0)
        ds.gap_to_ahead = driver_data.get("gap_to_ahead", 0.0)
        ds.tire_compound = driver_data.get("tire_compound", "unknown")
        ds.tire_age = driver_data.get("tire_age", 0)
        ds.pits_completed = driver_data.get("pits_completed", 0)
        ds.last_lap_time = driver_data.get("last_lap_time", 0.0)
        ds.is_retired = driver_data.get("is_retired", False)
        state.drivers[ds.driver_id] = ds

    return state
