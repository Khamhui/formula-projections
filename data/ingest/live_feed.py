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
from typing import Dict, List, Optional, Tuple

import pandas as pd

from data.models.live import RaceState, DriverState

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL = 5.0

FLAG_MAP = {
    "green": "clear", "clear": "clear",
    "yellow": "clear", "double yellow": "clear",
    "safety car": "sc", "sc": "sc",
    "virtual safety car": "vsc", "vsc": "vsc",
    "red": "red", "red flag": "red",
}

# 2026 grid — update when drivers change numbers
DRIVER_NUMBER_MAP: Dict[int, str] = {
    1: "max_verstappen", 4: "norris", 16: "leclerc", 44: "hamilton",
    63: "russell", 81: "piastri", 14: "alonso", 18: "stroll",
    10: "gasly", 31: "ocon", 23: "albon", 55: "sainz",
    27: "hulkenberg", 87: "bearman", 12: "antonelli",
    6: "hadjar", 30: "lawson", 43: "colapinto", 5: "bortoleto",
    77: "bottas", 11: "perez", 40: "arvid_lindblad",
}

# 3-letter codes for track map labels
DRIVER_CODES: Dict[str, str] = {
    "max_verstappen": "VER", "norris": "NOR", "leclerc": "LEC",
    "hamilton": "HAM", "russell": "RUS", "piastri": "PIA",
    "alonso": "ALO", "stroll": "STR", "gasly": "GAS", "ocon": "OCO",
    "albon": "ALB", "sainz": "SAI", "hulkenberg": "HUL",
    "bottas": "BOT", "perez": "PER",
    "bearman": "BEA", "antonelli": "ANT", "hadjar": "HAD", "lawson": "LAW",
    "colapinto": "COL", "bortoleto": "BOR", "arvid_lindblad": "LIN",
}

# Driver -> constructor mapping for team colors
DRIVER_CONSTRUCTOR: Dict[str, str] = {
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
        self._track_outline: Optional[List[Tuple[float, float]]] = None
        self._total_laps: int = 0
        self._retired_drivers: set = set()
        self._poll_count: int = 0
        # Cache for infrequently-changing data (stints, race_control, weather)
        self._cached_stints: Optional[pd.DataFrame] = None
        self._cached_race_control: Optional[pd.DataFrame] = None
        self._cached_weather: Optional[pd.DataFrame] = None
        # Telemetry tracking — only poll car_data for these driver numbers
        self._tracked_drivers: List[int] = [16, 1, 4]  # default: Leclerc, Verstappen, Norris
        # F1 Live Timing enhancement layer (optional)
        self._f1_live_client = None
        self._f1_live_enabled = False
        # Battery SOC estimation
        self._battery_estimator = None
        self._battery_import_failed = False

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
        """Find the currently active race session."""
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

    def get_track_outline(self) -> List[Tuple[float, float]]:
        """
        Fetch the track outline from one driver's location data.
        Cached per session — only fetched once.

        Uses lap boundary timestamps to extract a single clean racing lap,
        avoiding pit stops and safety car periods.

        Returns list of (x, y) tuples tracing the circuit shape.
        """
        if self._track_outline is not None:
            return self._track_outline

        if not self.session_key:
            return []

        client = self._get_client()
        try:
            # Try multiple drivers in case the leader has incomplete data
            driver_number = None
            loc = pd.DataFrame()
            for dn in [1, 4, 16, 44, 63]:
                loc = client.get_location(session_key=self.session_key, driver_number=dn)
                if not loc.empty and "x" in loc.columns:
                    driver_number = dn
                    break

            if loc.empty or "x" not in loc.columns:
                return []

            # Filter out stationary points (pit/grid)
            moving = loc[(loc["x"] != 0) | (loc["y"] != 0)].copy()
            if moving.empty:
                return []

            # Use lap timestamps to find a clean racing lap (mid-race, not lap 1)
            lap_points = self._extract_clean_lap(client, driver_number, moving)

            # Downsample to ~200 points for the outline
            step = max(1, len(lap_points) // 200)
            sampled = lap_points.iloc[::step]

            self._track_outline = list(zip(
                sampled["x"].astype(float).tolist(),
                sampled["y"].astype(float).tolist(),
            ))
            logger.info("Track outline cached: %d points", len(self._track_outline))
            return self._track_outline

        except Exception as e:
            logger.warning("Track outline fetch failed: %s", e)
            return []

    def _extract_clean_lap(
        self, client, driver_number: int, moving: pd.DataFrame,
    ) -> pd.DataFrame:
        """Extract location points for a single clean racing lap.

        Tries to use lap boundary timestamps from the laps endpoint.
        Falls back to a mid-session window if lap data is unavailable.
        """
        try:
            laps = client.get_laps(
                session_key=self.session_key, driver_number=driver_number,
            )
            if (
                not laps.empty
                and "date_start" in laps.columns
                and "lap_number" in laps.columns
                and "date" in moving.columns
            ):
                # Pick a lap from the middle third of the race (clean racing conditions)
                valid_laps = laps.dropna(subset=["date_start"])
                if len(valid_laps) >= 6:
                    n = len(valid_laps)
                    mid_laps = valid_laps.iloc[n // 3 : 2 * n // 3]
                    # Prefer shortest lap in the middle third (fastest = cleanest)
                    if "lap_duration" in mid_laps.columns:
                        timed = mid_laps.dropna(subset=["lap_duration"])
                        if not timed.empty:
                            target_lap = timed.loc[timed["lap_duration"].idxmin()]
                        else:
                            target_lap = mid_laps.iloc[len(mid_laps) // 2]
                    else:
                        target_lap = mid_laps.iloc[len(mid_laps) // 2]

                    lap_start = pd.to_datetime(target_lap["date_start"], format="mixed")
                    moving_dt = pd.to_datetime(moving["date"], format="mixed")

                    # A lap is ~60-120 seconds; take 120s window from lap start
                    lap_end = lap_start + pd.Timedelta(seconds=120)
                    mask = (moving_dt >= lap_start) & (moving_dt <= lap_end)
                    lap_data = moving.loc[mask]

                    if len(lap_data) >= 100:
                        return lap_data
        except Exception as e:
            logger.debug("Lap-based outline extraction failed: %s", e)

        # Fallback: mid-session window (~500 points)
        n = len(moving)
        mid = n // 2
        return moving.iloc[mid:mid + 500]

    def get_driver_locations(self) -> Dict[str, Tuple[float, float]]:
        """
        Fetch the latest x, y position for each driver on track.

        Uses a single bulk API call instead of per-driver requests.
        Returns dict of driver_id -> (x, y).
        """
        if not self.session_key:
            return {}

        client = self._get_client()
        locations: Dict[str, Tuple[float, float]] = {}

        try:
            all_loc = client.get_location(session_key=self.session_key)
            if all_loc.empty or "x" not in all_loc.columns or "driver_number" not in all_loc.columns:
                return {}

            for driver_num, group in all_loc.groupby("driver_number"):
                driver_num = int(driver_num)
                driver_id = DRIVER_NUMBER_MAP.get(driver_num)
                if driver_id is None or driver_id in self._retired_drivers:
                    continue
                latest = group.iloc[-1]
                x, y = float(latest["x"]), float(latest["y"])
                if x != 0 or y != 0:
                    locations[driver_id] = (x, y)
        except Exception as e:
            logger.warning("Location fetch failed: %s", e)

        return locations

    def poll_once(self) -> Optional[RaceState]:
        """
        Poll OpenF1 once and build a RaceState snapshot.
        """
        if not self.session_key:
            self.discover_session()
        if not self.session_key:
            return None

        client = self._get_client()
        state = RaceState()

        try:
            # Fetch positions
            positions_df = client.get_positions(session_key=self.session_key)
            if positions_df.empty:
                return None

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

            # Fetch laps — lap count + per-driver lap/sector times
            laps_df = client.get_laps(session_key=self.session_key)
            latest_laps: Dict[int, pd.Series] = {}
            best_laps: Dict[int, float] = {}
            if not laps_df.empty and "lap_number" in laps_df.columns:
                state.lap = int(laps_df["lap_number"].max())
                if "driver_number" in laps_df.columns:
                    for driver_num, group in laps_df.groupby("driver_number"):
                        dn = int(driver_num)
                        latest_laps[dn] = group.iloc[-1]
                        if "lap_duration" in group.columns:
                            valid = group["lap_duration"].dropna()
                            if not valid.empty:
                                best_laps[dn] = float(valid.min())

            # Total laps (detect once)
            if self._total_laps == 0:
                try:
                    sessions = client.get_sessions(year=pd.Timestamp.now().year, session_type="Race")
                    if not sessions.empty:
                        for _, sess in sessions.iterrows():
                            if sess.get("session_key") == self.session_key:
                                tl = sess.get("total_laps")
                                if tl and int(tl) > 0:
                                    self._total_laps = int(tl)
                                break
                except Exception:
                    pass
                if self._total_laps == 0:
                    self._total_laps = 57  # sensible default
            state.total_laps = self._total_laps

            # Stints, race control, and weather change infrequently — poll every
            # 6th cycle (~30s) instead of every 5s to reduce API load
            slow_poll = (self._poll_count % 6 == 0)

            if slow_poll or self._cached_stints is None:
                self._cached_stints = client.get_stints(session_key=self.session_key)
            stints_df = self._cached_stints

            latest_stints: Dict[int, pd.Series] = {}
            if not stints_df.empty and "driver_number" in stints_df.columns:
                for driver_num, group in stints_df.groupby("driver_number"):
                    latest_stints[int(driver_num)] = group.iloc[-1]

            if slow_poll or self._cached_race_control is None:
                self._cached_race_control = client.get_race_control(session_key=self.session_key)
            race_control_df = self._cached_race_control

            if not race_control_df.empty:
                latest_rc = race_control_df.iloc[-1]
                flag = str(latest_rc.get("flag", "")).lower()
                state.track_status = FLAG_MAP.get(flag, "clear")
                if state.track_status == "clear":
                    for key, val in FLAG_MAP.items():
                        if key in flag:
                            state.track_status = val
                            break

                if "message" in race_control_df.columns:
                    for _, rc in race_control_df.iterrows():
                        msg = str(rc.get("message", "")).upper()
                        if "RETIRED" in msg or "OUT OF THE RACE" in msg or "STOPPED" in msg:
                            dn = rc.get("driver_number")
                            if dn is not None:
                                driver_id = DRIVER_NUMBER_MAP.get(int(dn))
                                if driver_id:
                                    self._retired_drivers.add(driver_id)

            if slow_poll or self._cached_weather is None:
                try:
                    self._cached_weather = client.get_weather(session_key=self.session_key)
                except Exception:
                    self._cached_weather = pd.DataFrame()
            weather_df = self._cached_weather

            if not weather_df.empty:
                latest_w = weather_df.iloc[-1]
                state.air_temp = float(latest_w.get("air_temperature", 0) or 0)
                state.track_temp = float(latest_w.get("track_temperature", 0) or 0)
                state.rainfall = bool(latest_w.get("rainfall", False))

            # Build driver states
            for driver_num, pos_data in latest_positions.items():
                driver_id = DRIVER_NUMBER_MAP.get(driver_num, str(driver_num))

                ds = DriverState(driver_id)
                ds.position = int(pos_data.get("position", 0))
                ds.is_retired = driver_id in self._retired_drivers

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

                # Lap times + sector times
                if driver_num in latest_laps:
                    lap = latest_laps[driver_num]
                    dur = lap.get("lap_duration")
                    if dur is not None:
                        try:
                            ds.last_lap_time = float(dur)
                        except (ValueError, TypeError):
                            pass
                    if driver_num in best_laps:
                        ds.best_lap_time = best_laps[driver_num]
                    # Sector times
                    for s in [1, 2, 3]:
                        st = lap.get(f"duration_sector_{s}")
                        if st is not None:
                            try:
                                setattr(ds, f"sector{s}", float(st))
                            except (ValueError, TypeError):
                                pass

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

            # Cache track outline on first successful poll
            if self._track_outline is None:
                self.get_track_outline()
            state.track_outline = self._track_outline or []

            # Fetch driver locations for track map
            state.driver_locations = self.get_driver_locations()

            # Fetch car telemetry for tracked drivers
            self._poll_telemetry(state)

            self._update_battery_soc(state)

            self._poll_count += 1

            with self._lock:
                self._current_state = state

            return state

        except Exception as e:
            logger.error("Poll failed: %s", e)
            return None

    def start_polling(self):
        """Start live data via F1 Live Timing SignalR (free, works during races)."""
        if self._polling:
            return

        self._polling = True
        if self.enable_f1_live_timing():
            logger.info("F1 Live Timing connected — live data active")
        else:
            logger.error("F1 Live Timing failed to connect")

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

    def set_tracked_drivers(self, driver_numbers: List[int]):
        """Set which drivers to poll telemetry for (max 3)."""
        self._tracked_drivers = driver_numbers[:3]
        if self._f1_live_client:
            self._f1_live_client.set_tracked_drivers(self._tracked_drivers)
        logger.info("Tracked drivers set to: %s", self._tracked_drivers)

    def get_tracked_drivers(self) -> List[int]:
        return list(self._tracked_drivers)

    def _poll_telemetry(self, state: RaceState):
        """Fetch latest car telemetry for tracked drivers and update their DriverState."""
        if not self.session_key or not self._tracked_drivers:
            return

        client = self._get_client()
        cutoff = (
            pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=10)
        ).strftime("%Y-%m-%dT%H:%M:%S")

        for driver_num in self._tracked_drivers:
            driver_id = DRIVER_NUMBER_MAP.get(driver_num)
            if driver_id is None or driver_id not in state.drivers:
                continue

            try:
                car_data = client.get_car_data(
                    session_key=self.session_key,
                    driver_number=driver_num,
                    date_gt=cutoff,
                )
                if car_data.empty:
                    continue

                latest = car_data.iloc[-1]
                ds = state.drivers[driver_id]
                ds.speed = int(latest.get("speed", 0) or 0)
                ds.rpm = int(latest.get("rpm", 0) or 0)
                ds.gear = int(latest.get("n_gear", 0) or 0)
                ds.throttle = int(latest.get("throttle", 0) or 0)
                brake_val = latest.get("brake", 0)
                ds.brake = int(brake_val) if brake_val else 0
                ds.drs = int(latest.get("drs", 0) or 0)
            except Exception as e:
                logger.debug("Telemetry fetch failed for driver %d: %s", driver_num, e)

    def _update_battery_soc(self, state: RaceState):
        """Update battery SOC estimates for tracked drivers."""
        if self._battery_estimator is None:
            if self._battery_import_failed:
                return
            try:
                from data.models.battery_model import BatteryEstimator
                self._battery_estimator = BatteryEstimator()
            except ImportError:
                self._battery_import_failed = True
                return

        for driver_num in self._tracked_drivers:
            driver_id = DRIVER_NUMBER_MAP.get(driver_num)
            if driver_id is None or driver_id not in state.drivers:
                continue
            ds = state.drivers[driver_id]
            if ds.speed == 0 and ds.throttle == 0:
                continue
            ds.battery_soc = self._battery_estimator.update(
                driver_id=driver_id,
                throttle=ds.throttle,
                brake=ds.brake,
                speed=ds.speed,
                overtake_active=ds.overtake_active,
                dt=self.poll_interval,
            )

    def enable_f1_live_timing(self) -> bool:
        """Start the F1 Live Timing SignalR client as an enhancement layer."""
        if self._f1_live_enabled:
            return True

        try:
            from data.ingest.f1_live_timing import F1LiveTimingClient

            self._f1_live_client = F1LiveTimingClient(
                tracked_driver_numbers=self._tracked_drivers,
                driver_number_map=DRIVER_NUMBER_MAP,
            )
            # Core race data callbacks
            self._f1_live_client.on_timing_data(self._on_f1_live_timing_data)
            self._f1_live_client.on_lap_count(self._on_f1_live_lap_count)
            self._f1_live_client.on_weather(self._on_f1_live_weather)
            self._f1_live_client.on_positions(self._on_f1_live_positions)
            self._f1_live_client.on_race_control(self._on_f1_live_race_control)
            # Enhancement callbacks
            self._f1_live_client.on_car_status(self._on_f1_live_car_status)
            self._f1_live_client.on_timing_app(self._on_f1_live_timing_app)

            if self._f1_live_client.start():
                self._f1_live_enabled = True
                logger.info("F1 Live Timing enhancement layer enabled")
                return True
            else:
                logger.warning("F1 Live Timing failed to connect — OpenF1 only")
                return False
        except Exception as e:
            logger.warning("F1 Live Timing unavailable: %s — OpenF1 only", e)
            return False

    def disable_f1_live_timing(self):
        """Stop the F1 Live Timing client."""
        if self._f1_live_client:
            self._f1_live_client.stop()
            self._f1_live_client = None
        self._f1_live_enabled = False
        logger.info("F1 Live Timing disabled")

    def _get_live_driver_state(self, data: dict) -> Optional[DriverState]:
        """Resolve a live timing callback to its DriverState under lock. Returns None if not found."""
        state = self._current_state
        if not state:
            return None
        driver_id = data.get("driver_id")
        if driver_id not in state.drivers:
            return None
        return state.drivers[driver_id]

    def _ensure_signalr_state(self):
        """Ensure a RaceState exists for SignalR data to write into."""
        with self._lock:
            if self._current_state is None:
                state = RaceState()
                state.total_laps = self._total_laps or 53
                self._current_state = state

    def _on_f1_live_timing_data(self, data: dict):
        """Core timing: positions, gaps, lap times for all drivers."""
        self._ensure_signalr_state()
        with self._lock:
            state = self._current_state
            driver_id = data.get("driver_id")
            if not driver_id:
                return

            if driver_id not in state.drivers:
                state.drivers[driver_id] = DriverState(driver_id)
            ds = state.drivers[driver_id]

            if "position" in data and data["position"] is not None:
                ds.position = data["position"]
            if "gap_to_leader" in data:
                ds.gap_to_leader = data["gap_to_leader"]
            if "gap_to_ahead" in data:
                ds.gap_to_ahead = data["gap_to_ahead"]
            if "is_in_pit" in data:
                ds.is_in_pit = data["is_in_pit"]
            if "is_retired" in data:
                ds.is_retired = True
                self._retired_drivers.add(driver_id)

            if "last_lap_time_str" in data:
                try:
                    parts = data["last_lap_time_str"].split(":")
                    if len(parts) == 2:
                        ds.last_lap_time = float(parts[0]) * 60 + float(parts[1])
                    elif len(parts) == 1 and parts[0]:
                        ds.last_lap_time = float(parts[0])
                except (ValueError, IndexError):
                    pass

    def _on_f1_live_lap_count(self, data: dict):
        """Update current lap and total laps."""
        self._ensure_signalr_state()
        with self._lock:
            state = self._current_state
            if "lap" in data:
                state.lap = data["lap"]
            if "total_laps" in data:
                state.total_laps = data["total_laps"]
                self._total_laps = data["total_laps"]

    def _on_f1_live_weather(self, data: dict):
        """Update weather conditions."""
        self._ensure_signalr_state()
        with self._lock:
            state = self._current_state
            if "air_temp" in data:
                state.air_temp = data["air_temp"]
            if "track_temp" in data:
                state.track_temp = data["track_temp"]
            if "rainfall" in data:
                state.rainfall = data["rainfall"]

    def _on_f1_live_positions(self, positions: list):
        """Update GPS positions for track map."""
        self._ensure_signalr_state()
        with self._lock:
            state = self._current_state
            for pos in positions:
                did = pos.get("driver_id")
                if did:
                    state.driver_locations[did] = (pos["x"], pos["y"])

    def _on_f1_live_race_control(self, data: dict):
        """Update track status from race control messages."""
        self._ensure_signalr_state()
        with self._lock:
            state = self._current_state
            flag = str(data.get("Flag", data.get("flag", ""))).lower()
            status = FLAG_MAP.get(flag, None)
            if status is None:
                for key, val in FLAG_MAP.items():
                    if key in flag:
                        status = val
                        break
            if status:
                state.track_status = status

            msg = str(data.get("Message", "")).upper()
            if "RETIRED" in msg or "OUT OF THE RACE" in msg or "STOPPED" in msg:
                # Try to extract driver number from message
                for num, did in DRIVER_NUMBER_MAP.items():
                    code = DRIVER_CODES.get(did, "")
                    if code and code in msg:
                        self._retired_drivers.add(did)
                        if did in state.drivers:
                            state.drivers[did].is_retired = True

    def _on_f1_live_car_status(self, data: dict):
        """Callback from F1 Live Timing — ERS, overtake mode, brake %."""
        with self._lock:
            ds = self._get_live_driver_state(data)
            if not ds:
                return
            if "ers_deploy" in data:
                ds.ers_deploy = data["ers_deploy"]
            if "overtake_active" in data:
                ds.overtake_active = data["overtake_active"]
            if "brake_pct" in data:
                ds.brake = data["brake_pct"]

    def _on_f1_live_timing_app(self, data: dict):
        """Callback from F1 Live Timing — tire pressures/temps, lap delta."""
        with self._lock:
            ds = self._get_live_driver_state(data)
            if not ds:
                return
            for corner in ["fl", "fr", "rl", "rr"]:
                press = data.get(f"tire_pressure_{corner}")
                if press is not None:
                    setattr(ds, f"tire_pressure_{corner}", press)
                temp = data.get(f"tire_temp_{corner}")
                if temp is not None:
                    setattr(ds, f"tire_temp_{corner}", temp)

    @property
    def f1_live_enabled(self) -> bool:
        return self._f1_live_enabled


def create_race_state_from_dict(data: Dict) -> RaceState:
    """
    Create a RaceState from a flat dict (for testing / manual input).
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
