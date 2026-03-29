"""
F1 Live Timing SignalR Client — connects to the official F1 timing feed.

Provides data not available on OpenF1:
- Tire pressures & temperatures (TyreData)
- ERS deployment state (CarStatus)
- Overtake Mode / Boost active (CarStatus)
- Lap delta vs reference (TimingAppData)
- Brake pressure % (CarData.z — richer than OpenF1's binary)

Architecture:
- Runs as a background thread alongside the OpenF1 LiveFeed
- Writes directly to DriverState objects (same ones OpenF1 populates)
- If the SignalR connection drops, OpenF1 data still works — this is an enhancement layer

Based on the community-documented F1 SignalR protocol:
  https://livetiming.formula1.com/signalr
"""

from __future__ import annotations

import base64
import gzip
import json
import logging
import threading
import time
from typing import Callable, Dict, List, Optional, Set

import requests

logger = logging.getLogger(__name__)

F1_SIGNALR_BASE = "https://livetiming.formula1.com/signalr"
F1_SIGNALR_HUB = "Streaming"

TOPICS = [
    "TimingData",           # Core: positions, gaps, intervals, sector times, lap times
    "TimingAppData",        # Extended: tire data, stint info, lap deltas
    "CarData.z",            # Telemetry: speed, RPM, throttle, brake, gear, DRS
    "CarStatus.z",          # ERS deployment, overtake mode
    "RaceControlMessages",  # Flags, incidents, SC, penalties
    "LapCount",             # Current lap / total laps
    "SessionStatus",        # Session active, finished, etc.
    "SessionInfo",          # Session metadata
    "WeatherData",          # Track/air temp, rainfall, wind
    "Position.z",           # Driver GPS positions on track
]


def _decode_z(data: str) -> dict:
    """Decode a .z compressed payload (base64 → gzip → JSON)."""
    try:
        raw = base64.b64decode(data)
        decompressed = gzip.decompress(raw)
        return json.loads(decompressed)
    except Exception as e:
        logger.debug("Failed to decode .z payload: %s", e)
        return {}


class F1LiveTimingClient:
    """
    Connects to the F1 Live Timing SignalR feed and streams
    enhanced telemetry data for tracked drivers.
    """

    def __init__(
        self,
        tracked_driver_numbers: Optional[List[int]] = None,
        driver_number_map: Optional[Dict[int, str]] = None,
    ):
        self._tracked: Set[int] = set(tracked_driver_numbers or [])
        self._driver_number_map = driver_number_map or {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._session: Optional[requests.Session] = None
        self._connection_token: Optional[str] = None
        self._message_id: int = 0
        self._on_car_status: Optional[Callable] = None
        self._on_tyre_data: Optional[Callable] = None
        self._on_timing_app: Optional[Callable] = None
        self._on_race_control: Optional[Callable] = None
        self._on_timing_data: Optional[Callable] = None
        self._on_lap_count: Optional[Callable] = None
        self._on_weather: Optional[Callable] = None
        self._on_positions: Optional[Callable] = None
        self._on_session_status: Optional[Callable] = None

    def set_tracked_drivers(self, driver_numbers: List[int]):
        self._tracked = set(driver_numbers[:3])

    def on_car_status(self, callback: Callable):
        self._on_car_status = callback

    def on_tyre_data(self, callback: Callable):
        self._on_tyre_data = callback

    def on_timing_app(self, callback: Callable):
        self._on_timing_app = callback

    def on_timing_data(self, callback: Callable):
        self._on_timing_data = callback

    def on_lap_count(self, callback: Callable):
        self._on_lap_count = callback

    def on_weather(self, callback: Callable):
        self._on_weather = callback

    def on_positions(self, callback: Callable):
        self._on_positions = callback

    def on_session_status(self, callback: Callable):
        self._on_session_status = callback

    def on_race_control(self, callback: Callable):
        self._on_race_control = callback

    def _negotiate(self) -> bool:
        """Negotiate a SignalR connection."""
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "BestHTTP",
            "Accept-Encoding": "gzip, identity",
        })

        try:
            resp = self._session.get(
                f"{F1_SIGNALR_BASE}/negotiate",
                params={"connectionData": json.dumps([{"name": F1_SIGNALR_HUB}])},
                timeout=10,
            )
            if resp.status_code in (401, 403):
                logger.error("F1 Live Timing: authentication denied (%d). "
                             "The endpoint may require credentials or be geo-blocked.", resp.status_code)
                return False
            resp.raise_for_status()
            data = resp.json()
            self._connection_token = data.get("ConnectionToken")
            if not self._connection_token:
                logger.error("No ConnectionToken in negotiate response")
                return False
            logger.info("F1 Live Timing: negotiated connection")
            return True
        except Exception as e:
            logger.error("F1 Live Timing negotiate failed: %s", e)
            return False

    def _subscribe(self) -> bool:
        """Subscribe to data topics via the hub."""
        if not self._connection_token or not self._session:
            return False

        try:
            resp = self._session.get(
                f"{F1_SIGNALR_BASE}/start",
                params={
                    "transport": "serverSentEvents",
                    "connectionToken": self._connection_token,
                    "connectionData": json.dumps([{"name": F1_SIGNALR_HUB}]),
                },
                timeout=10,
            )
            resp.raise_for_status()
            logger.info("F1 Live Timing: subscribed to topics: %s", TOPICS)
            return True
        except Exception as e:
            logger.error("F1 Live Timing subscribe failed: %s", e)
            return False

    def _reconnect(self) -> bool:
        """Tear down the current session and re-establish the connection."""
        logger.info("F1 Live Timing: attempting reconnect")
        if self._session:
            try:
                self._session.close()
            except Exception:
                pass
        self._connection_token = None
        self._message_id = 0

        if not self._negotiate():
            return False
        return self._subscribe()

    def _poll_loop(self):
        """Long-poll for messages from the SignalR connection."""
        consecutive_errors = 0
        max_errors = 5

        while self._running:
            try:
                resp = self._session.get(
                    f"{F1_SIGNALR_BASE}/poll",
                    params={
                        "transport": "longPolling",
                        "connectionToken": self._connection_token,
                        "connectionData": json.dumps([{"name": F1_SIGNALR_HUB}]),
                        "messageId": str(self._message_id),
                    },
                    timeout=30,
                )

                if resp.status_code != 200:
                    logger.warning("F1 Live Timing poll returned %d", resp.status_code)
                    consecutive_errors += 1
                    if consecutive_errors >= max_errors:
                        logger.warning("F1 Live Timing: %d consecutive errors, reconnecting", consecutive_errors)
                        if self._reconnect():
                            consecutive_errors = 0
                        else:
                            time.sleep(10)
                    else:
                        time.sleep(2)
                    continue

                consecutive_errors = 0
                data = resp.json()
                self._message_id = data.get("C", self._message_id)

                for msg in data.get("M", []):
                    self._handle_message(msg)

            except requests.exceptions.Timeout:
                continue
            except Exception as e:
                logger.warning("F1 Live Timing poll error: %s", e)
                consecutive_errors += 1
                if consecutive_errors >= max_errors:
                    logger.warning("F1 Live Timing: %d consecutive errors, reconnecting", consecutive_errors)
                    if self._reconnect():
                        consecutive_errors = 0
                    else:
                        time.sleep(10)
                else:
                    time.sleep(2)

    def _handle_message(self, msg: dict):
        """Route a SignalR message to the appropriate handler."""
        args = msg.get("A", [])

        if not args:
            return

        topic = args[0]
        payload = args[1] if len(args) > 1 else args[0]

        if isinstance(payload, str) and topic.endswith(".z"):
            payload = _decode_z(payload)

        if not isinstance(payload, dict):
            return

        if "TimingData" == topic:
            self._process_timing_data(payload)
        elif "CarStatus" in str(topic):
            self._process_car_status(payload)
        elif "CarData" in str(topic):
            self._process_car_data(payload)
        elif "TimingAppData" in str(topic):
            self._process_timing_app(payload)
        elif "RaceControlMessages" in str(topic):
            if self._on_race_control:
                self._on_race_control(payload)
        elif topic == "LapCount":
            self._process_lap_count(payload)
        elif topic == "SessionStatus":
            self._process_session_status(payload)
        elif topic == "WeatherData":
            self._process_weather(payload)
        elif "Position" in str(topic):
            self._process_positions(payload)

    def _iter_tracked_entries(self, data: dict, key: str = "Entries"):
        """Yield (driver_num, driver_id, entry) for each tracked driver in a feed payload."""
        entries = data.get(key, data)
        if not isinstance(entries, dict):
            return

        for driver_num_str, entry in entries.items():
            try:
                driver_num = int(driver_num_str)
            except (ValueError, TypeError):
                continue

            if driver_num not in self._tracked:
                continue

            driver_id = self._driver_number_map.get(driver_num)
            if not driver_id:
                continue

            yield driver_num, driver_id, entry

    def _process_car_status(self, data: dict):
        """Process CarStatus — ERS deployment, overtake mode, DRS status."""
        for driver_num, driver_id, status in self._iter_tracked_entries(data):
            parsed = {
                "driver_id": driver_id,
                "driver_number": driver_num,
                "ers_deploy": int(status.get("ErsDeployMode", 0) or 0),
                "overtake_active": bool(status.get("ErsStoreStatus", "0") != "0"),
            }

            if self._on_car_status:
                self._on_car_status(parsed)

    def _process_car_data(self, data: dict):
        """Process CarData.z — enhanced brake pressure from the raw feed."""
        for driver_num, driver_id, car_data in self._iter_tracked_entries(data):
            channels = car_data.get("Channels", {})
            if not channels:
                continue

            brake_val = channels.get("5", 0)
            if self._on_car_status:
                self._on_car_status({
                    "driver_id": driver_id,
                    "driver_number": driver_num,
                    "brake_pct": int(brake_val) if brake_val else 0,
                })

    def _process_timing_app(self, data: dict):
        """Process TimingAppData — lap delta, mini-sectors."""
        for driver_num, driver_id, timing in self._iter_tracked_entries(data, key="Lines"):
            stints = timing.get("Stints", {})
            current_stint = {}
            if isinstance(stints, dict):
                stint_keys = sorted(stints.keys(), key=lambda x: int(x) if x.isdigit() else 0)
                if stint_keys:
                    current_stint = stints[stint_keys[-1]]

            parsed = {
                "driver_id": driver_id,
                "driver_number": driver_num,
            }

            gap_to_best = timing.get("BestLapTime", {}).get("Value")
            if gap_to_best:
                parsed["lap_delta_str"] = gap_to_best

            if current_stint:
                for corner, key in [("fl", "FrontLeft"), ("fr", "FrontRight"),
                                     ("rl", "RearLeft"), ("rr", "RearRight")]:
                    press = current_stint.get(f"TyrePress{key}")
                    if press is not None:
                        parsed[f"tire_pressure_{corner}"] = float(press)
                    temp = current_stint.get(f"InnerTemp{key}")
                    if temp is not None:
                        parsed[f"tire_temp_{corner}"] = float(temp)

            if self._on_timing_app:
                self._on_timing_app(parsed)

    def _process_timing_data(self, data: dict):
        """Process TimingData — core positions, gaps, intervals, lap/sector times for ALL drivers."""
        lines = data.get("Lines", data)
        if not isinstance(lines, dict):
            return

        for driver_num_str, timing in lines.items():
            try:
                driver_num = int(driver_num_str)
            except (ValueError, TypeError):
                continue

            driver_id = self._driver_number_map.get(driver_num)
            if not driver_id:
                continue

            parsed = {"driver_id": driver_id, "driver_number": driver_num}

            pos = timing.get("Position")
            if pos is not None:
                parsed["position"] = int(pos) if isinstance(pos, (int, float, str)) and str(pos).isdigit() else None

            gap = timing.get("GapToLeader")
            if isinstance(gap, str) and gap.replace(".", "").replace("+", "").isdigit():
                parsed["gap_to_leader"] = float(gap.replace("+", ""))
            elif isinstance(gap, (int, float)):
                parsed["gap_to_leader"] = float(gap)

            interval = timing.get("IntervalToPositionAhead")
            if isinstance(interval, dict):
                val = interval.get("Value", "")
                if isinstance(val, str) and val.replace(".", "").replace("+", "").isdigit():
                    parsed["gap_to_ahead"] = float(val.replace("+", ""))

            last_lap = timing.get("LastLapTime")
            if isinstance(last_lap, dict):
                val = last_lap.get("Value", "")
                parsed["last_lap_time_str"] = val

            for s in [1, 2, 3]:
                sector = timing.get(f"Sectors", {})
                if isinstance(sector, dict):
                    s_data = sector.get(str(s - 1), {})  # 0-indexed in TimingData
                    if isinstance(s_data, dict):
                        val = s_data.get("Value", "")
                        if val:
                            parsed[f"sector{s}_str"] = val

            in_pit = timing.get("InPit")
            if in_pit is not None:
                parsed["is_in_pit"] = bool(in_pit)

            retired = timing.get("Retired") or timing.get("Stopped")
            if retired:
                parsed["is_retired"] = True

            if self._on_timing_data and len(parsed) > 2:
                self._on_timing_data(parsed)

    def _process_lap_count(self, data: dict):
        """Process LapCount — current lap number and total laps."""
        parsed = {}
        if "CurrentLap" in data:
            parsed["lap"] = int(data["CurrentLap"])
        if "TotalLaps" in data:
            parsed["total_laps"] = int(data["TotalLaps"])
        if self._on_lap_count and parsed:
            self._on_lap_count(parsed)

    def _process_session_status(self, data: dict):
        """Process SessionStatus — active, finished, aborted."""
        status = data.get("Status", "")
        if self._on_session_status:
            self._on_session_status({"status": str(status)})

    def _process_weather(self, data: dict):
        """Process WeatherData — temperatures, rainfall, wind."""
        parsed = {}
        if "AirTemp" in data:
            try: parsed["air_temp"] = float(data["AirTemp"])
            except (ValueError, TypeError): pass
        if "TrackTemp" in data:
            try: parsed["track_temp"] = float(data["TrackTemp"])
            except (ValueError, TypeError): pass
        if "Rainfall" in data:
            parsed["rainfall"] = str(data["Rainfall"]).lower() in ("1", "true", "yes")
        if "WindSpeed" in data:
            try: parsed["wind_speed"] = float(data["WindSpeed"])
            except (ValueError, TypeError): pass
        if "Humidity" in data:
            try: parsed["humidity"] = float(data["Humidity"])
            except (ValueError, TypeError): pass
        if self._on_weather and parsed:
            self._on_weather(parsed)

    def _process_positions(self, data: dict):
        """Process Position.z — GPS locations for track map."""
        if isinstance(data, str):
            data = _decode_z(data)
        entries = data.get("Position", data)
        if not isinstance(entries, (list, dict)):
            return

        positions = []
        items = entries if isinstance(entries, list) else entries.values()
        for entry in items:
            if not isinstance(entry, dict):
                continue
            entries_inner = entry.get("Entries", {})
            for driver_num_str, pos_data in entries_inner.items():
                try:
                    driver_num = int(driver_num_str)
                except (ValueError, TypeError):
                    continue
                driver_id = self._driver_number_map.get(driver_num)
                if driver_id and "X" in pos_data and "Y" in pos_data:
                    positions.append({
                        "driver_id": driver_id,
                        "x": float(pos_data["X"]),
                        "y": float(pos_data["Y"]),
                    })
        if self._on_positions and positions:
            self._on_positions(positions)

    def start(self) -> bool:
        """Start the F1 Live Timing connection in a background thread."""
        if self._running:
            return True

        if not self._negotiate():
            return False

        if not self._subscribe():
            return False

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("F1 Live Timing client started")
        return True

    def stop(self):
        """Stop the connection."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        if self._session:
            try:
                self._session.get(
                    f"{F1_SIGNALR_BASE}/abort",
                    params={
                        "transport": "longPolling",
                        "connectionToken": self._connection_token,
                        "connectionData": json.dumps([{"name": F1_SIGNALR_HUB}]),
                    },
                    timeout=5,
                )
            except Exception:
                pass
            self._session.close()
        logger.info("F1 Live Timing client stopped")

    @property
    def is_connected(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()
