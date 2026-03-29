"""
Live Prediction Flask Blueprint — SSE endpoint for real-time race updates.

Provides:
- GET /live/stream         — SSE stream of live predictions + driver locations
- GET /live/state          — Current race state as JSON
- GET /live/track          — Track outline as JSON (cached per session)
- GET /live/positions      — Current driver positions on track as JSON
- GET /live/probabilities  — Current live probabilities + full history for charts
- POST /live/start         — Start live polling
- POST /live/stop          — Stop live polling
- GET /live/replay/sessions — List available race sessions for replay
- GET /live/replay/data/<session_key> — Fetch replay data for a past race
- GET /live/replay/status/<session_key> — Check replay data build progress
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, Response, jsonify, render_template, request

from data.ingest.live_feed import (
    DRIVER_CODES,
    DRIVER_CONSTRUCTOR,
    DRIVER_NUMBER_MAP,
)

logger = logging.getLogger(__name__)

live_bp = Blueprint("live", __name__, url_prefix="/live")

# Module-level state (single-server, single-session)
_feed: Optional[object] = None
_predictor: Optional[object] = None
_in_race_predictor: Optional[object] = None
_pre_race_predictions: Optional[object] = None  # cached for InRacePredictor init


def _get_feed():
    """Lazy-initialize the live feed."""
    global _feed
    if _feed is None:
        from data.ingest.live_feed import LiveFeed
        _feed = LiveFeed(poll_interval=5.0)
    return _feed


def _get_predictor(pre_race_predictions=None):
    """Lazy-initialize the live predictor."""
    global _predictor
    if _predictor is None and pre_race_predictions is not None:
        from data.models.live import LiveRacePredictor
        _predictor = LiveRacePredictor(pre_race_predictions)
    return _predictor


def _get_in_race_predictor(pre_race_predictions=None, total_laps: int = 57,
                            circuit_type: str = "mixed"):
    """Lazy-initialize the InRacePredictor (Monte Carlo based)."""
    global _in_race_predictor
    if _in_race_predictor is None and pre_race_predictions is not None:
        from data.models.live import InRacePredictor
        _in_race_predictor = InRacePredictor(
            pre_race_predictions=pre_race_predictions,
            total_laps=total_laps,
            circuit_type=circuit_type,
            n_sims=1000,
        )
    return _in_race_predictor


def _driver_meta(did: str) -> Dict[str, str]:
    """Return shared code/team metadata for a driver_id."""
    return {
        "code": DRIVER_CODES.get(did, did[:3].upper()),
        "team": DRIVER_CONSTRUCTOR.get(did, "unknown"),
    }


def _build_locations(state):
    """Build locations list from race state for JSON serialization."""
    locations = []
    for did, ds in state.drivers.items():
        loc = state.driver_locations.get(did)
        entry = _driver_meta(did)
        entry.update({
            "id": did,
            "x": loc[0] if loc else None,
            "y": loc[1] if loc else None,
            "position": ds.position,
            "gap_to_ahead": ds.gap_to_ahead,
            "is_in_pit": ds.is_in_pit,
            "is_retired": ds.is_retired,
            "sector1": ds.sector1,
            "sector2": ds.sector2,
            "sector3": ds.sector3,
        })
        locations.append(entry)
    return locations


def _build_telemetry(state):
    """Build telemetry dict for tracked drivers only."""
    feed = _get_feed()
    tracked_numbers = feed.get_tracked_drivers()
    tracked_ids = {DRIVER_NUMBER_MAP[n] for n in tracked_numbers if n in DRIVER_NUMBER_MAP}

    telemetry = {}
    for did, ds in state.drivers.items():
        if did not in tracked_ids:
            continue
        entry = _driver_meta(did)
        entry.update({
            "speed": ds.speed,
            "rpm": ds.rpm,
            "gear": ds.gear,
            "throttle": ds.throttle,
            "brake": ds.brake,
            "drs": ds.drs,
            "tire_compound": ds.tire_compound,
            "tire_age": ds.tire_age,
            "position": ds.position,
            "gap_to_leader": ds.gap_to_leader,
            "last_lap_time": ds.last_lap_time,
            "best_lap_time": ds.best_lap_time,
            "pits_completed": ds.pits_completed,
            "ers_deploy": ds.ers_deploy,
            "overtake_active": ds.overtake_active,
            "battery_soc": ds.battery_soc,
            "lap_delta": ds.lap_delta,
            "tire_pressures": {
                "fl": ds.tire_pressure_fl, "fr": ds.tire_pressure_fr,
                "rl": ds.tire_pressure_rl, "rr": ds.tire_pressure_rr,
            },
            "tire_temps": {
                "fl": ds.tire_temp_fl, "fr": ds.tire_temp_fr,
                "rl": ds.tire_temp_rl, "rr": ds.tire_temp_rr,
            },
        })
        telemetry[did] = entry
    return telemetry


@live_bp.route("/start", methods=["POST"])
def start_live():
    """Start live data polling."""
    global _pre_race_predictions
    feed = _get_feed()

    # Load pre-race predictions
    try:
        import pandas as pd
        from data.models.predictor import F1Predictor

        cache_dir = Path(__file__).parent.parent / "data" / "cache" / "processed"
        model = F1Predictor()
        model.load()

        fm = pd.read_parquet(cache_dir / "feature_matrix.parquet")
        latest_season = int(fm["season"].max())
        latest_round = int(fm[fm["season"] == latest_season]["round"].max())
        race_data = fm[(fm["season"] == latest_season) & (fm["round"] == latest_round)]
        predictions = model.predict_race(race_data)
        _pre_race_predictions = predictions

        # Detect circuit type from feature matrix
        circuit_type = "mixed"
        if "circuit_type" in race_data.columns:
            ct = race_data["circuit_type"].iloc[0]
            if ct in ("street", "high_speed", "technical", "mixed"):
                circuit_type = ct

        # Detect total laps (from request body or default)
        total_laps = 57
        if request.is_json and request.json:
            total_laps = request.json.get("total_laps", 57)

        _get_predictor(predictions)
        _get_in_race_predictor(predictions, total_laps=total_laps, circuit_type=circuit_type)
    except Exception as e:
        logger.error("Failed to load pre-race predictions: %s", e)
        return jsonify({"error": str(e)}), 500

    if request.is_json and request.json and request.json.get("session_key"):
        feed.session_key = int(request.json["session_key"])
        logger.info("Using provided session key: %d", feed.session_key)

    feed.start_polling()
    return jsonify({"status": "polling_started", "session_key": feed.session_key})


@live_bp.route("/stop", methods=["POST"])
def stop_live():
    """Stop live data polling."""
    global _in_race_predictor, _pre_race_predictions
    feed = _get_feed()
    feed.stop_polling()
    _in_race_predictor = None
    _pre_race_predictions = None
    return jsonify({"status": "polling_stopped"})


@live_bp.route("/state")
def get_state():
    """Get current race state as JSON."""
    feed = _get_feed()
    state = feed.get_current_state()

    if state is None:
        return jsonify({"error": "No live data available", "hint": "POST /live/start first"})

    predictor = _get_predictor()
    if predictor is None:
        return jsonify({"error": "Predictor not initialized"})

    predictions = predictor.update(state)

    return jsonify({
        "lap": state.lap,
        "total_laps": state.total_laps,
        "track_status": state.track_status,
        "predictions": predictions.to_dict("records"),
    })


@live_bp.route("/track")
def get_track():
    """Return the track outline as JSON (cached per session)."""
    feed = _get_feed()
    outline = feed.get_track_outline()
    if not outline:
        return jsonify({"outline": [], "error": "No track data available"})
    return jsonify({"outline": outline})


@live_bp.route("/positions")
def get_positions():
    """Return current driver positions on track + metadata."""
    feed = _get_feed()
    state = feed.get_current_state()
    if state is None:
        return jsonify({"drivers": [], "error": "No live data"})

    return jsonify({
        "drivers": _build_locations(state),
        "lap": state.lap,
        "total_laps": state.total_laps,
        "track_status": state.track_status,
    })


@live_bp.route("/stream")
def stream():
    """
    SSE endpoint for live prediction updates + driver locations.

    Sends data every ~5 seconds. Predictions recompute on lap changes;
    driver locations are included in every update for smooth track map rendering.

    Connect via EventSource:
        const es = new EventSource('/live/stream');
        es.onmessage = (e) => console.log(JSON.parse(e.data));
    """
    def generate():
        feed = _get_feed()
        predictor = _get_predictor()
        in_race = _get_in_race_predictor()
        last_lap = -1
        last_predictions = None
        last_live_probs = None
        last_payload_hash: Optional[str] = None
        prev_win_probs: Dict[str, float] = {}

        while True:
            state = feed.get_current_state()

            if state and (predictor or in_race):
                # Recompute predictions only on lap change (expensive)
                if state.lap != last_lap:
                    last_lap = state.lap
                    if predictor:
                        last_predictions = predictor.update(state)

                    # InRacePredictor: full Monte Carlo update on lap change
                    if in_race and state.lap > 0:
                        # Update total_laps from live feed if available
                        if state.total_laps > 0:
                            in_race.total_laps = state.total_laps
                        last_live_probs = in_race.update(state)

                        # Detect big probability shifts for event callouts
                        events = _detect_probability_events(
                            last_live_probs, prev_win_probs, state,
                        )
                        prev_win_probs = {
                            row["driver_id"]: row["live_win_prob"]
                            for _, row in last_live_probs.iterrows()
                        }
                    else:
                        events = []

                # Always send locations for track map updates
                locations = _build_locations(state)

                data: Dict[str, Any] = {
                    "lap": state.lap,
                    "total_laps": state.total_laps,
                    "track_status": state.track_status,
                    "laps_remaining": state.laps_remaining,
                    "air_temp": state.air_temp,
                    "track_temp": state.track_temp,
                    "rainfall": state.rainfall,
                    "predictions": last_predictions.to_dict("records") if last_predictions is not None else [],
                    "locations": locations,
                }

                # Include telemetry for tracked drivers
                data["telemetry"] = _build_telemetry(state)

                # Include live probabilities when available
                if last_live_probs is not None:
                    data["live_probabilities"] = last_live_probs.to_dict("records")
                    if in_race:
                        data["probability_history"] = in_race.get_probability_history()
                    if events:
                        data["probability_events"] = events

                # Skip sending if nothing changed since last tick
                payload = json.dumps(data)
                payload_hash = hashlib.md5(payload.encode()).hexdigest()
                if payload_hash != last_payload_hash:
                    last_payload_hash = payload_hash
                    yield f"data: {payload}\n\n"
                else:
                    # Send a lightweight heartbeat so the connection stays alive
                    yield f"data: {json.dumps({'status': 'no_change'})}\n\n"
            else:
                yield f"data: {json.dumps({'status': 'waiting'})}\n\n"

            time.sleep(5)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _detect_probability_events(
    current_df, prev_win_probs: Dict[str, float], state,
) -> List[Dict[str, Any]]:
    """Detect big probability shifts (>10%) for event callouts."""
    events: List[Dict[str, Any]] = []
    if current_df is None or not prev_win_probs:
        return events

    for _, row in current_df.iterrows():
        did = row["driver_id"]
        cur = row.get("live_win_prob", 0.0)
        prev = prev_win_probs.get(did, cur)
        shift = cur - prev

        if abs(shift) > 0.10:
            code = DRIVER_CODES.get(did, did[:3].upper())
            direction = "+" if shift > 0 else ""
            reason = ""
            if state.track_status == "sc":
                reason = " (Safety Car)"
            elif state.track_status == "vsc":
                reason = " (VSC)"
            elif row.get("tire_age", 0) <= 2:
                reason = " (fresh tyres)"

            events.append({
                "lap": state.lap,
                "driver_id": did,
                "driver_code": code,
                "shift": round(shift * 100, 1),
                "message": f"LAP {state.lap}: {code} {direction}{shift * 100:.1f}%{reason}",
            })

    return events


@live_bp.route("/probabilities")
def live_probabilities():
    """Return current live probabilities + full history for charts."""
    feed = _get_feed()
    state = feed.get_current_state()
    in_race = _get_in_race_predictor()

    if state is None:
        return jsonify({"error": "No live data available", "hint": "POST /live/start first"})

    if in_race is None:
        return jsonify({"error": "InRacePredictor not initialized"})

    if state.lap > 0:
        current = in_race.update(state)
        current_records = current.to_dict("records")
    else:
        current_records = []

    return jsonify({
        "current": current_records,
        "history": in_race.get_probability_history(),
        "lap": state.lap,
        "total_laps": state.total_laps,
        "track_status": state.track_status,
    })


# ---------------------------------------------------------------------------
# What-If Scenario Simulator
# ---------------------------------------------------------------------------


@live_bp.route("/whatif", methods=["POST"])
def whatif():
    """
    Run a what-if scenario simulation.

    POST body: {"scenario": {"type": "safety_car"}}
    Combined:  {"scenario": {"scenarios": [{"type": "safety_car"}, ...]}}
    """
    body = request.get_json(silent=True) or {}
    scenario = body.get("scenario", {})

    if not scenario:
        return jsonify({"error": "No scenario provided"}), 400

    feed = _get_feed()
    state = feed.get_current_state() if feed else None
    in_race = _get_in_race_predictor()

    # Fallback: build synthetic state from pre-race predictions if no live session
    if state is None or state.lap == 0:
        from data.models.live import RaceState, DriverState

        if _pre_race_predictions is None and in_race is None:
            return jsonify({"error": "No live session and no pre-race predictions available"}), 404

        state = RaceState()
        state.lap = body.get("lap", 1)
        state.total_laps = body.get("total_laps", 57)
        state.track_status = "clear"

        preds = _pre_race_predictions
        if preds is not None:
            for _, row in preds.iterrows():
                did = row.get("driver_id", "")
                if not did:
                    continue
                ds = DriverState(did)
                pos = int(row.get("predicted_position", 10))
                ds.position = pos
                ds.gap_to_leader = (pos - 1) * 1.5
                ds.tire_compound = "medium"
                ds.tire_age = state.lap
                ds.pits_completed = 0
                ds.last_lap_time = 90.0 + pos * 0.2
                state.drivers[did] = ds

    # Initialize predictor if needed
    if in_race is None and _pre_race_predictions is not None:
        in_race = _get_in_race_predictor(
            _pre_race_predictions, total_laps=state.total_laps
        )

    if in_race is None:
        return jsonify({"error": "Cannot initialize predictor"}), 500

    result = in_race.simulate_scenario(state, scenario)
    return jsonify(result)


# ---------------------------------------------------------------------------
# Race Replay — fetch, downsample, cache, and serve past race location data
# ---------------------------------------------------------------------------

REPLAY_CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "replays"
QUALITY_CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "quality_checks"

# Track in-progress builds: session_key -> {progress, total, status, error?}
_replay_builds: Dict[int, Dict[str, Any]] = {}
_replay_lock = threading.Lock()


def _build_replay_data(session_key: int) -> None:
    """Background worker: fetch all driver locations, downsample, cache to disk."""
    from data.ingest.openf1_client import OpenF1Client

    import pandas as pd

    client = OpenF1Client()

    with _replay_lock:
        _replay_builds[session_key] = {
            "status": "fetching",
            "progress": 0,
            "total": len(DRIVER_NUMBER_MAP),
            "message": "Discovering session...",
        }

    try:
        # Get session metadata
        sessions = client.get_sessions(session_type="Race")
        sess_row = None
        if not sessions.empty:
            match = sessions[sessions["session_key"] == session_key]
            if not match.empty:
                sess_row = match.iloc[0]

        session_name = ""
        if sess_row is not None:
            session_name = str(sess_row.get("session_name", ""))
            if not session_name:
                session_name = str(sess_row.get("circuit_short_name", f"Session {session_key}"))

        # Fetch total laps from session info
        total_laps = 0
        if sess_row is not None:
            tl = sess_row.get("total_laps")
            if tl and pd.notna(tl):
                total_laps = int(tl)

        # Determine race start time from lap data (first lap of any driver)
        # This filters out practice/formation data
        race_start_ts: Optional[float] = None
        for dn in [1, 16, 44, 63, 81]:
            try:
                laps = client.get_laps(session_key=session_key, driver_number=dn)
                if not laps.empty and "date_start" in laps.columns:
                    first_lap_ts = pd.to_datetime(
                        laps.iloc[0]["date_start"], format="mixed", utc=True
                    )
                    race_start_ts = first_lap_ts.timestamp() * 1000
                    # Subtract ~30s for formation lap
                    race_start_ts -= 30_000
                    logger.info("Race start from laps: %s", first_lap_ts)
                    break
            except Exception:
                continue

        # Fetch all driver locations — one per driver (bulk returns 422)
        # Use a thread pool to parallelize API calls (3 concurrent to avoid rate limits)
        from concurrent.futures import ThreadPoolExecutor, as_completed

        all_driver_data: Dict[str, Dict[str, Any]] = {}
        raw_location_data: Dict[int, pd.DataFrame] = {}  # driver_number -> raw DataFrame
        driver_numbers = list(DRIVER_NUMBER_MAP.keys())
        completed_count = 0

        def _fetch_driver_location(dn: int) -> Tuple[int, Optional[pd.DataFrame]]:
            try:
                loc = client.get_location(session_key=session_key, driver_number=dn)
                if loc.empty or "x" not in loc.columns or "date" not in loc.columns:
                    return dn, None
                return dn, loc
            except Exception as e:
                logger.warning("Replay: failed to fetch driver %d: %s", dn, e)
                return dn, None

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(_fetch_driver_location, dn): dn for dn in driver_numbers}
            for future in as_completed(futures):
                dn, loc = future.result()
                completed_count += 1
                driver_id = DRIVER_NUMBER_MAP[dn]
                with _replay_lock:
                    _replay_builds[session_key]["progress"] = completed_count
                    _replay_builds[session_key]["message"] = f"Fetching {DRIVER_CODES.get(driver_id, '???')} ({completed_count}/{len(driver_numbers)})"

                if loc is None:
                    continue

                raw_location_data[dn] = loc

                # Filter stationary points and parse timestamps
                moving = loc[(loc["x"] != 0) | (loc["y"] != 0)].copy()
                if moving.empty:
                    continue

                moving["ts"] = pd.to_datetime(moving["date"], utc=True, format="mixed")
                moving = moving.sort_values("ts")

                # Filter to race window only (skip practice/formation data)
                if race_start_ts is not None:
                    race_start_dt = pd.Timestamp(race_start_ts / 1000, unit="s", tz="UTC")
                    race_end_dt = race_start_dt + pd.Timedelta(hours=3)
                    moving = moving[(moving["ts"] >= race_start_dt) & (moving["ts"] <= race_end_dt)]
                    if moving.empty:
                        continue

                # Set race_start_ts from first driver if we couldn't get it from laps
                if race_start_ts is None:
                    race_start_ts = moving["ts"].iloc[0].timestamp() * 1000

                # Filter constant-coordinate data (OpenF1 sometimes returns unprocessed data)
                unique_coords = moving[["x", "y"]].drop_duplicates()
                if len(unique_coords) < 10:
                    logger.warning("Replay: driver %s has only %d unique coords, skipping", driver_id, len(unique_coords))
                    continue

                # Downsample from ~3.7Hz to ~1Hz (keep every 4th point)
                downsampled = moving.iloc[::4].copy()

                # Build points array: [offset_ms, x, y]
                points: List[List[float]] = []
                for _, row in downsampled.iterrows():
                    ts_ms = row["ts"].timestamp() * 1000
                    points.append([ts_ms, float(row["x"]), float(row["y"])])

                all_driver_data[driver_id] = {
                    "code": DRIVER_CODES.get(driver_id, driver_id[:3].upper()),
                    "team": DRIVER_CONSTRUCTOR.get(driver_id, "unknown"),
                    "points": points,
                }

        if not all_driver_data or race_start_ts is None:
            with _replay_lock:
                _replay_builds[session_key] = {
                    "status": "error",
                    "progress": 0,
                    "total": 0,
                    "message": "No location data available for this session",
                }
            return

        # Normalize all timestamps to offsets from race start
        for driver_id, dd in all_driver_data.items():
            dd["points"] = [
                [p[0] - race_start_ts, p[1], p[2]] for p in dd["points"]
            ]

        # Calculate race duration
        max_offset = 0
        for dd in all_driver_data.values():
            if dd["points"]:
                last_offset = dd["points"][-1][0]
                if last_offset > max_offset:
                    max_offset = last_offset
        duration_seconds = int(max_offset / 1000)

        # Extract track outline from the driver with the most data (cleanest lap)
        with _replay_lock:
            _replay_builds[session_key]["message"] = "Extracting track outline..."

        track_outline = _extract_replay_track_outline(client, session_key, all_driver_data, raw_location_data)

        # Fetch lap timing data
        with _replay_lock:
            _replay_builds[session_key]["message"] = "Fetching lap data..."

        laps_data = _fetch_lap_markers(client, session_key, race_start_ts)

        # If total_laps not from session info, infer from laps data
        if total_laps == 0 and laps_data:
            total_laps = max(m["lap"] for m in laps_data)

        # Fetch race control messages for track status
        race_control_events = _fetch_race_control_events(client, session_key, race_start_ts)

        # Build final payload
        replay_data = {
            "session_key": session_key,
            "session_name": session_name,
            "total_laps": total_laps,
            "track_outline": track_outline,
            "duration_seconds": duration_seconds,
            "drivers": all_driver_data,
            "laps": laps_data,
            "race_control": race_control_events,
        }

        # Cache to disk
        REPLAY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = REPLAY_CACHE_DIR / f"{session_key}.json"
        with open(cache_path, "w") as f:
            json.dump(replay_data, f)

        logger.info(
            "Replay data cached: session=%d, drivers=%d, duration=%ds, file=%.1fMB",
            session_key, len(all_driver_data), duration_seconds,
            cache_path.stat().st_size / 1_000_000,
        )

        with _replay_lock:
            _replay_builds[session_key] = {
                "status": "done",
                "progress": len(driver_numbers),
                "total": len(driver_numbers),
                "message": "Ready",
            }

    except Exception as e:
        logger.error("Replay build failed for session %d: %s", session_key, e)
        with _replay_lock:
            _replay_builds[session_key] = {
                "status": "error",
                "progress": 0,
                "total": 0,
                "message": str(e),
            }


def _extract_replay_track_outline(
    client, session_key: int, all_driver_data: Dict[str, Dict[str, Any]],
    raw_location_data: Optional[Dict[int, Any]] = None,
) -> List[List[float]]:
    """Extract track outline from the driver with the most location points.

    Uses the same clean-lap approach as LiveFeed.get_track_outline():
    pick a lap from the middle third, take the fastest, extract ~200 points.

    Accepts raw_location_data to reuse already-fetched DataFrames from _build_replay_data,
    avoiding redundant API calls.
    """
    import pandas as pd

    if raw_location_data is None:
        raw_location_data = {}

    # Find driver with most points
    best_driver_num = None
    best_count = 0

    num_to_id = DRIVER_NUMBER_MAP
    id_to_num = {v: k for k, v in num_to_id.items()}

    for driver_id, dd in all_driver_data.items():
        if len(dd["points"]) > best_count:
            best_count = len(dd["points"])
            best_driver_num = id_to_num.get(driver_id)

    if best_driver_num is None:
        return []

    def _fallback_from_points():
        """Extract outline from the already-fetched driver points.
        Uses mid-race data to avoid grid/pit positions."""
        driver_id = num_to_id.get(best_driver_num, "")
        pts = all_driver_data.get(driver_id, {}).get("points", [])
        if len(pts) < 400:
            return []
        # Skip first 20% (formation lap/grid) and use a segment from mid-race
        start = len(pts) // 3
        segment = pts[start:start + 500]
        # Filter out stationary points (x,y both the same as the first point)
        if segment:
            moving = [p for p in segment if not (p[1] == segment[0][1] and p[2] == segment[0][2])]
            if len(moving) < 100:
                moving = segment
        else:
            moving = segment
        step = max(1, len(moving) // 200)
        return [[p[1], p[2]] for p in moving[::step]]

    try:
        try:
            laps = client.get_laps(session_key=session_key, driver_number=best_driver_num)
        except Exception:
            laps = pd.DataFrame()

        # Reuse raw location data from _build_replay_data if available
        loc = raw_location_data.get(best_driver_num, pd.DataFrame())
        if loc.empty:
            try:
                loc = client.get_location(session_key=session_key, driver_number=best_driver_num)
            except Exception:
                loc = pd.DataFrame()

        if loc.empty:
            return _fallback_from_points()
        if laps.empty:
            # Have location data but no laps — use mid-session window
            moving = loc[(loc["x"] != 0) | (loc["y"] != 0)]
            if len(moving) >= 400:
                n = len(moving)
                segment = moving.iloc[n // 2: n // 2 + 500]
                step = max(1, len(segment) // 200)
                sampled = segment.iloc[::step]
                return list(zip(sampled["x"].astype(float).tolist(), sampled["y"].astype(float).tolist()))
            return _fallback_from_points()

        moving = loc[(loc["x"] != 0) | (loc["y"] != 0)].copy()
        if moving.empty or "date" not in moving.columns:
            return []

        valid_laps = laps.dropna(subset=["date_start"]) if "date_start" in laps.columns else laps
        if len(valid_laps) >= 6:
            n = len(valid_laps)
            mid_laps = valid_laps.iloc[n // 3: 2 * n // 3]
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
            lap_end = lap_start + pd.Timedelta(seconds=120)
            mask = (moving_dt >= lap_start) & (moving_dt <= lap_end)
            lap_data = moving.loc[mask]

            if len(lap_data) >= 100:
                step = max(1, len(lap_data) // 200)
                sampled = lap_data.iloc[::step]
                return list(zip(
                    sampled["x"].astype(float).tolist(),
                    sampled["y"].astype(float).tolist(),
                ))

        # Fallback
        n = len(moving)
        mid = n // 2
        segment = moving.iloc[mid:mid + 500]
        step = max(1, len(segment) // 200)
        sampled = segment.iloc[::step]
        return list(zip(
            sampled["x"].astype(float).tolist(),
            sampled["y"].astype(float).tolist(),
        ))

    except Exception as e:
        logger.warning("Track outline extraction failed: %s", e)
        return []


def _fetch_lap_markers(
    client, session_key: int, race_start_ms: float,
) -> List[Dict[str, Any]]:
    """Fetch lap boundary timestamps, return as offset-based markers."""
    import pandas as pd

    try:
        # Fetch leader's laps (driver_number=1 is common leader, but get all and use lap_number grouping)
        laps = client.get_laps(session_key=session_key)
        if laps.empty or "lap_number" not in laps.columns or "date_start" not in laps.columns:
            return []

        # Group by lap_number and take the earliest date_start (leader's timestamp)
        laps["date_start_dt"] = pd.to_datetime(laps["date_start"], utc=True, format="mixed")
        lap_starts = laps.groupby("lap_number")["date_start_dt"].min().sort_index()

        markers = []
        for lap_num, ts in lap_starts.items():
            offset_ms = ts.timestamp() * 1000 - race_start_ms
            if offset_ms >= 0:
                markers.append({
                    "lap": int(lap_num),
                    "timestamp_offset_ms": round(offset_ms),
                })

        return markers

    except Exception as e:
        logger.warning("Lap markers fetch failed: %s", e)
        return []


def _fetch_race_control_events(
    client, session_key: int, race_start_ms: float,
) -> List[Dict[str, Any]]:
    """Fetch race control messages (flags, SC, VSC, red flag) as timed events."""
    import pandas as pd

    try:
        rc = client.get_race_control(session_key=session_key)
        if rc.empty or "date" not in rc.columns:
            return []

        rc["ts"] = pd.to_datetime(rc["date"], utc=True, format="mixed")
        events = []
        for _, row in rc.iterrows():
            flag = str(row.get("flag", "")).lower()
            message = str(row.get("message", ""))

            event_type = None
            if "safety" in flag and "virtual" not in flag:
                event_type = "sc"
            elif "virtual" in flag:
                event_type = "vsc"
            elif "red" in flag:
                event_type = "red"
            elif "green" in flag or "clear" in flag:
                event_type = "clear"

            if event_type:
                offset_ms = row["ts"].timestamp() * 1000 - race_start_ms
                if offset_ms >= 0:
                    events.append({
                        "type": event_type,
                        "timestamp_offset_ms": round(offset_ms),
                        "message": message,
                    })

        return events

    except Exception as e:
        logger.warning("Race control fetch failed: %s", e)
        return []


@live_bp.route("/replay/sessions")
def replay_sessions():
    """List completed race sessions from the current season for replay."""
    from data.ingest.openf1_client import OpenF1Client
    import pandas as pd

    client = OpenF1Client()
    try:
        current_year = pd.Timestamp.now().year
        sessions = client.get_sessions(year=current_year, session_type="Race")
        if sessions.empty:
            return jsonify({"sessions": []})

        today = pd.Timestamp.now(tz="UTC").normalize()  # start of today (midnight UTC)

        # Filter to completed main races only (no sprints, no future races)
        if "date_start" in sessions.columns:
            sessions["date_start"] = pd.to_datetime(sessions["date_start"], utc=True, format="mixed")
            sessions = sessions[sessions["date_start"] < today]
        if "session_name" in sessions.columns:
            sessions = sessions[sessions["session_name"].str.strip().str.lower() == "race"]
        sessions = sessions.sort_values("date_start", ascending=False)

        result = []
        for _, row in sessions.iterrows():
            sk = row.get("session_key")
            if sk is None:
                continue

            cache_path = REPLAY_CACHE_DIR / f"{int(sk)}.json"
            cached = cache_path.exists()

            # Quick location data quality check (skip sessions with unprocessed data)
            # Only check uncached sessions — cached ones already passed quality check
            # Quality results are persisted to disk to avoid repeated API calls
            if not cached:
                QUALITY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                quality_path = QUALITY_CACHE_DIR / f"{int(sk)}.json"
                quality_ok = None

                if quality_path.exists():
                    try:
                        with open(quality_path, "r") as qf:
                            quality_ok = json.load(qf).get("ok")
                    except (json.JSONDecodeError, OSError):
                        quality_path.unlink(missing_ok=True)

                if quality_ok is None:
                    try:
                        sample_loc = client.get_location(session_key=int(sk), driver_number=1)
                        if not sample_loc.empty and "x" in sample_loc.columns:
                            unique = len(sample_loc[["x", "y"]].drop_duplicates())
                            quality_ok = unique >= 10
                            if not quality_ok:
                                logger.info("Skipping session %d — location data not yet processed (%d unique coords)", int(sk), unique)
                        else:
                            quality_ok = True  # No data to check — let user try
                    except Exception:
                        quality_ok = True  # If check fails, still list it — user can try
                    # Persist the result
                    with open(quality_path, "w") as qf:
                        json.dump({"ok": quality_ok, "session_key": int(sk)}, qf)

                if not quality_ok:
                    continue

            session_name = str(row.get("session_name", ""))
            circuit = str(row.get("circuit_short_name", ""))
            # Make sprint vs race clear in the label
            label = f"{circuit} — {session_name}" if session_name else circuit

            result.append({
                "session_key": int(sk),
                "session_name": session_name,
                "label": label,
                "circuit_short_name": circuit,
                "country_name": str(row.get("country_name", "")),
                "date_start": str(row.get("date_start", "")),
                "year": current_year,
                "cached": cached,
            })

        return jsonify({"sessions": result})

    except Exception as e:
        logger.error("Failed to list replay sessions: %s", e)
        return jsonify({"sessions": [], "error": str(e)}), 500


@live_bp.route("/replay/data/<int:session_key>")
def replay_data(session_key: int):
    """
    Fetch replay data for a past race session.

    Returns cached data immediately if available.
    Otherwise, kicks off a background build and returns 202 with progress info.
    """
    # Check disk cache first
    cache_path = REPLAY_CACHE_DIR / f"{session_key}.json"
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
            return jsonify(data)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Corrupt cache for session %d, rebuilding: %s", session_key, e)
            cache_path.unlink(missing_ok=True)

    # Check if a build is already in progress
    with _replay_lock:
        build = _replay_builds.get(session_key)

    if build and build["status"] == "fetching":
        return jsonify({
            "status": "building",
            "progress": build["progress"],
            "total": build["total"],
            "message": build["message"],
        }), 202

    if build and build["status"] == "done":
        # Build just finished, serve from cache
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                return jsonify(data)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Corrupt cache for session %d after build, rebuilding: %s", session_key, e)
                cache_path.unlink(missing_ok=True)
                # Fall through to start a new build

    if build and build["status"] == "error":
        # Clear the error so the user can retry
        with _replay_lock:
            _replay_builds.pop(session_key, None)
        return jsonify({
            "status": "error",
            "message": build["message"],
        }), 500

    # Start background build (under lock to prevent duplicate threads)
    with _replay_lock:
        current = _replay_builds.get(session_key)
        if current and current["status"] == "fetching":
            return jsonify({
                "status": "building",
                "progress": current["progress"],
                "total": current["total"],
                "message": current["message"],
            }), 202
        _replay_builds[session_key] = {
            "status": "fetching",
            "progress": 0,
            "total": 20,
            "message": "Starting data fetch...",
        }

    thread = threading.Thread(
        target=_build_replay_data,
        args=(session_key,),
        daemon=True,
    )
    thread.start()

    return jsonify({
        "status": "building",
        "progress": 0,
        "total": 20,
        "message": "Starting data fetch...",
    }), 202


@live_bp.route("/replay/status/<int:session_key>")
def replay_status(session_key: int):
    """Check the build status for a replay dataset."""
    cache_path = REPLAY_CACHE_DIR / f"{session_key}.json"
    if cache_path.exists():
        return jsonify({"status": "done", "message": "Ready"})

    with _replay_lock:
        build = _replay_builds.get(session_key)

    if not build:
        return jsonify({"status": "not_started", "message": "No build in progress"})

    return jsonify(build)


@live_bp.route("/tracked", methods=["GET", "POST"])
def tracked_drivers():
    """Get or set which drivers have telemetry tracking enabled (max 3)."""
    feed = _get_feed()

    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        drivers = data.get("drivers", [])
        if not isinstance(drivers, list) or not all(isinstance(d, int) for d in drivers):
            return jsonify({"error": "drivers must be a list of driver numbers"}), 400
        feed.set_tracked_drivers(drivers[:3])
        return jsonify({"tracked": feed.get_tracked_drivers()})

    return jsonify({"tracked": feed.get_tracked_drivers()})


@live_bp.route("/f1live", methods=["GET", "POST"])
def f1_live_toggle():
    """Enable or disable the F1 Live Timing enhancement layer."""
    feed = _get_feed()

    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        enable = data.get("enable", True)
        if enable:
            success = feed.enable_f1_live_timing()
            return jsonify({"f1_live": feed.f1_live_enabled, "connected": success})
        else:
            feed.disable_f1_live_timing()
            return jsonify({"f1_live": False})

    return jsonify({"f1_live": feed.f1_live_enabled})


@live_bp.route("/cockpit")
def cockpit():
    """Serve the live cockpit telemetry page."""
    from src.shared import TEAM_COLORS_HEX

    all_drivers = [
        {"num": num, **_driver_meta(did), "id": did}
        for num, did in sorted(DRIVER_NUMBER_MAP.items())
    ]

    return render_template(
        "cockpit.html",
        TEAM_COLORS_HEX=TEAM_COLORS_HEX,
        ALL_DRIVERS=all_drivers,
    )
