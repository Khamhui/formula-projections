"""
Live Prediction Flask Blueprint — SSE endpoint for real-time race updates.

Provides:
- GET /live/stream — SSE stream of live predictions (EventSource-compatible)
- GET /live/state  — Current race state as JSON
- GET /live/start  — Start live polling
- GET /live/stop   — Stop live polling
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

from flask import Blueprint, Response, jsonify, request

logger = logging.getLogger(__name__)

live_bp = Blueprint("live", __name__, url_prefix="/live")

# Module-level state (single-server, single-session)
_feed: Optional[object] = None
_predictor: Optional[object] = None


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


@live_bp.route("/start", methods=["POST"])
def start_live():
    """Start live data polling."""
    feed = _get_feed()

    # Load pre-race predictions
    try:
        import pandas as pd
        from pathlib import Path
        from data.models.predictor import F1Predictor

        cache_dir = Path(__file__).parent.parent / "data" / "cache" / "processed"
        model = F1Predictor()
        model.load()

        fm = pd.read_parquet(cache_dir / "feature_matrix.parquet")
        latest_season = int(fm["season"].max())
        latest_round = int(fm[fm["season"] == latest_season]["round"].max())
        race_data = fm[(fm["season"] == latest_season) & (fm["round"] == latest_round)]
        predictions = model.predict_race(race_data)

        _get_predictor(predictions)
    except Exception as e:
        logger.error("Failed to load pre-race predictions: %s", e)
        return jsonify({"error": str(e)}), 500

    feed.start_polling()
    return jsonify({"status": "polling_started"})


@live_bp.route("/stop", methods=["POST"])
def stop_live():
    """Stop live data polling."""
    feed = _get_feed()
    feed.stop_polling()
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


@live_bp.route("/stream")
def stream():
    """
    SSE endpoint for live prediction updates.

    Connect via EventSource:
        const es = new EventSource('/live/stream');
        es.onmessage = (e) => console.log(JSON.parse(e.data));
    """
    def generate():
        feed = _get_feed()
        predictor = _get_predictor()
        last_lap = -1

        while True:
            state = feed.get_current_state()

            if state and predictor:
                if state.lap != last_lap:
                    last_lap = state.lap
                    predictions = predictor.update(state)
                    data = {
                        "lap": state.lap,
                        "total_laps": state.total_laps,
                        "track_status": state.track_status,
                        "laps_remaining": state.laps_remaining,
                        "predictions": predictions.to_dict("records"),
                    }
                    yield f"data: {json.dumps(data)}\n\n"
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
