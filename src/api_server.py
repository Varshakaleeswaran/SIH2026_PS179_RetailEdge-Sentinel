"""
RetailEdge Sentinel — FastAPI API Server
-------------------------------------------------------------------------------
Serves real-time telemetry from the RetailPipeline to the Enterprise HTML
Command Center (command_center.html).

Endpoints:
    GET  /                  → health check / index
    GET  /metrics           → full live telemetry JSON (polled by HTML every 2 s)
    GET  /metrics/stream    → Server-Sent Events (SSE) for push-based updates
    POST /scenario/rush     → trigger rush-hour simulation state
    POST /scenario/depletion→ trigger shelf depletion simulation state
    POST /scenario/reset    → reset to baseline
    GET  /video/frame/{cam} → JPEG frame from pipeline (MJPEG-style)
    GET  /patent/tier1      → Tier-1 Adaptive Trigger telemetry
    GET  /patent/pqci       → PQCI engine telemetry
    GET  /patent/inventory  → Inventory Density engine telemetry

Run with:
    python -m uvicorn src.api_server:app --host 0.0.0.0 --port 8502 --reload
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from fastapi import FastAPI, HTTPException, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError:
    raise RuntimeError(
        "FastAPI not installed. Run: pip install fastapi uvicorn[standard]"
    )

import yaml

from src.tier1_trigger import Tier1AdaptiveTrigger
from src.pqci_engine import PredictiveQueueEngine
from src.inventory_density import InventoryDensityEngine

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Load project settings
# ---------------------------------------------------------------------------
def _load_settings() -> Dict:
    cfg = PROJECT_ROOT / "config" / "settings.yaml"
    if cfg.exists():
        with open(cfg) as f:
            return yaml.safe_load(f) or {}
    return {}


SETTINGS = _load_settings()
NODE_ID = "RETAIL_EDGE_QCS6490_01"

# ---------------------------------------------------------------------------
# Shared engine instances (module-level singletons)
# ---------------------------------------------------------------------------
tier1 = Tier1AdaptiveTrigger(sensitivity_threshold=1500)
pqci = PredictiveQueueEngine(window_sec=60)
inv_density = InventoryDensityEngine()

# ---------------------------------------------------------------------------
# In-process shared store state (updated by /scenario endpoints for demo)
# ---------------------------------------------------------------------------
_sim_state: Dict[str, Any] = {
    "sim_rush_hour": False,
    "sim_depletion": False,
    # baseline metrics (updated as pipeline runs, or simulated below)
    "occupancy": 8,
    "footfall": 42,
    "queue_count": 2,
    "tfif": 1.18,
    "shelf_density_pct": 74.0,
    "fps": 15.0,
    "inference_ms": 35.0,
    "risk_score": 22.0,
    "risk_level": "LOW",
    "store_state": "NORMAL",
    "traffic_level": "LOW",
    "inflow_count": 14,
    "billing_count": 12,
    "avg_dwell_sec": 48.0,
    "entrance_occ": 2,
    "shelf_occ": 4,
    "billing_occ": 2,
    "tier1_efficiency_pct": 0.0,
    "recommendations": [],
    "last_updated": time.time(),
    # per-camera simulated values
    "cam1_inflow": 18,
    "cam1_outflow": 11,
    "cam1_roi_count": 3,
    "cam1_cross_today": 42,
    "cam2_density": 74.0,
    "cam2_dwell": 78,
    "cam2_oos_prob": 12.0,
    "cam2_status": "STOCK ADEQUATE",
    "cam3_queue_count": 2,
    "cam3_wait_est": 2.4,
    "cam3_status": "BALANCED",
    "hw_temp": 42.4,
    "hw_ram": 1.84,
}

# Try to import the running pipeline (optional - ok if Streamlit not running)
_pipeline_ref = None

def _get_pipeline():
    """Try to get a reference to the running pipeline (best-effort)."""
    global _pipeline_ref
    if _pipeline_ref is not None:
        return _pipeline_ref
    try:
        # Check if there is a shared state file written by the pipeline
        state_path = PROJECT_ROOT / "data" / "pipeline_state.json"
        if state_path.exists():
            with open(state_path) as f:
                return json.load(f)
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RetailEdge Sentinel API",
    description="Real-time retail intelligence telemetry for SIH26179",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_telemetry() -> Dict:
    """Build the live telemetry dict — merges sim overrides with base state."""
    s = dict(_sim_state)

    # Pull from pipeline state file if present
    live = _get_pipeline()
    if live:
        s.update(live)

    # Apply scenario overrides
    if s["sim_rush_hour"]:
        s["occupancy"] = max(s.get("occupancy", 8), 24)
        s["queue_count"] = max(s.get("queue_count", 2), 8)
        s["tfif"] = 3.84
        s["cam3_queue_count"] = 8
        s["cam3_wait_est"] = 6.8
        s["cam3_status"] = "SURGE DETECTED"
        s["risk_level"] = "CRITICAL"
        s["risk_score"] = 88.0
        s["store_state"] = "HIGH_TRAFFIC"

    if s["sim_depletion"]:
        s["shelf_density_pct"] = 18.0
        s["cam2_density"] = 18.0
        s["cam2_oos_prob"] = 92.0
        s["cam2_status"] = "DEPLETION DETECTED"

    # Sync PQCI
    pqci.update_rates(s.get("inflow_count", 0), s.get("billing_count", 0))
    pqci_result = pqci.compute_pqci(current_queue_length=s.get("queue_count", 0))
    s["tfif"] = s["tfif"] if s["sim_rush_hour"] else pqci_result.tfif
    s["pqci_state"] = pqci_result.pqci_state if not s["sim_rush_hour"] else "SURGE"
    s["pqci_action"] = pqci_result.action
    s["pqci_eta_minutes"] = pqci_result.eta_minutes
    s["pqci_confidence"] = pqci_result.confidence

    # Tier-1 stats
    t1_stats = tier1.get_efficiency_stats()
    s["tier1_efficiency_pct"] = t1_stats["inference_saved_pct"]
    s["tier1_triggers"] = t1_stats["detection_triggers"]
    s["tier1_skips"] = t1_stats["frames_skipped"]

    # Inventory density engine summary
    inv_summary = inv_density.get_summary()
    s["inv_summary"] = inv_summary

    s["last_updated"] = time.time()
    s["node_id"] = NODE_ID

    return s


def _safe_frame_jpg(cam_id: int = 1) -> Optional[bytes]:
    """Return a JPEG bytes snapshot of the current annotated frame (best-effort)."""
    frame_path = PROJECT_ROOT / "data" / f"frame_cam{cam_id}.jpg"
    if frame_path.exists():
        with open(frame_path, "rb") as f:
            return f.read()
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=JSONResponse)
async def root():
    return {
        "service": "RetailEdge Sentinel API",
        "version": "1.0.0",
        "node_id": NODE_ID,
        "status": "operational",
        "uptime_sec": round(time.time() - _sim_state.get("_start", time.time()), 1),
        "endpoints": [
            "/metrics",
            "/metrics/stream",
            "/scenario/rush",
            "/scenario/depletion",
            "/scenario/reset",
            "/video/frame/1",
            "/patent/tier1",
            "/patent/pqci",
            "/patent/inventory",
        ],
    }


@app.get("/metrics", response_class=JSONResponse)
async def get_metrics():
    """Full live telemetry — polled by command_center.html every 2 s."""
    return _build_telemetry()


@app.get("/metrics/stream")
async def metrics_stream():
    """Server-Sent Events stream for push-based dashboard updates."""
    def _event_generator():
        while True:
            data = json.dumps(_build_telemetry())
            yield f"data: {data}\n\n"
            time.sleep(2)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/scenario/rush")
async def trigger_rush():
    """Activate rush-hour simulation (TFIF surge, queue overload)."""
    _sim_state["sim_rush_hour"] = True
    _sim_state["occupancy"] = 27
    _sim_state["queue_count"] = 8
    _sim_state["inflow_count"] += 6
    logger.info("Scenario: Rush Hour activated")
    return {"status": "ok", "scenario": "rush_hour", "active": True}


@app.post("/scenario/depletion")
async def trigger_depletion():
    """Activate shelf depletion simulation."""
    _sim_state["sim_depletion"] = True
    logger.info("Scenario: Shelf Depletion activated")
    return {"status": "ok", "scenario": "depletion", "active": True}


@app.post("/scenario/reset")
async def reset_scenarios():
    """Reset all simulation overrides to baseline."""
    _sim_state["sim_rush_hour"] = False
    _sim_state["sim_depletion"] = False
    _sim_state["occupancy"] = 8
    _sim_state["queue_count"] = 2
    _sim_state["tfif"] = 1.18
    _sim_state["shelf_density_pct"] = 74.0
    _sim_state["inflow_count"] = 14
    _sim_state["billing_count"] = 12
    logger.info("Simulation state reset to baseline")
    return {"status": "ok", "reset": True}


@app.get("/video/frame/{cam_id}")
async def video_frame(cam_id: int = 1):
    """Return latest annotated JPEG frame for camera `cam_id`."""
    jpg = _safe_frame_jpg(cam_id)
    if jpg:
        return Response(content=jpg, media_type="image/jpeg")

    # Generate a placeholder frame with text
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(img, f"CAM_{cam_id:02d}", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 200), 2)
    cv2.putText(img, "Awaiting pipeline", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(img, "Run app.py first", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 200), 1)
    _, jpg_buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return Response(content=jpg_buf.tobytes(), media_type="image/jpeg")


@app.get("/patent/tier1", response_class=JSONResponse)
async def patent_tier1():
    """Return Tier-1 Adaptive Trigger telemetry."""
    return tier1.get_efficiency_stats()


@app.get("/patent/pqci", response_class=JSONResponse)
async def patent_pqci():
    """Return latest PQCI result."""
    s = _sim_state
    result = pqci.compute_pqci(current_queue_length=s.get("queue_count", 0))
    return {
        **pqci.to_dict(result),
        "tfif_history": pqci.get_tfif_history(last_n=30),
    }


@app.get("/patent/inventory", response_class=JSONResponse)
async def patent_inventory():
    """Return Inventory Density Engine summary."""
    return inv_density.get_summary()


# ---------------------------------------------------------------------------
# Background telemetry fluctuation (keeps demo alive without pipeline)
# ---------------------------------------------------------------------------
import asyncio
import random

@app.on_event("startup")
async def _start_background_fluctuation():
    """Slowly vary metrics to make the live dashboard feel dynamic."""
    _sim_state["_start"] = time.time()

    async def _fluctuate():
        while True:
            await asyncio.sleep(3)
            if not _sim_state["sim_rush_hour"]:
                _sim_state["occupancy"] = max(0, _sim_state["occupancy"] + random.randint(-1, 2))
                _sim_state["footfall"] = _sim_state["footfall"] + random.randint(0, 2)
                _sim_state["queue_count"] = max(0, _sim_state["queue_count"] + random.randint(-1, 1))
                _sim_state["fps"] = round(random.uniform(12.0, 18.0), 1)
                _sim_state["inference_ms"] = round(random.uniform(28.0, 55.0), 1)
                _sim_state["hw_temp"] = round(random.uniform(40.0, 46.0), 1)
                _sim_state["hw_ram"] = round(random.uniform(1.5, 2.2), 2)
                _sim_state["inflow_count"] = _sim_state["inflow_count"] + random.randint(0, 1)
                _sim_state["billing_count"] = _sim_state["billing_count"] + random.randint(0, 1)
                # Sync cam fields
                _sim_state["cam1_inflow"] = max(0, 18 + random.randint(-3, 5))
                _sim_state["cam1_roi_count"] = max(0, 3 + random.randint(-1, 2))
                _sim_state["cam3_queue_count"] = _sim_state["queue_count"]
                _sim_state["cam3_wait_est"] = round(max(0.5, _sim_state["queue_count"] * 1.2), 1)

            if not _sim_state["sim_depletion"]:
                _sim_state["shelf_density_pct"] = round(
                    max(15.0, min(95.0, _sim_state["shelf_density_pct"] + random.uniform(-1.0, 0.5))), 1
                )
                _sim_state["cam2_density"] = _sim_state["shelf_density_pct"]

            # Feed Tier-1 trigger with some simulated data
            dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            if random.random() > 0.4:
                cv2.rectangle(dummy_frame, (20, 20), (80, 80), (255, 255, 255), -1)
            tier1.should_run_detection(dummy_frame, zone_id="entrance")

    asyncio.create_task(_fluctuate())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api_server:app", host="0.0.0.0", port=8502, reload=False, log_level="info")
