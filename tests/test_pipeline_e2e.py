"""
Unit and End-to-End tests for Tracker, Zones, and Full Pipeline
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.tracker import AnonymousTracker
from src.detector import Detection
from src.zone_manager import ZoneManager
from src.pipeline import RetailPipeline
from src.privacy import PrivacyGuard

def test_anonymous_tracker_assigns_non_identifying_ids():
    tracker = AnonymousTracker()
    det1 = Detection(bbox=(50, 50, 100, 150), confidence=0.85, class_id=0, class_name="person")
    det2 = Detection(bbox=(350, 100, 400, 220), confidence=0.91, class_id=0, class_name="person")

    tracks = tracker.update([det1, det2])
    assert len(tracks) == 2
    assert tracks[0].anonymous_label == "Person #1"
    assert tracks[1].anonymous_label == "Person #2"

    # Next frame with slight movement
    det1_moved = Detection(bbox=(52, 53, 102, 153), confidence=0.87, class_id=0, class_name="person")
    tracks_f2 = tracker.update([det1_moved])
    assert len(tracks_f2) == 1
    assert tracks_f2[0].track_id == 1 # ID maintained across frames

def test_zone_point_in_polygon():
    zm = ZoneManager()
    # Foot position inside Entrance (20-260, 30-220)
    assert zm.determine_zone((100, 100)) == "ENTRANCE"
    # Foot position inside Shelf (320-620, 30-230)
    assert zm.determine_zone((400, 100)) == "SHELF"
    # Foot position inside Billing (220-620, 270-460)
    assert zm.determine_zone((300, 350)) == "BILLING"
    # Outside all zones
    assert zm.determine_zone((10, 470)) is None

def test_privacy_guard_enforces_rules():
    status = PrivacyGuard.get_privacy_status()
    assert status["status"] == "ACTIVE & COMPLIANT"
    assert status["mode"] == "LOCAL_ANONYMOUS_EDGE"
    assert len(status["guarantees"]) >= 5

def test_full_pipeline_processes_synthetic_frame():
    pipeline = RetailPipeline()
    # Create empty test frame (640x480)
    test_frame = np.full((480, 640, 3), 200, dtype=np.uint8)

    annotated_frame, telemetry = pipeline.process_frame(test_frame)
    assert annotated_frame is not None
    assert annotated_frame.shape == (480, 640, 3)
    assert "fps" in telemetry
    assert "shopper" in telemetry
    assert "store_state" in telemetry
    assert "risk" in telemetry
    assert "recommendations" in telemetry

if __name__ == "__main__":
    test_anonymous_tracker_assigns_non_identifying_ids()
    test_zone_point_in_polygon()
    test_privacy_guard_enforces_rules()
    test_full_pipeline_processes_synthetic_frame()
    print("Pipeline & Tracker tests: PASSED")
