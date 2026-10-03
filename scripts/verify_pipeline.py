"""
RetailEdge Sentinel - Pipeline Verification on Real Pedestrian CCTV Video
"""

import sys
import time
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import RetailPipeline

def verify():
    video_path = PROJECT_ROOT / "videos" / "people_cctv.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return False

    pipeline = RetailPipeline()
    print(f"Testing pipeline on {video_path.name}...")

    frame_idx = 0
    t0 = time.time()
    while cap.isOpened() and frame_idx < 30:
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated_frame, telemetry = pipeline.process_frame(frame)
        frame_idx += 1
        if frame_idx % 10 == 0:
            occ = telemetry["shopper"]["occupancy"]
            footfall = telemetry["shopper"]["footfall"]
            risk = telemetry["risk"]["risk_score"]
            state = telemetry["store_state"]["overall_state"]
            fps = telemetry["fps"]
            lat = telemetry["inference_time_ms"]
            tracks = telemetry["active_tracks_count"]
            print(f"Frame {frame_idx:02d} | FPS: {fps} | Latency: {lat}ms | Occupancy: {occ} | Active Tracks: {tracks} | Footfall: {footfall} | Risk: {risk}/100 | State: {state}")

    cap.release()
    total_time = time.time() - t0
    print(f"Processed {frame_idx} frames in {round(total_time, 2)}s.")
    print(f"Active tracks: {telemetry['active_tracks_count']}")
    print(f"Store state: {telemetry['store_state']['overall_state']}")
    print(f"Top Recommendation: {telemetry['recommendations'][0]['action']}")
    return True

if __name__ == "__main__":
    verify()
