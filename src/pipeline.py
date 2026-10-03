"""
RetailEdge Sentinel - Unified Processing Pipeline
Orchestrates the complete edge retail intelligence flow:
CAMERA / VIDEO -> DETECTOR -> TRACKER -> ZONES -> 3 INTELLIGENCE MODULES -> STORE STATE -> RISK ENGINE -> RECOMMENDATIONS
"""

import time
import logging
from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np
import psutil

from src.detector import LocalInferenceEngine, InferenceEngine, Detection
from src.tracker import AnonymousTracker, TrackedObject
from src.zone_manager import ZoneManager
from src.shopper_intelligence import ShopperIntelligence
from src.inventory_intelligence import InventoryIntelligence
from src.queue_intelligence import QueueIntelligence
from src.store_state_engine import StoreStateEngine
from src.risk_engine import RiskEngine
from src.recommendation_engine import RecommendationEngine
from src.database import DatabaseManager

logger = logging.getLogger(__name__)

class RetailPipeline:
    def __init__(self, settings: Dict[str, Any] = None):
        self.settings = settings or {}
        
        # 1. Initialize detector (Local by default)
        model_cfg = self.settings.get("model", {})
        self.detector: InferenceEngine = LocalInferenceEngine(
            model_name=model_cfg.get("model_type", "yolov8n") + ".pt",
            confidence_thresh=model_cfg.get("confidence_threshold", 0.35)
        )

        # 2. Anonymous tracker
        self.tracker = AnonymousTracker()

        # 3. Zone manager
        self.zone_manager = ZoneManager()

        # 4. Intelligence modules
        shopper_cfg = self.settings.get("shopper_intelligence", {})
        self.shopper_intel = ShopperIntelligence(
            traffic_thresholds=shopper_cfg.get("traffic_thresholds", {"low": 4, "medium": 8, "high": 12})
        )

        inv_cfg = self.settings.get("inventory_intelligence", {})
        self.shelf_intel = InventoryIntelligence(
            initial_shelves=inv_cfg.get("shelves")
        )

        queue_cfg = self.settings.get("queue_intelligence", {})
        self.queue_intel = QueueIntelligence(
            thresholds=queue_cfg.get("thresholds", {"low": 2, "medium": 5, "high": 8}),
            default_service_rate=queue_cfg.get("default_service_rate", 1.5)
        )

        # 5. Synthesis engines
        self.state_engine = StoreStateEngine()
        self.risk_engine = RiskEngine()
        self.rec_engine = RecommendationEngine()

        # 6. SQLite storage
        self.db = DatabaseManager(self.settings.get("storage", {}).get("db_path", "data/retail_edge.db"))

        # Performance telemetry
        self.fps = 0.0
        self.inference_time_ms = 0.0
        self.frame_count = 0
        self.last_time = time.perf_counter()
        self.last_db_log_time = 0.0

        # Adaptive inference settings
        self.adaptive_enabled = self.settings.get("adaptive_inference", {}).get("enabled", True)
        self.low_skip = self.settings.get("adaptive_inference", {}).get("low_activity_skip", 3)
        self.last_detections: List[Detection] = []

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes one full cycle of the edge pipeline.
        Returns:
            Tuple of (Annotated visual frame, Full telemetry dictionary)
        """
        t_start = time.perf_counter()
        self.frame_count += 1

        if frame is None or frame.size == 0:
            return frame, {}

        # Resize if frame dimensions differ significantly for standard processing
        h, w = frame.shape[:2]
        if w > 1000:
            scale = 640.0 / w
            frame = cv2.resize(frame, (640, int(h * scale)))

        # Adaptive inference check
        skip_detection = False
        if self.adaptive_enabled and self.frame_count % self.low_skip != 0:
            # If previous occupancy was low, skip detection to conserve edge compute
            if len(self.tracker.tracks) <= 2:
                skip_detection = True

        if not skip_detection or not self.last_detections:
            detections, self.inference_time_ms = self.detector.detect(frame)
            self.last_detections = detections
        else:
            detections = self.last_detections

        # 1. Update anonymous tracking
        active_tracks = self.tracker.update(detections)

        # 2. Run Shopper Intelligence
        shopper_metrics = self.shopper_intel.process(active_tracks, self.zone_manager)

        # 3. Run Shelf / Inventory Intelligence
        shelf_occupancy = shopper_metrics.get("shelf_occupancy", 0)
        shelf_metrics = self.shelf_intel.process(shelf_occupancy)

        # 4. Run Queue Intelligence
        billing_occupancy = shopper_metrics.get("billing_occupancy", 0)
        queue_metrics = self.queue_intel.process(billing_occupancy)

        # 5. Synthesize Store State
        store_state = self.state_engine.evaluate(shopper_metrics, shelf_metrics, queue_metrics)

        # 6. Evaluate Operational Risk
        risk_metrics = self.risk_engine.compute_risk(shopper_metrics, shelf_metrics, queue_metrics)

        # 7. Generate Practical Recommendations
        recommendations = self.rec_engine.generate_recommendations(store_state, risk_metrics, shelf_metrics)

        # Calculate FPS
        t_end = time.perf_counter()
        elapsed = t_end - self.last_time
        self.last_time = t_end
        if elapsed > 0:
            instant_fps = 1.0 / elapsed
            self.fps = round(self.fps * 0.85 + instant_fps * 0.15, 1)

        cpu_usage = psutil.cpu_percent(interval=None)

        # Periodic SQLite snapshot logging (every 5 seconds)
        now = time.time()
        if now - self.last_db_log_time >= 5.0:
            self.last_db_log_time = now
            snapshot = {
                "footfall": shopper_metrics.get("footfall", 0),
                "occupancy": shopper_metrics.get("occupancy", 0),
                "queue_length": queue_metrics.get("queue_length", 0),
                "risk_score": risk_metrics.get("risk_score", 0.0),
                "risk_level": risk_metrics.get("risk_level", "LOW"),
                "store_state": store_state.get("overall_state", "NORMAL"),
                "fps": self.fps
            }
            self.db.log_metrics_snapshot(snapshot)

            # Check if high severity alert should be logged
            if risk_metrics.get("risk_level") in ["HIGH", "CRITICAL"]:
                self.db.log_event(
                    event_type=store_state.get("overall_state", "RISK_ALERT"),
                    zone="CHECKOUT" if queue_metrics.get("queue_length", 0) >= 4 else "STORE",
                    severity=risk_metrics.get("risk_level", "HIGH"),
                    message=f"Elevated risk: {risk_metrics.get('reasons', ['Risk spike'])[0]}"
                )

        # Draw visual annotations on frame
        annotated_frame = self._annotate_frame(frame, active_tracks, shopper_metrics, store_state, risk_metrics)

        # Bundle full system state for dashboard
        telemetry = {
            "fps": self.fps,
            "inference_time_ms": round(self.inference_time_ms, 1),
            "cpu_percent": cpu_usage,
            "shopper": shopper_metrics,
            "shelf": shelf_metrics,
            "queue": queue_metrics,
            "store_state": store_state,
            "risk": risk_metrics,
            "recommendations": recommendations,
            "adaptive_mode": "ACTIVE" if self.adaptive_enabled else "OFF",
            "active_tracks_count": len(active_tracks),
            "platform_info": self.detector.get_platform_info()
        }

        return annotated_frame, telemetry

    def _annotate_frame(self, frame: np.ndarray, tracks: List[TrackedObject], shopper_metrics: dict, store_state: dict, risk_metrics: dict) -> np.ndarray:
        """Render high-contrast visual cues: zones, track boxes, anonymous labels, and top status HUD."""
        out = frame.copy()

        # 1. Draw zones
        zone_occs = shopper_metrics.get("zone_breakdown", {})
        out = self.zone_manager.draw_zones(out, zone_occs)

        # 2. Draw tracks
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            color = (50, 205, 50) if track.current_zone == "ENTRANCE" else ((0, 165, 255) if track.current_zone == "SHELF" else (0, 0, 255))
            
            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            
            # Ground foot circle
            fx, fy = track.foot_position
            cv2.circle(out, (fx, fy), 4, color, -1)

            # Trajectory line
            if len(track.trajectory) > 1:
                pts = np.array(track.trajectory[-15:], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(out, [pts], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)

            # Anonymous label badge
            label = track.anonymous_label
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out, (x1, max(0, y1 - 20)), (x1 + w + 8, y1), color, -1)
            cv2.putText(out, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Top HUD Bar
        hud_h = 32
        hud_bg = out[:hud_h, :].copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], hud_h), (15, 20, 25), -1)
        cv2.addWeighted(out[:hud_h, :], 0.85, hud_bg, 0.15, 0, out[:hud_h, :])

        hud_text = (
            f"RetailEdge Sentinel  |  FPS: {self.fps}  |  Infer: {round(self.inference_time_ms, 1)}ms  |  "
            f"Occ: {shopper_metrics.get('occupancy', 0)}  |  State: {store_state.get('overall_state', 'NORMAL')}  |  "
            f"Risk: {risk_metrics.get('risk_score', 0)}/100"
        )
        cv2.putText(out, hud_text, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

        return out
