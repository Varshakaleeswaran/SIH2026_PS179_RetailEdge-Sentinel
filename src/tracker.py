"""
RetailEdge Sentinel - Anonymous Object Tracker
Provides temporary, non-identifying tracking across consecutive frames:
- Assigns ephemeral IDs (Person #1, Person #2, ...)
- Tracks trajectory, centroid, ground/foot contact point, and dwell time
- Guarantees NO personal identity, face, or demographic association
"""

import math
import time
from typing import List, Dict, Optional, Tuple
import numpy as np
from src.detector import Detection
from src.privacy import PrivacyGuard

class TrackedObject:
    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int], confidence: float):
        self.track_id = track_id
        self.anonymous_label = PrivacyGuard.format_anonymous_id(track_id)
        self.bbox = bbox
        self.confidence = confidence
        self.first_seen = time.time()
        self.last_seen = self.first_seen
        self.frames_missing = 0
        self.counted_entry = False
        self.current_zone: Optional[str] = None
        self.zone_entry_time: float = self.first_seen
        self.zone_dwell_times: Dict[str, float] = {}

        # Trajectory history for motion & line-crossing
        cx, cy = self._calc_centroid(bbox)
        self.trajectory: List[Tuple[int, int]] = [(cx, cy)]

    @staticmethod
    def _calc_centroid(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    @property
    def centroid(self) -> Tuple[int, int]:
        return self._calc_centroid(self.bbox)

    @property
    def foot_position(self) -> Tuple[int, int]:
        """Ground plane contact point: center-x, bottom-y."""
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int(y2))

    @property
    def total_dwell_time(self) -> float:
        return time.time() - self.first_seen

    def update(self, bbox: Tuple[int, int, int, int], confidence: float):
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = time.time()
        self.frames_missing = 0
        cx, cy = self._calc_centroid(bbox)
        self.trajectory.append((cx, cy))
        if len(self.trajectory) > 40:
            self.trajectory.pop(0)

    def update_zone(self, new_zone: Optional[str]):
        now = time.time()
        if self.current_zone != new_zone:
            if self.current_zone is not None:
                dwell = now - self.zone_entry_time
                self.zone_dwell_times[self.current_zone] = self.zone_dwell_times.get(self.current_zone, 0.0) + dwell
            self.current_zone = new_zone
            self.zone_entry_time = now


class AnonymousTracker:
    """
    Lightweight, resilient IoU and centroid-based tracker for anonymous shopper tracking.
    Operates without heavy external C++ binaries, ensuring reliable performance on Windows laptops.
    """

    def __init__(self, max_missing_frames: int = 15, iou_threshold: float = 0.25):
        self.max_missing_frames = max_missing_frames
        self.iou_threshold = iou_threshold
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_track_id = 1

    @staticmethod
    def _compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
        boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

        denom = float(boxAArea + boxBArea - interArea)
        return interArea / denom if denom > 0 else 0.0

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """Update existing tracks with incoming detections, match using IoU/distance, and register new tracks."""
        # Increment missing counter on all existing tracks
        for track in self.tracks.values():
            track.frames_missing += 1

        matched_tracks = set()
        matched_detections = set()

        # Match existing tracks with detections based on IoU
        if self.tracks and detections:
            track_ids = list(self.tracks.keys())
            iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)

            for i, tid in enumerate(track_ids):
                for j, det in enumerate(detections):
                    iou_matrix[i, j] = self._compute_iou(self.tracks[tid].bbox, det.bbox)

            # Greedy matching
            for _ in range(min(len(track_ids), len(detections))):
                max_val = np.max(iou_matrix)
                if max_val < self.iou_threshold:
                    break
                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                tid = track_ids[i]
                self.tracks[tid].update(detections[j].bbox, detections[j].confidence)
                matched_tracks.add(tid)
                matched_detections.add(j)
                iou_matrix[i, :] = -1.0
                iou_matrix[:, j] = -1.0

        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_detections:
                new_track = TrackedObject(self.next_track_id, det.bbox, det.confidence)
                self.tracks[self.next_track_id] = new_track
                self.next_track_id += 1

        # Remove dead tracks
        dead_ids = [
            tid for tid, track in self.tracks.items()
            if track.frames_missing > self.max_missing_frames
        ]
        for tid in dead_ids:
            del self.tracks[tid]

        # Return currently active tracks
        active_tracks = [track for track in self.tracks.values() if track.frames_missing == 0]
        return active_tracks
