"""
RetailEdge Sentinel - Shopper Intelligence Module
Calculates real-time customer analytics:
- Cumulative footfall with strict deduplication per tracking ID
- Real-time store and zone occupancy
- Average dwell time across active and completed visits
- Traffic level classification (LOW, MEDIUM, HIGH)
- Spatial heat density accumulation for shopper flow analysis
"""

from typing import List, Dict, Any, Set, Tuple
import time
import numpy as np
from src.tracker import TrackedObject
from src.zone_manager import ZoneManager

class ShopperIntelligence:
    def __init__(self, traffic_thresholds: Dict[str, int] = None, frame_dims: Tuple[int, int] = (640, 480)):
        self.thresholds = traffic_thresholds or {"low": 4, "medium": 8, "high": 12}
        self.frame_dims = frame_dims
        self.cumulative_footfall = 0
        self.entered_track_ids: Set[int] = set()
        self.completed_dwell_times: List[float] = []
        
        # Spatial heatmap accumulator (downscaled grid for efficiency)
        grid_w, grid_h = max(10, frame_dims[0] // 10), max(10, frame_dims[1] // 10)
        self.heat_grid = np.zeros((grid_h, grid_w), dtype=np.float32)

    def process(self, active_tracks: List[TrackedObject], zone_manager: ZoneManager) -> Dict[str, Any]:
        """Process active tracks and update shopper intelligence metrics."""
        now = time.time()
        occupancy = len(active_tracks)

        # Initialize zone counters
        zone_occupancy: Dict[str, int] = {
            "ENTRANCE": 0,
            "SHELF": 0,
            "BILLING": 0
        }

        active_dwell_times: List[float] = []

        for track in active_tracks:
            # 1. Update zone location
            foot_pos = track.foot_position
            current_zone = zone_manager.determine_zone(foot_pos)
            track.update_zone(current_zone)

            if current_zone in zone_occupancy:
                zone_occupancy[current_zone] += 1

            # 2. Footfall entrance line crossing check
            if not track.counted_entry:
                # Check line crossing or if person is firmly inside the entrance zone
                crossed_line = zone_manager.check_entrance_line_crossing(track.trajectory)
                in_entrance = current_zone == "ENTRANCE"

                if crossed_line or in_entrance:
                    if track.track_id not in self.entered_track_ids:
                        self.cumulative_footfall += 1
                        self.entered_track_ids.add(track.track_id)
                        track.counted_entry = True

            # 3. Dwell time tracking
            dwell = track.total_dwell_time
            active_dwell_times.append(dwell)

            # 4. Accumulate spatial heat
            gx = int(np.clip(foot_pos[0] / self.frame_dims[0] * self.heat_grid.shape[1], 0, self.heat_grid.shape[1] - 1))
            gy = int(np.clip(foot_pos[1] / self.frame_dims[1] * self.heat_grid.shape[0], 0, self.heat_grid.shape[0] - 1))
            self.heat_grid[gy, gx] += 1.0

        # Calculate average dwell time
        all_dwells = active_dwell_times + self.completed_dwell_times[-30:] # combine active & recent past
        avg_dwell = round(float(np.mean(all_dwells)), 1) if all_dwells else 0.0

        # Classify traffic level
        if occupancy < self.thresholds.get("low", 4):
            traffic_level = "LOW"
        elif occupancy <= self.thresholds.get("medium", 8):
            traffic_level = "MEDIUM"
        else:
            traffic_level = "HIGH"

        return {
            "footfall": self.cumulative_footfall,
            "occupancy": occupancy,
            "entrance_occupancy": zone_occupancy.get("ENTRANCE", 0),
            "shelf_occupancy": zone_occupancy.get("SHELF", 0),
            "billing_occupancy": zone_occupancy.get("BILLING", 0),
            "average_dwell_time": avg_dwell,
            "traffic_level": traffic_level,
            "zone_breakdown": zone_occupancy
        }

    def record_departed_track(self, track: TrackedObject):
        """Record completed customer dwell time when track leaves."""
        self.completed_dwell_times.append(track.total_dwell_time)
        if len(self.completed_dwell_times) > 100:
            self.completed_dwell_times.pop(0)

    def get_heat_map_normalized(self) -> np.ndarray:
        """Returns normalized 2D heatmap matrix (0.0 to 1.0) for visual display."""
        max_val = np.max(self.heat_grid)
        if max_val > 0:
            return self.heat_grid / max_val
        return self.heat_grid
