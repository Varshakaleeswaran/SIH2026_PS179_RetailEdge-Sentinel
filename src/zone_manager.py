"""
RetailEdge Sentinel - Retail Zone Manager
Manages store zones (ENTRANCE, SHELF, BILLING/CHECKOUT):
- Point-in-polygon geometric testing
- Entrance line-crossing detection for footfall counting
- High-contrast visual overlays with zone labels and active occupancy counters
"""

from typing import Dict, List, Tuple, Optional
import yaml
from pathlib import Path
import cv2
import numpy as np

class RetailZone:
    def __init__(self, zone_id: str, name: str, zone_type: str, polygon: List[List[int]], color: List[int], entry_line: Optional[dict] = None):
        self.zone_id = zone_id
        self.name = name
        self.zone_type = zone_type # entry, browsing, checkout
        self.polygon_pts = np.array(polygon, dtype=np.int32)
        # color in RGB converted to BGR for OpenCV
        self.color_bgr = (int(color[2]), int(color[1]), int(color[0]))
        self.entry_line = entry_line
        self.current_occupancy = 0

    def contains_point(self, point: Tuple[int, int]) -> bool:
        """Check if a point (x, y) is inside this zone's polygon."""
        res = cv2.pointPolygonTest(self.polygon_pts, (float(point[0]), float(point[1])), False)
        return res >= 0

    def get_centroid(self) -> Tuple[int, int]:
        """Compute the visual centroid of the polygon for label placement."""
        M = cv2.moments(self.polygon_pts)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        return (int(self.polygon_pts[0][0]), int(self.polygon_pts[0][1]))


class ZoneManager:
    """Manages zone definitions, geometric queries, and visual rendering."""

    def __init__(self, config_path: str = "config/zones.yaml"):
        self.config_path = Path(config_path)
        self.zones: Dict[str, RetailZone] = {}
        self.load_zones()

    def load_zones(self):
        """Load zone definitions from YAML configuration."""
        if not self.config_path.exists():
            # Create default zones
            self._create_default_zones()
            return

        try:
            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f)
            
            raw_zones = data.get("zones", {})
            for key, z_info in raw_zones.items():
                zone = RetailZone(
                    zone_id=z_info.get("id", key.upper()),
                    name=z_info.get("name", key.capitalize()),
                    zone_type=z_info.get("type", "general"),
                    polygon=z_info.get("polygon", []),
                    color=z_info.get("color", [100, 100, 100]),
                    entry_line=z_info.get("entry_line")
                )
                self.zones[zone.zone_id] = zone
        except Exception as e:
            self._create_default_zones()

    def _create_default_zones(self):
        """Fallback zone layout if configuration is missing."""
        self.zones["ENTRANCE"] = RetailZone(
            "ENTRANCE", "Entrance & Foyer", "entry",
            [[20, 30], [260, 30], [260, 220], [20, 220]],
            [46, 204, 113],
            {"start": [20, 190], "end": [260, 190]}
        )
        self.zones["SHELF"] = RetailZone(
            "SHELF", "Merchandise & Shelf Aisle", "browsing",
            [[320, 30], [620, 30], [620, 230], [320, 230]],
            [243, 156, 18]
        )
        self.zones["BILLING"] = RetailZone(
            "BILLING", "Billing Counter & Queue", "checkout",
            [[220, 270], [620, 270], [620, 460], [220, 460]],
            [231, 76, 60]
        )

    def determine_zone(self, foot_pos: Tuple[int, int]) -> Optional[str]:
        """Determine which zone contains the ground foot contact point."""
        for zone_id, zone in self.zones.items():
            if zone.contains_point(foot_pos):
                return zone_id
        return None

    def check_entrance_line_crossing(self, trajectory: List[Tuple[int, int]]) -> bool:
        """
        Check if a track trajectory crosses the configured entrance line.
        Trajectory: list of recent points (x, y).
        """
        entrance_zone = self.zones.get("ENTRANCE")
        if not entrance_zone or not entrance_zone.entry_line:
            return False

        line_y = entrance_zone.entry_line["start"][1]
        line_x1 = entrance_zone.entry_line["start"][0]
        line_x2 = entrance_zone.entry_line["end"][0]

        if len(trajectory) < 2:
            return False

        # Check the transition between previous point and current point
        prev_x, prev_y = trajectory[-2]
        curr_x, curr_y = trajectory[-1]

        # In horizontal entry line: crossing occurs if moving from prev_y < line_y to curr_y >= line_y
        in_x_bounds = min(line_x1, line_x2) - 30 <= curr_x <= max(line_x1, line_x2) + 30
        if in_x_bounds and prev_y < line_y <= curr_y:
            return True

        return False

    def draw_zones(self, frame: np.ndarray, zone_occupancies: Optional[Dict[str, int]] = None) -> np.ndarray:
        """
        Draw translucent colored polygons and high-contrast labels over the video frame.
        """
        overlay = frame.copy()
        output = frame.copy()
        alpha = 0.22 # Transparency factor

        for zone_id, zone in self.zones.items():
            pts = zone.polygon_pts
            # Fill polygon on overlay
            cv2.fillPoly(overlay, [pts], zone.color_bgr)
            # Outline on output frame
            cv2.polylines(output, [pts], isClosed=True, color=zone.color_bgr, thickness=2, lineType=cv2.LINE_AA)

            # Draw entry line if present
            if zone.entry_line:
                p1 = tuple(zone.entry_line["start"])
                p2 = tuple(zone.entry_line["end"])
                cv2.line(output, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(output, "ENTRY GATE", (p1[0] + 10, p1[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            # Zone label badge
            cx, cy = zone.get_centroid()
            occ = zone_occupancies.get(zone_id, 0) if zone_occupancies else 0
            label = f"{zone.name.upper()} | {occ} present"

            # Draw badge background box
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.45
            thick = 1
            (w, h), _ = cv2.getTextSize(label, font, scale, thick)
            box_x1 = max(10, cx - w // 2 - 8)
            box_y1 = max(10, cy - h - 10)
            box_x2 = box_x1 + w + 16
            box_y2 = box_y1 + h + 14

            cv2.rectangle(output, (box_x1, box_y1), (box_x2, box_y2), (20, 24, 30), -1)
            cv2.rectangle(output, (box_x1, box_y1), (box_x2, box_y2), zone.color_bgr, 1)
            cv2.putText(output, label, (box_x1 + 8, box_y1 + h + 5), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

        # Blend overlay
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        return output
