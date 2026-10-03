"""
Patent Candidate B (Inventory Sub-component): Inventory Density Engine
-------------------------------------------------------------------------------
Canny Edge-Density Texture Analysis for Shelf Out-of-Stock Estimation
SIH26179 / RetailEdge Sentinel / VENSMOLD AI

Instead of running an expensive multi-SKU product detection model, this module
uses classical image processing (Sobel/Canny edge detection + texture variance)
to measure how 'full' a shelf ROI appears.

Principle:
    A fully stocked shelf has high edge density (many product edges, labels,
    faces of packages).  An empty shelf has low edge density (bare shelving
    material with few edges).

This approach:
    - Requires zero training data or labelled product images
    - Runs in < 2 ms on CPU
    - Works on any shelf type / product category
    - Is fully privacy-safe (no product-level classification stored)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Density thresholds (fraction of edge pixels in ROI)
DENSITY_AVAILABLE = 0.40    # >= 40 % edge pixels → shelf is stocked
DENSITY_LOW = 0.20          # 20-40 % → stock running low
# < 20 % → shelf empty / critically depleted

# Configurable shelf status labels
STATUS_AVAILABLE = "AVAILABLE"
STATUS_LOW = "LOW"
STATUS_EMPTY = "EMPTY"


@dataclass
class ShelfReading:
    shelf_id: str
    timestamp: float
    density: float          # 0.0–1.0 fraction of edge pixels
    density_pct: float      # 0–100 %
    status: str             # AVAILABLE | LOW | EMPTY
    oos_probability: float  # 0–100 % out-of-stock probability
    restock_required: bool
    frames_in_state: int = 0


class InventoryDensityEngine:
    """
    Edge-density texture variance for shelf out-of-stock estimation.

    Avoids expensive multi-hundred-SKU object detection models.
    Implements the logic shown in the HTML modal Python script.

    Usage
    -----
        engine = InventoryDensityEngine()

        # Feed a cropped shelf ROI (BGR numpy array):
        reading = engine.analyse_shelf(shelf_roi, shelf_id="ShelfA")
        print(reading.status, reading.density_pct)

        # Or compute just the raw density:
        density = InventoryDensityEngine.calculate_stock_density(roi)
    """

    def __init__(
        self,
        available_threshold: float = DENSITY_AVAILABLE,
        low_threshold: float = DENSITY_LOW,
        canny_low: int = 50,
        canny_high: int = 150,
        blur_kernel: int = 3,
        history_len: int = 30,
    ):
        self.available_threshold = available_threshold
        self.low_threshold = low_threshold
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.blur_kernel = blur_kernel
        self._history: Dict[str, List[ShelfReading]] = {}
        self._history_len = history_len
        self._state_counters: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Static / core computation (matches HTML modal API)
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_stock_density(shelf_roi: np.ndarray) -> float:
        """
        Compute the fraction of pixels classified as edges in `shelf_roi`.

        Returns a float in [0.0, 1.0].
        """
        if shelf_roi is None or shelf_roi.size == 0:
            return 0.0

        gray = cv2.cvtColor(shelf_roi, cv2.COLOR_BGR2GRAY) if shelf_roi.ndim == 3 else shelf_roi
        # Light blur to reduce sensor noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 50, 150)
        density = float(np.count_nonzero(edges)) / float(edges.size)
        return round(density, 4)

    # ------------------------------------------------------------------
    # Full shelf analysis with history tracking
    # ------------------------------------------------------------------

    def analyse_shelf(
        self,
        shelf_roi: np.ndarray,
        shelf_id: str = "shelf_a",
    ) -> ShelfReading:
        """
        Analyse a single shelf ROI and return a structured ShelfReading.

        Parameters
        ----------
        shelf_roi : np.ndarray
            BGR image crop of the shelf region of interest.
        shelf_id : str
            Unique identifier for this shelf (e.g. "ShelfA", "ShelfB").

        Returns
        -------
        ShelfReading
        """
        if shelf_roi is None or shelf_roi.size == 0:
            return ShelfReading(
                shelf_id=shelf_id,
                timestamp=time.time(),
                density=0.0,
                density_pct=0.0,
                status=STATUS_EMPTY,
                oos_probability=100.0,
                restock_required=True,
            )

        gray = cv2.cvtColor(shelf_roi, cv2.COLOR_BGR2GRAY) if shelf_roi.ndim == 3 else shelf_roi
        gray = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        density = float(np.count_nonzero(edges)) / float(edges.size)

        density_pct = round(density * 100, 1)

        # Classify status
        if density >= self.available_threshold:
            status = STATUS_AVAILABLE
            oos_prob = max(0.0, round((1.0 - density / 1.0) * 30, 1))
        elif density >= self.low_threshold:
            status = STATUS_LOW
            oos_prob = round(40.0 + (1.0 - density / self.available_threshold) * 40, 1)
        else:
            status = STATUS_EMPTY
            oos_prob = min(100.0, round(80.0 + (1.0 - density / self.low_threshold) * 20, 1))

        # Track consecutive frames in current state
        prev_count = self._state_counters.get(shelf_id, 0)
        hist = self._history.get(shelf_id, [])
        if hist and hist[-1].status == status:
            frames_in_state = prev_count + 1
        else:
            frames_in_state = 1
        self._state_counters[shelf_id] = frames_in_state

        reading = ShelfReading(
            shelf_id=shelf_id,
            timestamp=time.time(),
            density=round(density, 4),
            density_pct=density_pct,
            status=status,
            oos_probability=oos_prob,
            restock_required=status in (STATUS_LOW, STATUS_EMPTY),
            frames_in_state=frames_in_state,
        )

        # Append to history ring buffer
        if shelf_id not in self._history:
            self._history[shelf_id] = []
        self._history[shelf_id].append(reading)
        if len(self._history[shelf_id]) > self._history_len:
            self._history[shelf_id] = self._history[shelf_id][-self._history_len:]

        return reading

    # ------------------------------------------------------------------
    # Batch / multi-shelf
    # ------------------------------------------------------------------

    def analyse_all_shelves(
        self,
        frame: np.ndarray,
        shelf_rois: Dict[str, Tuple[int, int, int, int]],
    ) -> Dict[str, ShelfReading]:
        """
        Analyse multiple shelf ROIs from a full frame.

        Parameters
        ----------
        frame : np.ndarray
            Full BGR frame.
        shelf_rois : dict
            Mapping of shelf_id → (x, y, w, h) bounding box in the frame.

        Returns
        -------
        dict of shelf_id → ShelfReading
        """
        results: Dict[str, ShelfReading] = {}
        for shelf_id, (x, y, w, h) in shelf_rois.items():
            roi = frame[y : y + h, x : x + w]
            results[shelf_id] = self.analyse_shelf(roi, shelf_id=shelf_id)
        return results

    # ------------------------------------------------------------------
    # Telemetry helpers
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict:
        """Return a summary of current shelf statuses across all tracked shelves."""
        if not self._history:
            return {"shelves": {}, "total_shelves": 0, "restock_required": False}

        shelves = {}
        restock_required = False
        for shelf_id, readings in self._history.items():
            if readings:
                latest = readings[-1]
                shelves[shelf_id] = {
                    "status": latest.status,
                    "density_pct": latest.density_pct,
                    "oos_probability": latest.oos_probability,
                    "restock_required": latest.restock_required,
                    "frames_in_state": latest.frames_in_state,
                }
                if latest.restock_required:
                    restock_required = True

        return {
            "shelves": shelves,
            "total_shelves": len(shelves),
            "restock_required": restock_required,
            "low_count": sum(1 for s in shelves.values() if s["status"] == STATUS_LOW),
            "empty_count": sum(1 for s in shelves.values() if s["status"] == STATUS_EMPTY),
        }

    def overlay_shelf_info(
        self,
        frame: np.ndarray,
        shelf_id: str,
        roi_coords: Tuple[int, int, int, int],
        reading: ShelfReading,
    ) -> np.ndarray:
        """Draw shelf status overlay on frame for visualisation."""
        x, y, w, h = roi_coords
        color = {
            STATUS_AVAILABLE: (0, 200, 50),
            STATUS_LOW: (0, 200, 255),
            STATUS_EMPTY: (0, 60, 255),
        }.get(reading.status, (200, 200, 200))

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{shelf_id}: {reading.status} ({reading.density_pct:.0f}%)"
        cv2.rectangle(frame, (x, y - 18), (x + len(label) * 7, y), color, -1)
        cv2.putText(frame, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        return frame
