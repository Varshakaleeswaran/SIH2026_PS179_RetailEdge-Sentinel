"""
RetailEdge Sentinel - Inventory & Shelf Intelligence Module
Prototype Shelf Availability Estimation:
- Tracks shelf zone activity and stock levels across defined shelves
- Status indicators: AVAILABLE, LOW, EMPTY
- Provides dynamic simulation of stock depletion based on customer dwell/browsing
- Transparently labeled as a prototype estimation model
"""

from typing import Dict, Any, List
import time

class InventoryIntelligence:
    def __init__(self, initial_shelves: List[Dict[str, Any]] = None):
        self.estimation_label = "Prototype Shelf Availability Estimation"
        self.last_update_time = time.time()
        
        # Initialize internal shelf states
        self.shelves: Dict[str, Dict[str, Any]] = {}
        if initial_shelves:
            for s in initial_shelves:
                sid = s.get("id", "SHELF_A")
                status = s.get("default_status", "AVAILABLE")
                fill = 85 if status == "AVAILABLE" else (30 if status == "LOW" else 5)
                self.shelves[sid] = {
                    "id": sid,
                    "name": s.get("name", sid),
                    "status": status,
                    "fill_percentage": fill,
                    "last_replenished": time.time()
                }
        else:
            self._init_default_shelves()

    def _init_default_shelves(self):
        self.shelves = {
            "SHELF_A": {
                "id": "SHELF_A",
                "name": "Fresh Produce / Beverages",
                "status": "AVAILABLE",
                "fill_percentage": 88,
                "last_replenished": time.time()
            },
            "SHELF_B": {
                "id": "SHELF_B",
                "name": "Snacks & Packaged Goods",
                "status": "LOW",
                "fill_percentage": 24,
                "last_replenished": time.time() - 3600
            },
            "SHELF_C": {
                "id": "SHELF_C",
                "name": "Personal Care & Household",
                "status": "AVAILABLE",
                "fill_percentage": 92,
                "last_replenished": time.time()
            }
        }

    def process(self, shelf_occupancy: int, simulated_depletion_speed: float = 0.05) -> Dict[str, Any]:
        """
        Process shelf inventory state:
        If shoppers are actively browsing the shelf zone, simulate gradual stock depletion.
        """
        now = time.time()
        elapsed = now - self.last_update_time
        self.last_update_time = now

        # If shelf occupancy > 0, slightly deplete the active shelf (Shelf B) to demonstrate dynamic response
        if shelf_occupancy > 0 and elapsed > 0:
            depletion = shelf_occupancy * simulated_depletion_speed * min(elapsed, 2.0)
            shelf_b = self.shelves.get("SHELF_B")
            if shelf_b and shelf_b["fill_percentage"] > 5:
                shelf_b["fill_percentage"] = max(0, round(shelf_b["fill_percentage"] - depletion, 1))
                if shelf_b["fill_percentage"] == 0:
                    shelf_b["status"] = "EMPTY"
                elif shelf_b["fill_percentage"] <= 35:
                    shelf_b["status"] = "LOW"

        # Count low and empty shelves
        low_count = sum(1 for s in self.shelves.values() if s["status"] == "LOW")
        empty_count = sum(1 for s in self.shelves.values() if s["status"] == "EMPTY")

        if empty_count > 0:
            overall_shelf_status = "CRITICAL_EMPTY"
        elif low_count > 0:
            overall_shelf_status = "LOW"
        else:
            overall_shelf_status = "AVAILABLE"

        replenishment_required = (low_count > 0 or empty_count > 0)

        return {
            "shelf_status": overall_shelf_status,
            "low_shelves": low_count,
            "empty_shelves": empty_count,
            "replenishment_required": replenishment_required,
            "shelves": {sid: dict(info) for sid, info in self.shelves.items()},
            "estimation_label": self.estimation_label
        }

    def set_shelf_status(self, shelf_id: str, new_status: str, fill_pct: int = 80):
        """Allows direct status adjustment for interactive demonstration."""
        if shelf_id in self.shelves:
            self.shelves[shelf_id]["status"] = new_status
            self.shelves[shelf_id]["fill_percentage"] = fill_pct
            if new_status == "AVAILABLE":
                self.shelves[shelf_id]["last_replenished"] = time.time()

    def replenish_shelf(self, shelf_id: str):
        """Trigger replenishment action."""
        self.set_shelf_status(shelf_id, "AVAILABLE", 95)
