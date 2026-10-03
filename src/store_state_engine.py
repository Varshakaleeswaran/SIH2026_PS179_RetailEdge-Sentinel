"""
RetailEdge Sentinel - Store State Engine
Synthesizes signals from Shopper, Shelf, and Queue Intelligence into a unified Store State.
States:
- NORMAL
- HIGH_TRAFFIC
- LOW_STOCK
- QUEUE_BUILDUP
- OPERATIONAL_STRESS
- CRITICAL_RETAIL_STATE
Uses transparent, interpretable, deterministic operational rules.
"""

from typing import Dict, Any

class StoreStateEngine:
    def __init__(self):
        pass

    def evaluate(self, shopper_signals: Dict[str, Any], shelf_signals: Dict[str, Any], queue_signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize retail intelligence signals into a comprehensive store status.
        """
        traffic = shopper_signals.get("traffic_level", "LOW")
        occupancy = shopper_signals.get("occupancy", 0)
        
        shelf_status = shelf_signals.get("shelf_status", "AVAILABLE")
        low_shelves = shelf_signals.get("low_shelves", 0)
        empty_shelves = shelf_signals.get("empty_shelves", 0)
        
        queue_level = queue_signals.get("queue_level", "LOW")
        queue_growth = queue_signals.get("growth_trend", "STABLE")
        queue_len = queue_signals.get("queue_length", 0)

        # Transparent rule hierarchy:
        # 1. Critical state: Multiple critical bottlenecks
        if (queue_level == "HIGH" and empty_shelves > 0) or (traffic == "HIGH" and queue_level == "HIGH" and low_shelves > 0):
            overall_state = "CRITICAL_RETAIL_STATE"
            summary = "Simultaneous critical queue surge and stock exhaustion under heavy traffic."

        # 2. Operational stress: High queue or compound pressure
        elif queue_level == "HIGH" and (traffic == "HIGH" or queue_growth == "INCREASING"):
            overall_state = "OPERATIONAL_STRESS"
            summary = "Checkout bottleneck compounding with growing store traffic."

        elif queue_level == "HIGH":
            overall_state = "QUEUE_BUILDUP"
            summary = "Billing counter queue exceeds normal throughput threshold."

        elif empty_shelves > 0 or low_shelves >= 2:
            overall_state = "LOW_STOCK"
            summary = "Multiple merchandise displays require immediate shelf replenishment."

        elif traffic == "HIGH":
            overall_state = "HIGH_TRAFFIC"
            summary = "Store footfall is elevated; floor monitoring recommended."

        elif low_shelves == 1 and queue_level == "MEDIUM":
            overall_state = "OPERATIONAL_STRESS"
            summary = "Moderate queue length alongside low inventory on active shelves."

        else:
            overall_state = "NORMAL"
            summary = "Store operations, footfall, and queue throughput are within balanced thresholds."

        return {
            "traffic": traffic,
            "occupancy": occupancy,
            "shelf_status": shelf_status,
            "low_shelves": low_shelves,
            "empty_shelves": empty_shelves,
            "queue_level": queue_level,
            "queue_length": queue_len,
            "queue_growth": queue_growth,
            "overall_state": overall_state,
            "summary": summary
        }
