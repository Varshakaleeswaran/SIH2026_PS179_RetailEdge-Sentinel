"""
RetailEdge Sentinel - Queue Intelligence Module
Monitors the billing/checkout counter zone:
- Queue length calculation
- Growth detection across a sliding historical window
- Waiting-time trend estimation (queue_length / service_rate)
- Queue severity classification (LOW, MEDIUM, HIGH)
- Transparent prototype estimation disclosure
"""

from typing import Dict, Any, List
import numpy as np

class QueueIntelligence:
    def __init__(self, thresholds: Dict[str, int] = None, default_service_rate: float = 1.5, history_window: int = 40):
        self.thresholds = thresholds or {"low": 2, "medium": 5, "high": 8}
        self.default_service_rate = default_service_rate # customers served per minute
        self.history_window = history_window
        self.queue_history: List[int] = []

    def process(self, billing_occupancy: int) -> Dict[str, Any]:
        """
        Evaluate queue metrics given the current billing zone count.
        """
        self.queue_history.append(billing_occupancy)
        if len(self.queue_history) > self.history_window:
            self.queue_history.pop(0)

        queue_len = billing_occupancy

        # Determine queue severity level
        if queue_len <= self.thresholds.get("low", 2):
            queue_level = "LOW"
        elif queue_len <= self.thresholds.get("medium", 5):
            queue_level = "MEDIUM"
        else:
            queue_level = "HIGH"

        # Calculate queue growth rate
        if len(self.queue_history) >= 10:
            past_len = self.queue_history[0]
            growth = queue_len - past_len
        else:
            growth = 0

        if growth > 0:
            growth_trend = "INCREASING"
        elif growth < 0:
            growth_trend = "DECREASING"
        else:
            growth_trend = "STABLE"

        # Estimated waiting time (minutes): queue_length / service_rate
        # Example: 7 people / 1.5 people/min = 4.7 minutes
        if self.default_service_rate > 0 and queue_len > 0:
            est_wait_minutes = round(queue_len / self.default_service_rate, 1)
        else:
            est_wait_minutes = 0.0

        if queue_level == "HIGH" or growth > 1:
            wait_trend = "INCREASING"
        elif growth < -1:
            wait_trend = "DECREASING"
        else:
            wait_trend = "STABLE"

        return {
            "queue_length": queue_len,
            "queue_level": queue_level,
            "queue_growth": growth,
            "growth_trend": growth_trend,
            "estimated_wait_minutes": est_wait_minutes,
            "waiting_time_trend": wait_trend,
            "service_rate": self.default_service_rate,
            "disclaimer": "Prototype Estimation (queue_length / service_rate)"
        }
