"""
Patent Candidate C: Predictive Queue Congestion Index (PQCI)
-------------------------------------------------------------------------------
Temporal Flow Imbalance Factor (TFIF) Engine — SIH26179 / RetailEdge Sentinel

PQCI quantifies the imbalance between the rate shoppers enter the store and
the rate they are being served at the billing counter.

    TFIF = Flow_in  /  Flow_billing

Where:
    Flow_in      = shoppers entering store per minute (sliding window)
    Flow_billing = shoppers completing checkout per minute (sliding window)

Alert levels:
    TFIF < 1.5          → BALANCED   — no action required
    1.5 ≤ TFIF < 2.5    → BUILDING   — monitor closely
    TFIF ≥ 2.5 (90 s+)  → CONGESTION — open additional counter
    TFIF ≥ 3.5          → SURGE      — immediate intervention
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Configurable thresholds
TFIF_BALANCED_MAX = 1.5
TFIF_BUILDING_MAX = 2.5
TFIF_SURGE_MIN = 3.5
SUSTAINED_ALERT_SEC = 90.0          # seconds TFIF must exceed threshold for escalation
WINDOW_SEC = 60.0                   # sliding window for rate calculation


@dataclass
class FlowEvent:
    timestamp: float
    delta: int          # number of people in this event


@dataclass
class PQCIResult:
    tfif: float
    pqci_state: str     # BALANCED | BUILDING | CONGESTION | SURGE
    inflow_rate: float  # shoppers per minute in window
    billing_rate: float # shoppers per minute in window
    congestion_predicted: bool
    eta_minutes: Optional[float]
    action: str
    confidence: float
    queue_count: int
    wait_time_minutes: float


class PredictiveQueueEngine:
    """
    Candidate Patent C: Predictive Queue Congestion Index (PQCI).

    Usage
    -----
        engine = PredictiveQueueEngine(window_sec=60)

        # Each time a shopper is detected crossing the entrance:
        engine.record_entry(count=1)

        # Each time a checkout transaction completes:
        engine.record_billing(count=1)

        result = engine.compute_pqci(current_queue_length=4)
        print(result.pqci_state, result.tfif)
    """

    def __init__(
        self,
        window_sec: float = WINDOW_SEC,
        congestion_threshold: float = TFIF_BUILDING_MAX,
        surge_threshold: float = TFIF_SURGE_MIN,
        sustained_sec: float = SUSTAINED_ALERT_SEC,
    ):
        self.window = window_sec
        self.congestion_threshold = congestion_threshold
        self.surge_threshold = surge_threshold
        self.sustained_sec = sustained_sec

        self._inflow: Deque[FlowEvent] = deque()
        self._billing: Deque[FlowEvent] = deque()

        self._alert_since: Optional[float] = None   # when TFIF first breached threshold
        self._last_result: Optional[PQCIResult] = None
        self._history: List[Tuple[float, float]] = []   # (timestamp, tfif)

    # ------------------------------------------------------------------
    # Ingestion helpers
    # ------------------------------------------------------------------

    def record_entry(self, count: int = 1):
        """Call when `count` shoppers are detected entering the store."""
        self._inflow.append(FlowEvent(timestamp=time.time(), delta=count))
        self._prune()

    def record_billing(self, count: int = 1):
        """Call when `count` shoppers complete checkout (leave billing zone)."""
        self._billing.append(FlowEvent(timestamp=time.time(), delta=count))
        self._prune()

    def update_rates(self, entries_delta: int, billed_delta: int):
        """Convenience: update both inflow and billing together."""
        if entries_delta > 0:
            self.record_entry(entries_delta)
        if billed_delta > 0:
            self.record_billing(billed_delta)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute_pqci(
        self,
        current_queue_length: int = 0,
        avg_service_time_sec: float = 120.0,
    ) -> PQCIResult:
        """
        Compute the current PQCI state.

        Parameters
        ----------
        current_queue_length : int
            Number of people currently waiting at billing.
        avg_service_time_sec : float
            Average seconds per checkout transaction (used for wait estimate).

        Returns
        -------
        PQCIResult
        """
        now = time.time()
        self._prune(now)

        # Sum flows in the sliding window
        in_sum = max(1, sum(e.delta for e in self._inflow))
        bill_sum = max(1, sum(e.delta for e in self._billing))

        # Convert to per-minute rates for readability
        window_minutes = self.window / 60.0
        inflow_rate = round(in_sum / window_minutes, 2)
        billing_rate = round(bill_sum / window_minutes, 2)

        tfif = round(in_sum / bill_sum, 3)

        # Determine state
        if tfif >= self.surge_threshold:
            state = "SURGE"
        elif tfif >= self.congestion_threshold:
            state = "CONGESTION"
        elif tfif >= TFIF_BALANCED_MAX:
            state = "BUILDING"
        else:
            state = "BALANCED"

        # Track how long we've been in alert state
        if state in ("CONGESTION", "SURGE"):
            if self._alert_since is None:
                self._alert_since = now
        else:
            self._alert_since = None

        sustained_duration = (now - self._alert_since) if self._alert_since else 0.0
        congestion_predicted = (
            state in ("CONGESTION", "SURGE")
            and sustained_duration >= self.sustained_sec
        ) or state == "SURGE"

        # Estimate time until queue saturation
        eta_minutes: Optional[float] = None
        if state in ("BUILDING", "CONGESTION", "SURGE"):
            # Simple linear model: queue grows at (inflow - billing) ppm
            net_growth_ppm = inflow_rate - billing_rate
            if net_growth_ppm > 0:
                # Capacity threshold: arbitrary 15 people
                capacity_gap = max(0, 15 - current_queue_length)
                eta_minutes = round(capacity_gap / net_growth_ppm, 1) if net_growth_ppm else None

        # Wait time estimate for current queue
        wait_time_minutes = round(
            (current_queue_length * avg_service_time_sec) / 60.0, 1
        )

        # Action
        if state == "SURGE":
            action = "OPEN_COUNTER_IMMEDIATELY"
        elif congestion_predicted:
            action = "OPEN_COUNTER_2"
        elif state == "BUILDING":
            action = "PREPARE_COUNTER_2"
        else:
            action = "STANDBY"

        # Confidence (higher TFIF → higher confidence alert is real)
        confidence = min(99.0, round(50.0 + (tfif - 1.0) * 15.0, 1))

        result = PQCIResult(
            tfif=tfif,
            pqci_state=state,
            inflow_rate=inflow_rate,
            billing_rate=billing_rate,
            congestion_predicted=congestion_predicted,
            eta_minutes=eta_minutes,
            action=action,
            confidence=confidence,
            queue_count=current_queue_length,
            wait_time_minutes=wait_time_minutes,
        )
        self._last_result = result
        self._history.append((now, tfif))
        if len(self._history) > 500:
            self._history = self._history[-500:]

        return result

    # ------------------------------------------------------------------
    # Telemetry / serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self, result: Optional[PQCIResult] = None) -> Dict:
        r = result or self._last_result
        if r is None:
            return {"tfif": 1.0, "pqci_state": "BALANCED", "action": "STANDBY"}
        return {
            "tfif": r.tfif,
            "pqci_state": r.pqci_state,
            "inflow_rate_ppm": r.inflow_rate,
            "billing_rate_ppm": r.billing_rate,
            "congestion_predicted": r.congestion_predicted,
            "eta_minutes": r.eta_minutes,
            "action": r.action,
            "confidence_pct": r.confidence,
            "queue_count": r.queue_count,
            "wait_time_minutes": r.wait_time_minutes,
        }

    def get_tfif_history(self, last_n: int = 30) -> List[Dict]:
        return [
            {"t": round(ts, 2), "tfif": round(tfif, 3)}
            for ts, tfif in self._history[-last_n:]
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune(self, now: Optional[float] = None):
        now = now or time.time()
        cutoff = now - self.window
        while self._inflow and self._inflow[0].timestamp < cutoff:
            self._inflow.popleft()
        while self._billing and self._billing[0].timestamp < cutoff:
            self._billing.popleft()
