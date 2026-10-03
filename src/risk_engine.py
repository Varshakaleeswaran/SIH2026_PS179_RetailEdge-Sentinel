"""
RetailEdge Sentinel - Operational Risk & Priority Engine
Calculates transparent operational risk scores (0 to 100):
- Component weights: Queue (45%), Inventory (30%), Traffic (25%)
- Risk Levels: LOW (0-30), MEDIUM (31-55), HIGH (56-80), CRITICAL (81-100)
- Identifies exact top contributing factors for root-cause explainability
- Labeled as an operational prototype metric
"""

from typing import Dict, Any, List

class RiskEngine:
    def __init__(self, weights: Dict[str, float] = None, thresholds: Dict[str, int] = None):
        self.weights = weights or {"queue": 0.45, "inventory": 0.30, "traffic": 0.25}
        self.thresholds = thresholds or {"low": 30, "medium": 55, "high": 80}

    def compute_risk(self, shopper_signals: Dict[str, Any], shelf_signals: Dict[str, Any], queue_signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate composite risk score and explainable contributing drivers.
        """
        reasons: List[str] = []

        # 1. Queue risk component (0 - 100)
        q_len = queue_signals.get("queue_length", 0)
        q_growth = queue_signals.get("growth_trend", "STABLE")
        
        # Base queue score: 0 to 7+ maps to 0-100
        raw_q_score = min(100.0, (q_len / 7.0) * 80.0)
        if q_growth == "INCREASING":
            raw_q_score = min(100.0, raw_q_score + 20.0)
            if q_len >= 3:
                reasons.append(f"Queue length is {q_len} and actively increasing (+{queue_signals.get('queue_growth', 1)})")
        elif q_len >= 5:
            reasons.append(f"High queue length at checkout ({q_len} customers waiting)")

        # 2. Inventory risk component (0 - 100)
        empty_count = shelf_signals.get("empty_shelves", 0)
        low_count = shelf_signals.get("low_shelves", 0)
        
        raw_inv_score = 0.0
        if empty_count > 0:
            raw_inv_score += empty_count * 50.0
            reasons.append(f"{empty_count} shelf display is completely empty")
        if low_count > 0:
            raw_inv_score += low_count * 25.0
            reasons.append(f"{low_count} shelf display has low remaining stock")
        raw_inv_score = min(100.0, raw_inv_score)

        # 3. Traffic risk component (0 - 100)
        occ = shopper_signals.get("occupancy", 0)
        traffic_lvl = shopper_signals.get("traffic_level", "LOW")
        
        raw_traffic_score = min(100.0, (occ / 15.0) * 100.0)
        if traffic_lvl == "HIGH":
            reasons.append(f"Elevated store occupancy ({occ} shoppers active)")

        # Weighted composite score
        w_q = self.weights.get("queue", 0.45)
        w_inv = self.weights.get("inventory", 0.30)
        w_tr = self.weights.get("traffic", 0.25)

        total_score = round(w_q * raw_q_score + w_inv * raw_inv_score + w_tr * raw_traffic_score, 1)
        total_score = max(0.0, min(100.0, total_score))

        # Risk classification
        if total_score <= self.thresholds.get("low", 30):
            risk_level = "LOW"
            color = "#2ECC71" # Green
        elif total_score <= self.thresholds.get("medium", 55):
            risk_level = "MEDIUM"
            color = "#F39C12" # Amber
        elif total_score <= self.thresholds.get("high", 80):
            risk_level = "HIGH"
            color = "#E67E22" # Orange
        else:
            risk_level = "CRITICAL"
            color = "#E74C3C" # Red

        if not reasons:
            reasons.append("All operational indicators within nominal baseline thresholds.")

        return {
            "risk_score": total_score,
            "risk_level": risk_level,
            "color_hex": color,
            "components": {
                "queue_score": round(raw_q_score, 1),
                "inventory_score": round(raw_inv_score, 1),
                "traffic_score": round(raw_traffic_score, 1)
            },
            "reasons": reasons,
            "disclaimer": "Operational Prototype Score (Rule-based, not validated statistical prediction)"
        }
