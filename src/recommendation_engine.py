"""
RetailEdge Sentinel - Action Recommendation Engine
Generates clear, prioritised, actionable instructions for retail store associates:
- Priorities: HIGH, MEDIUM, LOW
- Detailed structure: Priority, Issue, Recommended Action, and Underlying Reason
"""

from typing import Dict, Any, List

class RecommendationEngine:
    def __init__(self):
        pass

    def generate_recommendations(self, store_state: Dict[str, Any], risk_analysis: Dict[str, Any], shelf_signals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate contextual operational recommendations based on store condition and risk.
        """
        recommendations: List[Dict[str, Any]] = []

        q_len = store_state.get("queue_length", 0)
        q_growth = store_state.get("queue_growth", "STABLE")
        traffic = store_state.get("traffic", "LOW")
        occupancy = store_state.get("occupancy", 0)
        shelves = shelf_signals.get("shelves", {})

        # 1. Critical/High Queue buildup recommendations
        if q_len >= 5 or (q_len >= 3 and q_growth == "INCREASING"):
            recommendations.append({
                "priority": "HIGH",
                "badge_color": "#E74C3C",
                "issue": "Checkout bottleneck and queue buildup detected.",
                "action": "Open an additional billing counter (Counter 2) immediately.",
                "reason": f"Queue length is {q_len} customers with an {q_growth} growth trend."
            })
        elif q_len >= 3:
            recommendations.append({
                "priority": "MEDIUM",
                "badge_color": "#F39C12",
                "issue": "Moderate checkout line detected.",
                "action": "Alert backup cashier to stand by near checkout station.",
                "reason": f"Queue length has reached {q_len} customers."
            })

        # 2. Inventory / Shelf Replenishment recommendations
        for sid, sinfo in shelves.items():
            s_name = sinfo.get("name", sid)
            s_status = sinfo.get("status", "AVAILABLE")
            s_fill = sinfo.get("fill_percentage", 100)

            if s_status == "EMPTY":
                recommendations.append({
                    "priority": "HIGH",
                    "badge_color": "#E74C3C",
                    "issue": f"Merchandise stock exhausted at {s_name}.",
                    "action": f"Trigger immediate shelf replenishment for {s_name} from backroom stock.",
                    "reason": f"Display level is 0% (EMPTY) causing potential lost retail sales."
                })
            elif s_status == "LOW":
                recommendations.append({
                    "priority": "MEDIUM",
                    "badge_color": "#F39C12",
                    "issue": f"Stock depleted below threshold at {s_name}.",
                    "action": f"Assign floor associate to restock {s_name}.",
                    "reason": f"Display level is currently {s_fill}% with customer browsing traffic."
                })

        # 3. High Traffic / Floor Staffing recommendations
        if traffic == "HIGH" or occupancy >= 10:
            recommendations.append({
                "priority": "MEDIUM",
                "badge_color": "#F39C12",
                "issue": "High shopper traffic concentration on sales floor.",
                "action": "Redeploy available staff toward main aisles for customer assistance.",
                "reason": f"Store occupancy is currently {occupancy} active shoppers."
            })

        # 4. Nominal state fallback
        if not recommendations:
            recommendations.append({
                "priority": "LOW",
                "badge_color": "#2ECC71",
                "issue": "Store operating within nominal parameters.",
                "action": "Maintain scheduled routine floor assistance and monitor displays.",
                "reason": "Queue length is low, footfall is balanced, and shelves are well stocked."
            })

        return recommendations
