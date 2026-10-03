"""
Unit tests for Operational Risk & Priority Engine
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.risk_engine import RiskEngine

def test_baseline_risk_is_low():
    engine = RiskEngine()
    shopper = {"occupancy": 2, "traffic_level": "LOW"}
    shelf = {"empty_shelves": 0, "low_shelves": 0}
    queue = {"queue_length": 1, "growth_trend": "STABLE"}

    risk = engine.compute_risk(shopper, shelf, queue)
    assert risk["risk_score"] < 30.0
    assert risk["risk_level"] == "LOW"

def test_high_queue_escalates_risk():
    engine = RiskEngine()
    shopper = {"occupancy": 8, "traffic_level": "MEDIUM"}
    shelf = {"empty_shelves": 0, "low_shelves": 0}
    queue = {"queue_length": 7, "growth_trend": "INCREASING", "queue_growth": 4}

    risk = engine.compute_risk(shopper, shelf, queue)
    assert risk["risk_score"] > 55.0
    assert risk["risk_level"] in ["HIGH", "CRITICAL"]
    assert any("Queue" in r for r in risk["reasons"])

def test_empty_shelf_increases_risk():
    engine = RiskEngine()
    shopper = {"occupancy": 3, "traffic_level": "LOW"}
    shelf = {"empty_shelves": 1, "low_shelves": 1}
    queue = {"queue_length": 1, "growth_trend": "STABLE"}

    risk = engine.compute_risk(shopper, shelf, queue)
    assert risk["components"]["inventory_score"] >= 50.0

def test_risk_score_is_bounded():
    engine = RiskEngine()
    # Extreme conditions
    shopper = {"occupancy": 100, "traffic_level": "HIGH"}
    shelf = {"empty_shelves": 5, "low_shelves": 5}
    queue = {"queue_length": 50, "growth_trend": "INCREASING"}

    risk = engine.compute_risk(shopper, shelf, queue)
    assert 0.0 <= risk["risk_score"] <= 100.0
    assert risk["risk_level"] == "CRITICAL"

if __name__ == "__main__":
    test_baseline_risk_is_low()
    test_high_queue_escalates_risk()
    test_empty_shelf_increases_risk()
    test_risk_score_is_bounded()
    print("Risk Engine tests: PASSED")
