"""
Unit tests for Action Recommendation Engine
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommendation_engine import RecommendationEngine

def test_high_queue_generates_additional_counter_recommendation():
    engine = RecommendationEngine()
    store_state = {"queue_length": 7, "queue_growth": "INCREASING", "traffic": "HIGH", "occupancy": 10}
    risk_analysis = {"risk_level": "HIGH"}
    shelf_signals = {"shelves": {}}

    recs = engine.generate_recommendations(store_state, risk_analysis, shelf_signals)
    assert len(recs) > 0
    assert any("billing counter" in r["action"].lower() for r in recs)
    assert recs[0]["priority"] == "HIGH"

def test_empty_shelf_generates_restock_recommendation():
    engine = RecommendationEngine()
    store_state = {"queue_length": 1, "queue_growth": "STABLE", "traffic": "LOW", "occupancy": 2}
    risk_analysis = {"risk_level": "LOW"}
    shelf_signals = {
        "shelves": {
            "SHELF_B": {"name": "Snacks & Packaged Goods", "status": "EMPTY", "fill_percentage": 0}
        }
    }

    recs = engine.generate_recommendations(store_state, risk_analysis, shelf_signals)
    assert any("replenish" in r["action"].lower() or "restock" in r["action"].lower() for r in recs)
    assert any(r["priority"] == "HIGH" for r in recs)

def test_nominal_state_generates_low_priority_routine():
    engine = RecommendationEngine()
    store_state = {"queue_length": 1, "queue_growth": "STABLE", "traffic": "LOW", "occupancy": 2}
    risk_analysis = {"risk_level": "LOW"}
    shelf_signals = {
        "shelves": {
            "SHELF_A": {"name": "Produce", "status": "AVAILABLE", "fill_percentage": 90}
        }
    }

    recs = engine.generate_recommendations(store_state, risk_analysis, shelf_signals)
    assert len(recs) == 1
    assert recs[0]["priority"] == "LOW"

if __name__ == "__main__":
    test_high_queue_generates_additional_counter_recommendation()
    test_empty_shelf_generates_restock_recommendation()
    test_nominal_state_generates_low_priority_routine()
    print("Recommendation Engine tests: PASSED")
