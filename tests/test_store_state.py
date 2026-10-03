"""
Unit tests for Store State Engine
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.store_state_engine import StoreStateEngine

def test_nominal_state():
    engine = StoreStateEngine()
    shopper = {"traffic_level": "LOW", "occupancy": 3}
    shelf = {"shelf_status": "AVAILABLE", "low_shelves": 0, "empty_shelves": 0}
    queue = {"queue_level": "LOW", "growth_trend": "STABLE", "queue_length": 1}

    result = engine.evaluate(shopper, shelf, queue)
    assert result["overall_state"] == "NORMAL"

def test_high_traffic_generates_high_traffic_state():
    engine = StoreStateEngine()
    shopper = {"traffic_level": "HIGH", "occupancy": 12}
    shelf = {"shelf_status": "AVAILABLE", "low_shelves": 0, "empty_shelves": 0}
    queue = {"queue_level": "LOW", "growth_trend": "STABLE", "queue_length": 1}

    result = engine.evaluate(shopper, shelf, queue)
    assert result["overall_state"] == "HIGH_TRAFFIC"

def test_high_queue_generates_operational_stress():
    engine = StoreStateEngine()
    shopper = {"traffic_level": "HIGH", "occupancy": 10}
    shelf = {"shelf_status": "AVAILABLE", "low_shelves": 0, "empty_shelves": 0}
    queue = {"queue_level": "HIGH", "growth_trend": "INCREASING", "queue_length": 7}

    result = engine.evaluate(shopper, shelf, queue)
    assert result["overall_state"] == "OPERATIONAL_STRESS"

def test_empty_shelf_generates_low_stock_state():
    engine = StoreStateEngine()
    shopper = {"traffic_level": "LOW", "occupancy": 2}
    shelf = {"shelf_status": "CRITICAL_EMPTY", "low_shelves": 0, "empty_shelves": 1}
    queue = {"queue_level": "LOW", "growth_trend": "STABLE", "queue_length": 1}

    result = engine.evaluate(shopper, shelf, queue)
    assert result["overall_state"] == "LOW_STOCK"

def test_compound_bottleneck_generates_critical_retail_state():
    engine = StoreStateEngine()
    shopper = {"traffic_level": "HIGH", "occupancy": 15}
    shelf = {"shelf_status": "CRITICAL_EMPTY", "low_shelves": 1, "empty_shelves": 1}
    queue = {"queue_level": "HIGH", "growth_trend": "INCREASING", "queue_length": 8}

    result = engine.evaluate(shopper, shelf, queue)
    assert result["overall_state"] == "CRITICAL_RETAIL_STATE"

if __name__ == "__main__":
    test_nominal_state()
    test_high_traffic_generates_high_traffic_state()
    test_high_queue_generates_operational_stress()
    test_empty_shelf_generates_low_stock_state()
    test_compound_bottleneck_generates_critical_retail_state()
    print("Store State Engine tests: PASSED")
