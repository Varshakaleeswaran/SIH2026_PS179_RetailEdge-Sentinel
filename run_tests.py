"""
RetailEdge Sentinel - Unified Test Runner
Runs all unit and integration tests across intelligence modules, risk, store state, recommendations, tracker, zones, and pipeline.
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def run_all():
    print("==================================================")
    print("       RetailEdge Sentinel - Test Suite           ")
    print("==================================================")

    # 1. Store State Engine
    from tests.test_store_state import (
        test_nominal_state,
        test_high_traffic_generates_high_traffic_state,
        test_high_queue_generates_operational_stress,
        test_empty_shelf_generates_low_stock_state,
        test_compound_bottleneck_generates_critical_retail_state
    )
    test_nominal_state()
    test_high_traffic_generates_high_traffic_state()
    test_high_queue_generates_operational_stress()
    test_empty_shelf_generates_low_stock_state()
    test_compound_bottleneck_generates_critical_retail_state()
    print("[PASS] 1. Store State Engine (5/5 tests)")

    # 2. Risk Engine
    from tests.test_risk import (
        test_baseline_risk_is_low,
        test_high_queue_escalates_risk,
        test_empty_shelf_increases_risk,
        test_risk_score_is_bounded
    )
    test_baseline_risk_is_low()
    test_high_queue_escalates_risk()
    test_empty_shelf_increases_risk()
    test_risk_score_is_bounded()
    print("[PASS] 2. Operational Risk & Priority Engine (4/4 tests)")

    # 3. Recommendation Engine
    from tests.test_recommendations import (
        test_high_queue_generates_additional_counter_recommendation,
        test_empty_shelf_generates_restock_recommendation,
        test_nominal_state_generates_low_priority_routine
    )
    test_high_queue_generates_additional_counter_recommendation()
    test_empty_shelf_generates_restock_recommendation()
    test_nominal_state_generates_low_priority_routine()
    print("[PASS] 3. Action Recommendation Engine (3/3 tests)")

    # 4. Pipeline, Tracker, Zones & Privacy
    from tests.test_pipeline_e2e import (
        test_anonymous_tracker_assigns_non_identifying_ids,
        test_zone_point_in_polygon,
        test_privacy_guard_enforces_rules,
        test_full_pipeline_processes_synthetic_frame
    )
    test_anonymous_tracker_assigns_non_identifying_ids()
    test_zone_point_in_polygon()
    test_privacy_guard_enforces_rules()
    test_full_pipeline_processes_synthetic_frame()
    print("[PASS] 4. Tracker, Zones, Privacy & Pipeline (4/4 tests)")

    print("==================================================")
    print("       ALL 16 TESTS PASSED SUCCESSFULLY!          ")
    print("==================================================")

if __name__ == "__main__":
    run_all()
