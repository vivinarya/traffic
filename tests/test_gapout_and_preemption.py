"""
Unit tests for Green Gap-Out, Emergency Vehicle Preemption, and Lane Calibration
"""
import sys
import time
from pathlib import Path

# Ensure workspace root in path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.perception.lane_geometry import Lane, IntersectionGeometry
from detection.controller.adaptive_controller import AdaptiveTrafficSignalController, SignalPhase
from detection.controller.traffic_state import TrafficStateSnapshot, ApproachState

def test_lane_geometry_serialization():
    geom = IntersectionGeometry.create_standard_4way(1280, 720, px_per_m=12.0)
    test_json = ROOT / "tests" / "temp_calib.json"
    geom.save_to_json(str(test_json))

    assert test_json.exists(), "Calibration JSON file was not created"
    loaded = IntersectionGeometry.load_from_json(str(test_json))
    assert len(loaded.lanes) == 4, f"Expected 4 lanes, got {len(loaded.lanes)}"
    assert loaded.pixels_per_meter == 12.0

    # Clean up
    if test_json.exists():
        test_json.unlink()
    print("[PASS] Lane geometry serialization & deserialization")

def test_stopline_crossing_detection():
    # Stop line from (100, 200) to (200, 200)
    poly = [(100, 100), (200, 100), (200, 300), (100, 300)]
    stopline = ((100, 200), (200, 200))
    lane = Lane("Test_L1", poly, stopline, direction="Southbound", approach="North")

    # Vehicle crossing from (150, 180) to (150, 220) -> crosses line y=200
    crossed = lane.has_crossed_stopline((150.0, 180.0), (150.0, 220.0))
    assert crossed is True, "Vehicle crossing stop line was not detected"

    # Vehicle moving within lane but NOT crossing line -> (150, 150) to (150, 180)
    not_crossed = lane.has_crossed_stopline((150.0, 150.0), (150.0, 180.0))
    assert not_crossed is False, "False positive stop line crossing detected"
    print("[PASS] Stop line crossing detection")

def test_emergency_vehicle_preemption():
    controller = AdaptiveTrafficSignalController(
        min_green_sec=10,
        max_green_sec=40,
        yellow_sec=3,
        all_red_sec=2
    )
    # Controller starts in NS_GREEN
    assert controller.current_phase == SignalPhase.PHASE_NS_GREEN

    # Trigger emergency preemption for EW corridor
    controller.trigger_emergency_preemption(direction="EW", hold_duration_sec=10.0)
    assert controller.emergency_preemption_active is True
    assert controller.emergency_corridor == "EW"

    # Dummy snapshot
    apps = {
        "North": ApproachState("North", total_pcu=0.0, current_signal="RED"),
        "South": ApproachState("South", total_pcu=0.0, current_signal="RED"),
        "East": ApproachState("East", total_pcu=2.0, waiting_time_sec=15.0, current_signal="RED"),
        "West": ApproachState("West", total_pcu=1.0, waiting_time_sec=10.0, current_signal="RED")
    }
    snapshot = TrafficStateSnapshot(timestamp=time.time(), approaches=apps, total_intersection_pcu=3.0)

    # First decision should switch active NS_GREEN to NS_YELLOW for rapid transition
    decision = controller.compute_decision(snapshot, {})
    assert decision.is_emergency_preemption is True
    assert decision.emergency_corridor == "EW"
    assert decision.active_phase == SignalPhase.PHASE_NS_YELLOW
    print("[PASS] Emergency vehicle preemption rapid transition")

def test_green_gapout_logic():
    controller = AdaptiveTrafficSignalController(
        min_green_sec=10,
        max_green_sec=40,
        yellow_sec=3,
        all_red_sec=2,
        gap_threshold_sec=2.0,
        min_green_before_gapout=5.0
    )
    # Set phase start time to 6 seconds ago so min_green_before_gapout has elapsed
    now = time.time()
    controller.phase_start_time = now - 6.0
    # Last vehicle departure was 3.0 seconds ago (headway 3.0s > threshold 2.0s)
    controller.last_departure_time["NS"] = now - 3.0

    # Snapshot with competing traffic waiting on East/West (3.0 PCU)
    apps = {
        "North": ApproachState("North", total_pcu=0.0, current_signal="GREEN"),
        "South": ApproachState("South", total_pcu=0.0, current_signal="GREEN"),
        "East": ApproachState("East", total_pcu=2.0, waiting_time_sec=15.0, current_signal="RED"),
        "West": ApproachState("West", total_pcu=1.0, waiting_time_sec=10.0, current_signal="RED")
    }
    snapshot = TrafficStateSnapshot(timestamp=now, approaches=apps, total_intersection_pcu=3.0)

    # Controller should trigger GAP-OUT and transition to NS_YELLOW early!
    decision = controller.compute_decision(snapshot, {})
    assert decision.is_gapout is True, "Gap-out was not triggered"
    assert decision.active_phase == SignalPhase.PHASE_NS_YELLOW, f"Expected NS_YELLOW, got {decision.active_phase}"
    print("[PASS] Green Gap-Out early clearance")

if __name__ == "__main__":
    test_lane_geometry_serialization()
    test_stopline_crossing_detection()
    test_emergency_vehicle_preemption()
    test_green_gapout_logic()
    print("\nALL TESTS PASSED!")
