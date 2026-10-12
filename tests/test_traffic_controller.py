"""
Unit & Integration Tests for Traffic Light Vehicle Density Detector
Tests:
1. Traffic State S(t) Aggregation and PCU assignment
2. Short-term EWMA Traffic Prediction
3. Adaptive Controller Phase Transitions & Dynamic Green Calculation
4. End-to-end master app execution on real traffic footage
"""
import sys
import time
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from detection.controller.traffic_state import TrafficStateManager, ApproachState
from detection.controller.short_term_predictor import ShortTermTrafficPredictor
from detection.controller.adaptive_controller import AdaptiveTrafficSignalController, SignalPhase
from traffic_light_detector_app import TrafficLightVehicleDensityDetectorApp

class MockTrack:
    def __init__(self, tid, pcu, spd, bbox, lane_id):
        self.track_id = tid
        self.pcu_weight = pcu
        self.speed_kmh = spd
        self.last_bbox = bbox
        self.lane_id = lane_id

def test_traffic_state_and_prediction():
    print("\n[TEST 1] Testing Traffic State S(t) and Short-Term Prediction...")
    mgr = TrafficStateManager(approach_ids=["North", "South", "East", "West"])
    predictor = ShortTermTrafficPredictor(alpha=0.4, forecast_horizon_sec=10.0)

    # Simulate 3 cars on North approach and 1 bus on East approach
    tracks = [
        MockTrack(1, 1.0, 0.0, [600, 100, 650, 180], "North_L1"), # Queued car
        MockTrack(2, 1.0, 0.0, [600, 200, 650, 280], "North_L1"), # Queued car
        MockTrack(3, 1.0, 15.0, [600, 300, 650, 360], "North_L1"),# Moving car
        MockTrack(4, 3.0, 0.0, [1000, 300, 1150, 400], "East_L1"), # Queued bus
    ]

    signals = {"North": "RED", "South": "RED", "East": "RED", "West": "RED"}
    now = time.time()
    snapshot = mgr.update_from_tracks(tracks, now, signals, px_per_m=12.0)

    assert snapshot.approaches["North"].vehicle_count == 3
    assert snapshot.approaches["North"].total_pcu == 3.0
    assert snapshot.approaches["North"].queued_vehicles == 2
    assert snapshot.approaches["North"].queue_length_m > 0
    assert snapshot.approaches["East"].total_pcu == 3.0
    print(f"  ✓ S(t) Snapshot: North PCU={snapshot.approaches['North'].total_pcu}, Q={snapshot.approaches['North'].queue_length_m}m")

    # Predict short-term arrival
    preds = predictor.predict(snapshot)
    assert "North" in preds
    assert "predicted_pcu" in preds["North"]
    print(f"  ✓ Prediction: North Predicted PCU={preds['North']['predicted_pcu']}, Rate={preds['North']['arrival_rate_pcu_sec']} PCU/s")

def test_adaptive_controller_phases():
    print("\n[TEST 2] Testing Adaptive Controller State Machine...")
    controller = AdaptiveTrafficSignalController(
        min_green_sec=10,
        max_green_sec=30,
        yellow_sec=3,
        all_red_sec=2
    )

    mgr = TrafficStateManager()
    predictor = ShortTermTrafficPredictor()
    now = time.time()
    snap = mgr.update_from_tracks([], now, {"North": "GREEN", "South": "GREEN", "East": "RED", "West": "RED"})
    preds = predictor.predict(snap)

    # Initial state must be NS GREEN
    dec1 = controller.compute_decision(snap, preds)
    assert dec1.active_phase == SignalPhase.PHASE_NS_GREEN
    assert dec1.signals["North"] == "GREEN"
    assert dec1.signals["East"] == "RED"
    print(f"  ✓ Initial Phase: {dec1.active_phase.value} (Countdown: {dec1.countdown_seconds}s)")

    # Force expiration of green phase
    controller.phase_start_time = now - (controller.current_allocated_duration + 1.0)
    dec2 = controller.compute_decision(snap, preds)
    assert dec2.active_phase == SignalPhase.PHASE_NS_YELLOW
    assert dec2.signals["North"] == "YELLOW"
    print(f"  ✓ Transition to Yellow: {dec2.active_phase.value} (Timer: {dec2.countdown_seconds}s)")

    # Force expiration of yellow phase
    controller.phase_start_time = now - (controller.current_allocated_duration + 1.0)
    dec3 = controller.compute_decision(snap, preds)
    assert dec3.active_phase == SignalPhase.PHASE_ALL_RED
    assert dec3.signals["North"] == "RED" and dec3.signals["East"] == "RED"
    print(f"  ✓ Transition to All-Red: {dec3.active_phase.value}")

    # Force expiration of all-red phase -> EW GREEN
    controller.phase_start_time = now - (controller.current_allocated_duration + 1.0)
    dec4 = controller.compute_decision(snap, preds)
    assert dec4.active_phase == SignalPhase.PHASE_EW_GREEN
    assert dec4.signals["East"] == "GREEN"
    assert controller.min_green <= dec4.allocated_green_seconds <= controller.max_green
    print(f"  ✓ Dynamic Green Decision for EW: {dec4.allocated_green_seconds}s (Clamped [{controller.min_green}, {controller.max_green}])")

def test_full_pipeline_on_real_video():
    print("\n[TEST 3] Running Master Application on Real Traffic Video Clip...")
    out_video = Path(r"c:\g\tests\annotated_density_detector.mp4")
    if out_video.exists():
        out_video.unlink()

    app = TrafficLightVehicleDensityDetectorApp(
        model_path=r"c:\g\weights\edgeflow_yolov8n.pt",
        conf_thresh=0.18,
        device="0"
    )

    summary = app.run(
        video_source=r"c:\g\tests\real_traffic_video.mp4",
        output_path=str(out_video),
        max_frames=60,
        show_gui=False
    )

    assert summary["frames_processed"] == 60
    assert out_video.exists() and out_video.stat().st_size > 100000
    print(f"  ✓ Successfully processed {summary['frames_processed']} frames @ {summary['average_fps']} FPS")
    print(f"  ✓ Output video rendered: {out_video} ({out_video.stat().st_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    test_traffic_state_and_prediction()
    test_adaptive_controller_phases()
    test_full_pipeline_on_real_video()
    print("\n" + "=" * 60)
    print("   ALL TESTS PASSED SUCCESSFULLY!   ")
    print("=" * 60)
