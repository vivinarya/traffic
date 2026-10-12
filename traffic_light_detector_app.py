"""
=============================================================================
TRAFFIC LIGHT VEHICLE DENSITY DETECTOR - MASTER APPLICATION
=============================================================================
Closed-loop adaptive traffic signal control system driven by computer vision:

                  VIDEO / WEBCAM
                       ↓
                  YOLOv8n
                       ↓
                Vehicle Tracking
                       ↓
              Vehicle Count / Queue
                       ↓
             Traffic State S(t)
                       ↓
              Short-term Prediction
                       ↓
             Adaptive Controller
                       ↓
              GREEN TIME DECISION
                       ↓
             ┌─────────────────┐
             │ Software Traffic│ <---> Arduino Hardware (USB Serial)
             │ Signal Simulator │
             └─────────────────┘
                       ↓
              RED / YELLOW / GREEN
=============================================================================
"""
import sys
import os
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
import cv2
import numpy as np

# Ensure root is in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ultralytics import YOLO

from detection.perception.detector import PCU_WEIGHTS, TRAFFIC_NAME_CANONICAL
from detection.perception.tracker import ByteTrackManager
from detection.perception.lane_geometry import IntersectionGeometry
from detection.perception.kinematics import KinematicsEstimator
from detection.controller.traffic_state import TrafficStateManager, TrafficStateSnapshot
from detection.controller.short_term_predictor import ShortTermTrafficPredictor
from detection.controller.adaptive_controller import AdaptiveTrafficSignalController, ControllerDecision, SignalPhase
from detection.controller.signal_simulator import SoftwareTrafficSignalSimulator
from detection.controller.arduino_bridge import ArduinoTrafficLightBridge
from calibrate_lanes import InteractiveLaneCalibrator

# Visual Styling
COLOR_BBOX = (245, 245, 245)
COLOR_TRAJECTORY = (180, 200, 255) # Peach / Light Salmon
COLOR_MOTION_VEC = (0, 0, 235)     # Bright Red

class TrafficLightVehicleDensityDetectorApp:
    def __init__(
        self,
        model_path: str = r"c:\g\weights\detection_yolov8n.pt",
        conf_thresh: float = 0.18,
        arduino_port: Optional[str] = None,
        device: str = "cpu",
        calibration_file: Optional[str] = None
    ):
        print("=" * 70)
        print("   TRAFFIC LIGHT VEHICLE DENSITY DETECTOR - INITIALIZING   ")
        print("=" * 70)

        self.conf_thresh = conf_thresh
        self.device = device
        self.calibration_file = calibration_file or r"c:\g\lane_calibration.json"

        # 1. Load YOLO model
        if not Path(model_path).exists():
            if Path(r"c:\g\weights\edgeflow_yolov8n.pt").exists():
                model_path = r"c:\g\weights\edgeflow_yolov8n.pt"
            else:
                print(f"[Warning] Weights '{model_path}' not found. Falling back to 'yolov8n.pt'...")
                model_path = "yolov8n.pt"

        print(f"[Detector] Loading YOLOv8 architecture: {model_path} ({device})...")
        self.model = YOLO(model_path)

        # Check domain from class names
        self.model_names = getattr(self.model, "names", {})
        self.is_visdrone = len(self.model_names) == 10 and any(k in ["van", "awning-tricycle", "tricycle"] for k in self.model_names.values())
        domain_tag = "VisDrone (Overhead / Drone)" if self.is_visdrone else "COCO (Road / Dashcam / CCTV)"
        print(f"[Detector] Detected Domain: {domain_tag} ({len(self.model_names)} classes)")

        # Build traffic classes filter
        self.traffic_class_ids = []
        for cid, cname in self.model_names.items():
            if str(cname).lower().strip() in TRAFFIC_NAME_CANONICAL:
                self.traffic_class_ids.append(int(cid))

        # 2. Tracking & Geometry
        self.tracker_mgr = ByteTrackManager(max_missed_frames=15)

        if Path(self.calibration_file).exists():
            print(f"[Geometry] Loading calibrated lanes from: {self.calibration_file}")
            self.geometry = IntersectionGeometry.load_from_json(self.calibration_file)
        else:
            print("[Geometry] Using default 4-way intersection layout.")
            self.geometry = IntersectionGeometry.create_standard_4way(1280, 720, px_per_m=12.0)

        self.kinematics = KinematicsEstimator(self.geometry, queue_speed_threshold_kmh=4.0)

        # 3. Traffic State & Controller Subsystems (with Gap-Out and EVP)
        self.state_mgr = TrafficStateManager(approach_ids=["North", "South", "East", "West"])
        self.predictor = ShortTermTrafficPredictor(alpha=0.35, forecast_horizon_sec=10.0)
        self.controller = AdaptiveTrafficSignalController(
            min_green_sec=10,
            max_green_sec=40,
            yellow_sec=3,
            all_red_sec=2,
            gap_threshold_sec=2.5,
            min_green_before_gapout=6.0
        )

        # 4. Software Traffic Signal Simulator
        self.simulator = SoftwareTrafficSignalSimulator(1280, 720)

        # 5. Physical Arduino Hardware Bridge
        self.arduino = ArduinoTrafficLightBridge(port=arduino_port)

        print("[System] Perception, adaptive controller (with Gap-Out & EVP), and simulation subsystems ready.\n")

    def run(
        self,
        video_source: Any,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        show_gui: bool = True,
        loop: bool = True,
        playback_speed: float = 1.0,
        start_paused: bool = False,
        interactive_calibrate: bool = False
    ) -> Dict[str, Any]:
        """
        Executes the closed-loop perception and adaptive control loop.
        Supports interactive pause [SPACE], slow-down [-], speed-up [+],
        Emergency Preemption [E], frame stepping [.], and phase override [S] in GUI.
        """
        # Parse video source (webcam index vs file path)
        try:
            source = int(video_source)
            is_webcam = True
            src_name = f"Webcam Device #{source}"
        except ValueError:
            source = str(video_source)
            is_webcam = False
            src_name = Path(source).name

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video / camera source: {source}")

        # Interactive calibration option
        if interactive_calibrate:
            ret, first_frame = cap.read()
            if ret and first_frame is not None:
                print("\n[Calibration] Launching Interactive Lane Calibrator...")
                calib = InteractiveLaneCalibrator(first_frame, output_path=self.calibration_file)
                if calib.run():
                    self.geometry = IntersectionGeometry.load_from_json(self.calibration_file)
                    self.kinematics.geometry = self.geometry
                    print("[Calibration] Calibrated geometry reloaded successfully!\n")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 0 or np.isnan(fps):
            fps = 25.0

        # Adjust geometry if resolution varies and no custom calibration
        if not Path(self.calibration_file).exists() and (width != 1280 or height != 720):
            self.geometry = IntersectionGeometry.create_standard_4way(width, height, px_per_m=width / 100.0)
            self.kinematics.geometry = self.geometry

        self.simulator.width = width
        self.simulator.height = height

        writer = None
        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_p), fourcc, fps, (width, height))
            print(f"[Recording] Output video writing to: {out_p}")

        print(f"[Stream] Processing {src_name} ({width}x{height} @ {fps:.1f} FPS native)...")
        print("GUI Controls: [SPACE] Pause | [-/+] Speed | [E] Emergency Preemption | [S] Next Phase | [Q] Exit\n")

        frame_count = 0
        t_start = time.time()
        fps_smooth = 25.0

        current_signals = {"North": "GREEN", "South": "GREEN", "East": "RED", "West": "RED"}
        is_paused = start_paused
        step_one_frame = False
        last_rendered_frame = None

        try:
            while cap.isOpened():
                # -------------------------------------------------------------
                # Interactive Pause Loop
                # -------------------------------------------------------------
                if is_paused and not step_one_frame:
                    if show_gui and last_rendered_frame is not None:
                        pause_disp = last_rendered_frame.copy()
                        self._render_playback_controls(
                            pause_disp,
                            is_paused=True,
                            playback_speed=playback_speed,
                            effective_fps=fps_smooth,
                            native_fps=fps
                        )
                        cv2.imshow("Traffic Light Vehicle Density Detector", pause_disp)
                        key = cv2.waitKey(30) & 0xFF
                        if key in [ord(' '), ord('p'), ord('P')]:
                            is_paused = False
                            print(f"\n[Playback] RESUMED (Speed: {playback_speed:.2f}x)")
                        elif key in [ord('.'), ord('f'), ord('F')]:
                            step_one_frame = True
                            print("\n[Playback] Stepped 1 frame forward")
                        elif key in [ord('-'), ord('_'), ord('[')]:
                            playback_speed = max(0.05, round(playback_speed - 0.25 if playback_speed > 0.35 else playback_speed - 0.05, 2))
                            print(f"\n[Playback] Slowed down -> Speed: {playback_speed:.2f}x (Target: {fps * playback_speed:.1f} FPS)")
                        elif key in [ord('+'), ord('='), ord(']')]:
                            playback_speed = min(5.0, round(playback_speed + 0.25 if playback_speed >= 0.25 else playback_speed + 0.1, 2))
                            print(f"\n[Playback] Speed up -> Speed: {playback_speed:.2f}x (Target: {fps * playback_speed:.1f} FPS)")
                        elif key in [ord('r'), ord('R')]:
                            playback_speed = 1.0
                            print(f"\n[Playback] Speed reset to 1.0x (Target: {fps:.1f} FPS)")
                        elif key in [ord('s'), ord('S')]:
                            print("\n[Manual Override] Switching signal phase...")
                            self.controller.current_allocated_duration = 0.1
                        elif key in [ord('e'), ord('E')]:
                            if self.controller.emergency_preemption_active:
                                self.controller.clear_emergency_preemption()
                            else:
                                curr_dir = "EW" if self.controller.current_phase in [SignalPhase.PHASE_NS_GREEN, SignalPhase.PHASE_NS_YELLOW] else "NS"
                                self.controller.trigger_emergency_preemption(direction=curr_dir, hold_duration_sec=14.0)
                        elif key in [ord('q'), ord('Q'), 27]:
                            print("\nUser requested exit.")
                            break
                    else:
                        time.sleep(0.03)
                    continue

                step_one_frame = False
                t_frame_start = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    if is_webcam:
                        continue
                    elif loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret:
                            break
                    else:
                        break

                frame_count += 1
                now = time.time()

                # -------------------------------------------------------------
                # 1. YOLOv8n Object Detection + ByteTrack Tracking
                # -------------------------------------------------------------
                classes_filter = self.traffic_class_ids if self.traffic_class_ids else None
                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=self.conf_thresh,
                    classes=classes_filter,
                    device=self.device,
                    verbose=False
                )

                # 2. Update Active Tracks
                active_tracks = self.tracker_mgr.update_with_ultralytics_results(
                    results,
                    now,
                    is_visdrone=self.is_visdrone,
                    names_dict=self.model_names
                )

                # 3. Kinematics & Approach Estimation
                self.kinematics.update_kinematics(active_tracks)

                # 3b. Stop-line Crossing Detection (Vehicle Departure for Gap-Out)
                for t in active_tracks:
                    if len(t.history) >= 2:
                        p_prev = t.history[-2]
                        p_curr = t.history[-1]
                        lane_id = getattr(t, "current_lane", None) or self.geometry.assign_lane(p_curr)
                        if lane_id and lane_id in self.geometry.lanes:
                            lane = self.geometry.lanes[lane_id]
                            if lane.has_crossed_stopline(p_prev, p_curr):
                                self.controller.notify_vehicle_departure(lane.approach, timestamp=now)

                # 3c. Automated Emergency Vehicle Preemption Detection
                for t in active_tracks:
                    c_name = getattr(t, "class_name", "").lower()
                    if c_name in ["ambulance", "emergency", "police", "fire truck"]:
                        lane_id = getattr(t, "current_lane", None) or self.geometry.assign_lane(t.centroid)
                        if lane_id and lane_id in self.geometry.lanes:
                            app_name = self.geometry.lanes[lane_id].approach
                            corr = "NS" if app_name in ["North", "South"] else "EW"
                            if not self.controller.emergency_preemption_active:
                                self.controller.trigger_emergency_preemption(direction=corr, hold_duration_sec=14.0)

                # -------------------------------------------------------------
                # 4. Traffic State S(t) Estimation
                # -------------------------------------------------------------
                snapshot = self.state_mgr.update_from_tracks(
                    active_tracks=active_tracks,
                    timestamp=now,
                    current_signals=current_signals,
                    px_per_m=self.geometry.pixels_per_meter
                )

                # -------------------------------------------------------------
                # 5. Short-Term Prediction (EWMA arrivals & queue growth)
                # -------------------------------------------------------------
                predictions = self.predictor.predict(snapshot)

                # -------------------------------------------------------------
                # 6. Adaptive Controller -> GREEN TIME DECISION
                # -------------------------------------------------------------
                decision = self.controller.compute_decision(snapshot, predictions)
                current_signals = decision.signals

                # -------------------------------------------------------------
                # 7. Arduino Hardware Output
                # -------------------------------------------------------------
                ns_light = decision.signals.get("North", "RED")
                ew_light = decision.signals.get("East", "RED")
                self.arduino.send_signal_state(ns_light, ew_light, decision.countdown_seconds)

                # -------------------------------------------------------------
                # 8. Render Visual Overlays
                # -------------------------------------------------------------
                # Draw vehicle bounding boxes & trajectories
                self._render_vehicles(frame, active_tracks)

                # Draw Software Traffic Signal Simulator Overlay with Calibrated Geometry
                frame = self.simulator.render_overlay(
                    frame=frame,
                    decision=decision,
                    snapshot=snapshot,
                    arduino_status=self.arduino.status_text,
                    inference_fps=fps_smooth,
                    geometry=self.geometry
                )

                # Render bottom Playback Control Bar
                self._render_playback_controls(
                    frame,
                    is_paused=False,
                    playback_speed=playback_speed,
                    effective_fps=fps_smooth,
                    native_fps=fps
                )

                last_rendered_frame = frame.copy()

                # Write output frame if recording
                if writer:
                    writer.write(frame)

                # Compute running FPS
                dt_frame = time.perf_counter() - t_frame_start
                fps_instant = 1.0 / max(0.001, dt_frame)
                fps_smooth = 0.9 * fps_smooth + 0.1 * fps_instant

                # Periodic terminal log (every 30 frames)
                if frame_count % 30 == 0:
                    ns_pcu = snapshot.approaches['North'].total_pcu + snapshot.approaches['South'].total_pcu
                    ew_pcu = snapshot.approaches['East'].total_pcu + snapshot.approaches['West'].total_pcu
                    gap_info = " [GAP-OUT]" if decision.is_gapout else ""
                    em_info = f" [EVP:{decision.emergency_corridor}]" if decision.is_emergency_preemption else ""
                    print(f"[Frame {frame_count:04d}] Phase: {decision.active_phase.value:<22} | "
                          f"Timer: {decision.countdown_seconds:02d}s | "
                          f"NS: {ns_pcu:4.1f} PCU ({ns_light:<6}) | "
                          f"EW: {ew_pcu:4.1f} PCU ({ew_light:<6}) | "
                          f"Speed: {playback_speed:.2f}x{gap_info}{em_info} | "
                          f"{fps_smooth:4.1f} FPS")

                # GUI Display & Playback Speed Sync
                if show_gui:
                    if not is_webcam and playback_speed > 0:
                        target_frame_time = (1.0 / max(1.0, fps)) / playback_speed
                        delay_sec = max(0.001, target_frame_time - dt_frame)
                        wait_ms = max(1, int(delay_sec * 1000))
                    else:
                        wait_ms = 1

                    cv2.imshow("Traffic Light Vehicle Density Detector", frame)
                    key = cv2.waitKey(wait_ms) & 0xFF
                    if key in [ord('q'), ord('Q'), 27]:
                        print("\nUser requested exit.")
                        break
                    elif key in [ord(' '), ord('p'), ord('P')]:
                        is_paused = True
                        print(f"\n[Playback] PAUSED - Press SPACE to resume, [.] to step, [-/+] to change speed.")
                    elif key in [ord('-'), ord('_'), ord('[')]:
                        playback_speed = max(0.05, round(playback_speed - 0.25 if playback_speed > 0.35 else playback_speed - 0.05, 2))
                        print(f"\n[Playback] Slowed down -> Speed: {playback_speed:.2f}x (Target: {fps * playback_speed:.1f} FPS)")
                    elif key in [ord('+'), ord('='), ord(']')]:
                        playback_speed = min(5.0, round(playback_speed + 0.25 if playback_speed >= 0.25 else playback_speed + 0.1, 2))
                        print(f"\n[Playback] Speed up -> Speed: {playback_speed:.2f}x (Target: {fps * playback_speed:.1f} FPS)")
                    elif key in [ord('r'), ord('R')]:
                        playback_speed = 1.0
                        print(f"\n[Playback] Speed reset to 1.0x (Target: {fps:.1f} FPS)")
                    elif key in [ord('s'), ord('S')]:
                        print("[Manual Override] Switching signal phase...")
                        self.controller.current_allocated_duration = 0.1
                    elif key in [ord('e'), ord('E')]:
                        if self.controller.emergency_preemption_active:
                            self.controller.clear_emergency_preemption()
                        else:
                            curr_dir = "EW" if self.controller.current_phase in [SignalPhase.PHASE_NS_GREEN, SignalPhase.PHASE_NS_YELLOW] else "NS"
                            self.controller.trigger_emergency_preemption(direction=curr_dir, hold_duration_sec=14.0)

                if max_frames and frame_count >= max_frames:
                    print(f"\nReached max frame limit: {max_frames}")
                    break

        finally:
            cap.release()
            if writer:
                writer.release()
            if show_gui:
                cv2.destroyAllWindows()
            self.arduino.close()

        total_wall = time.time() - t_start
        avg_fps = frame_count / max(0.1, total_wall)

        summary = {
            "frames_processed": frame_count,
            "total_wall_seconds": round(total_wall, 2),
            "average_fps": round(avg_fps, 2),
            "output_saved": output_path if output_path else "None"
        }
        print("=" * 70)
        print(f"COMPLETED: Processed {frame_count} frames in {total_wall:.2f}s ({avg_fps:.1f} FPS)")
        print("=" * 70)
        return summary

    def _render_vehicles(self, frame: np.ndarray, active_tracks: List[Any]):
        """Renders bounding boxes, peach trajectory trails, and class/PCU badges."""
        for t in active_tracks:
            # Trajectory trail (peach)
            if len(t.history) >= 2:
                pts = np.array(t.history, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, COLOR_TRAJECTORY, 2, cv2.LINE_AA)

            if hasattr(t, "last_bbox") and len(t.last_bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in t.last_bbox]

                # Bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BBOX, 2)

                # Class and PCU badge
                c_name = getattr(t, "class_name", "car").upper()
                pcu_val = getattr(t, "pcu_weight", 1.0)
                spd_val = int(getattr(t, "speed_kmh", 0.0))
                tid = getattr(t, "track_id", 0)

                label_str = f"ID:{tid} {c_name} | {spd_val}km/h | {pcu_val:.1f}PCU"
                (lw, lh), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)

                badge_y1 = max(0, y1 - lh - 8)
                cv2.rectangle(frame, (x1, badge_y1), (x1 + lw + 8, y1), (25, 25, 30), -1)
                cv2.rectangle(frame, (x1, badge_y1), (x1 + lw + 8, y1), (80, 80, 95), 1)
                cv2.putText(frame, label_str, (x1 + 4, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

                # Motion vector (red arrow)
                vec = t.get_motion_vector()
                if vec:
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    vx, vy = vec
                    tip_x = int(cx + np.clip(vx * 0.4, -40, 40))
                    tip_y = int(cy + np.clip(vy * 0.4, -40, 40))
                    cv2.arrowedLine(frame, (cx, cy), (tip_x, tip_y), COLOR_MOTION_VEC, 2, tipLength=0.3)

    def _render_playback_controls(
        self,
        frame: np.ndarray,
        is_paused: bool,
        playback_speed: float,
        effective_fps: float,
        native_fps: float
    ):
        """Renders bottom control bar and large pause banner when paused."""
        h, w = frame.shape[:2]
        bar_h = 32
        bar_y = h - bar_h

        # Semi-transparent dark bottom bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bar_y), (w, h), (18, 18, 24), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.line(frame, (0, bar_y), (w, bar_y), (60, 60, 78), 1)

        # Status Pill: [⏸ PAUSED] or [▶ PLAYING]
        status_tag = "⏸ PAUSED" if is_paused else "▶ PLAYING"
        status_col = (0, 215, 255) if is_paused else (70, 240, 90)
        cv2.putText(frame, status_tag, (15, bar_y + 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.48, status_col, 1, cv2.LINE_AA)

        # Speed Multiplier + Target FPS
        target_fps = native_fps * playback_speed
        speed_str = f"Speed: {playback_speed:.2f}x ({target_fps:.1f} FPS | Vid: {native_fps:.1f} FPS)"
        cv2.putText(frame, speed_str, (120, bar_y + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 235), 1, cv2.LINE_AA)

        # Shortcut hints
        hints_str = "[SPACE] Pause | [-/+] Speed | [E] Emergency Preemption | [.] Step | [S] Phase | [Q] Exit"
        cv2.putText(frame, hints_str, (max(w - 760, 420), bar_y + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 195, 210), 1, cv2.LINE_AA)

        # Prominent central banner if paused
        if is_paused:
            bw, bh = 500, 76
            bx = (w - bw) // 2
            by = (h - bh) // 2
            banner = frame.copy()
            cv2.rectangle(banner, (bx, by), (bx + bw, by + bh), (15, 15, 25), -1)
            cv2.addWeighted(banner, 0.88, frame, 0.12, 0, frame)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 210, 255), 2)

            cv2.putText(frame, "PAUSED", (bx + 185, by + 34),
                        cv2.FONT_HERSHEY_DUPLEX, 0.95, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Press SPACE to Resume | [.] Step Frame | [-/+] Adjust Speed", (bx + 26, by + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 235, 245), 1, cv2.LINE_AA)

def main():
    parser = argparse.ArgumentParser(description="Traffic Light Vehicle Density Detector")
    parser.add_argument("--source", type=str, default=r"c:\g\tests\real_traffic_video.mp4",
                        help="Video file path OR webcam index (e.g. 0, 1)")
    parser.add_argument("--model", type=str, default=r"c:\g\weights\detection_yolov8n.pt",
                        help="Path to YOLO weights (.pt)")
    parser.add_argument("--conf", type=float, default=0.18,
                        help="Confidence threshold (default: 0.18 for sensitive detection)")
    parser.add_argument("--arduino", type=str, default=None,
                        help="Arduino COM port (e.g. COM3). Auto-detected if omitted.")
    parser.add_argument("--output", type=str, default=r"c:\g\tests\annotated_density_detector.mp4",
                        help="Path to save annotated video output")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum frames to process")
    parser.add_argument("--device", type=str, default="0",
                        help="Inference device: '0' for CUDA GPU, 'cpu' for CPU")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Initial playback speed multiplier relative to video real-time FPS (e.g. 0.5 = half speed, 1.0 = real-time)")
    parser.add_argument("--start-paused", action="store_true",
                        help="Start playback immediately paused")
    parser.add_argument("--calibrate", action="store_true",
                        help="Launch interactive lane ROI calibration before running detector")
    parser.add_argument("--calibration-file", type=str, default=r"c:\g\lane_calibration.json",
                        help="Path to save/load lane geometry calibration JSON")
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI preview window")
    parser.add_argument("--no-loop", dest="loop", action="store_false", default=True,
                        help="Do not loop video file continuously")
    args = parser.parse_args()

    # Fallback to CPU if CUDA is unavailable
    import torch
    dev = args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu"

    app = TrafficLightVehicleDensityDetectorApp(
        model_path=args.model,
        conf_thresh=args.conf,
        arduino_port=args.arduino,
        device=dev,
        calibration_file=args.calibration_file
    )

    app.run(
        video_source=args.source,
        output_path=args.output,
        max_frames=args.max_frames,
        show_gui=not args.headless,
        loop=args.loop,
        playback_speed=args.speed,
        start_paused=args.start_paused,
        interactive_calibrate=args.calibrate
    )

if __name__ == "__main__":
    main()
