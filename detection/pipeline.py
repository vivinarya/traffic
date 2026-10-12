"""
Detection Master Perception Pipeline
Connects Video Ingest -> YOLOv8 Detection -> ByteTrack Tracking ->
Kinematics / PCU Estimation -> SQLite Telemetry Logging -> HUD Visualization Overlay.
"""
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import cv2
import numpy as np
from ultralytics import YOLO

from detection.database.db_manager import EdgeDBManager
from detection.perception.lane_geometry import IntersectionGeometry
from detection.perception.kinematics import KinematicsEstimator
from detection.perception.tracker import ByteTrackManager

# HUD Color Palette (matching Detection Specification)
COLOR_BBOX = (240, 240, 240)       # Off-white
COLOR_TRAJECTORY = (180, 200, 255) # Peach / Light Salmon in BGR
COLOR_MOTION_VEC = (0, 0, 230)     # Vibrant Red
COLOR_STOPLINE = (0, 70, 255)       # Red / Amber stopline
COLOR_LANE_ZONE = (70, 150, 70)    # Translucent Green corridor
COLOR_TEXT = (255, 255, 255)
COLOR_PANEL_BG = (25, 20, 30)      # Sleek dark HUD panel

class EdgeFlowPerceptionPipeline:
    def __init__(
        self,
        model_weights: str = "yolov8n.pt",
        db_path: Path = Path(r"c:\g\edgeflow_telemetry.db"),
        conf_thresh: float = 0.30,
        device: str = "cpu",
        is_visdrone: bool = False
    ):
        self.model_weights = model_weights
        self.conf_thresh = conf_thresh
        self.device = device
        self.is_visdrone = is_visdrone

        # Initialize Subsystems
        print("Initializing Detection Subsystems...")
        self.db = EdgeDBManager(db_path)
        self.geometry = IntersectionGeometry.create_standard_4way(1280, 720, px_per_m=12.0)
        self.kinematics = KinematicsEstimator(self.geometry, queue_speed_threshold_kmh=4.0)
        self.tracker_mgr = ByteTrackManager(max_missed_frames=20)

        print(f"Loading YOLO tracking model ({model_weights})...")
        self.model = YOLO(model_weights)
        print("Detection Perception Subsystem is ready.")

    def process_video(
        self,
        video_source: str,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        show_preview: bool = False
    ) -> Dict[str, Any]:
        """
        Executes full perception loop on an input video source.
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {video_source}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_source = cap.get(cv2.CAP_PROP_FPS) or 25.0

        # Adjust geometry if resolution differs
        if width != 1280 or height != 720:
            self.geometry = IntersectionGeometry.create_standard_4way(width, height, px_per_m=width / 100.0)
            self.kinematics.geometry = self.geometry

        writer = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps_source, (width, height))

        frame_count = 0
        total_det_time = 0.0
        total_track_time = 0.0
        start_wall_time = time.time()

        print(f"Beginning Detection processing on {video_source} ({width}x{height} @ {fps_source:.1f} fps)...")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                timestamp = time.time()

                # 1. Detection + Tracking with ByteTrack
                t0 = time.perf_counter()
                classes_filter = None if self.is_visdrone else [0, 1, 2, 3, 5, 7]
                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=self.conf_thresh,
                    classes=classes_filter,
                    device=self.device,
                    verbose=False
                )
                t_det = (time.perf_counter() - t0) * 1000.0
                total_det_time += t_det

                # 2. Update Multi-Object Tracker State
                t1 = time.perf_counter()
                active_tracks = self.tracker_mgr.update_with_ultralytics_results(
                    results, timestamp, is_visdrone=self.is_visdrone
                )

                # 3. State Variable Estimation (Kinematics & Queue States)
                self.kinematics.update_kinematics(active_tracks)
                lane_aggregates = self.kinematics.compute_lane_aggregates(active_tracks, timestamp)
                t_track = (time.perf_counter() - t1) * 1000.0
                total_track_time += t_track

                # 4. Telemetry Logging to SQLite
                db_tracks = []
                for t in active_tracks:
                    if t.last_bbox:
                        db_tracks.append({
                            "timestamp": timestamp,
                            "frame_id": int(frame_count),
                            "track_id": int(t.track_id),
                            "class_id": int(t.class_id),
                            "class_name": str(t.class_name),
                            "confidence": float(t.confidence),
                            "bbox_x1": t.last_bbox[0],
                            "bbox_y1": t.last_bbox[1],
                            "bbox_x2": t.last_bbox[2],
                            "bbox_y2": t.last_bbox[3],
                            "lane_id": t.lane_id,
                            "speed_kmh": round(t.speed_kmh, 1),
                            "headway_m": round(t.headway_m, 1),
                            "pcu_weight": t.pcu_weight,
                            "is_queued": 1 if t.is_queued else 0
                        })
                self.db.log_tracks_batch(db_tracks)
                self.db.log_lane_aggregates(lane_aggregates)

                # Periodic Heartbeat Logging (every 10 frames)
                if frame_count % 10 == 0:
                    current_fps = 1000.0 / max(1.0, (t_det + t_track))
                    self.db.log_heartbeat(current_fps, t_det, t_track, t_det + t_track)

                # 5. Render HUD Overlay
                annotated_frame = self._render_hud(frame, active_tracks, lane_aggregates, t_det + t_track)

                if writer:
                    writer.write(annotated_frame)

                if max_frames and frame_count >= max_frames:
                    print(f"Reached max frame limit: {max_frames}")
                    break

        finally:
            cap.release()
            if writer:
                writer.release()

        total_wall_time = time.time() - start_wall_time
        avg_fps = frame_count / max(0.001, total_wall_time)
        summary = {
            "frames_processed": frame_count,
            "total_wall_seconds": round(total_wall_time, 2),
            "average_fps": round(avg_fps, 2),
            "avg_detection_latency_ms": round(total_det_time / max(1, frame_count), 2),
            "avg_tracking_latency_ms": round(total_track_time / max(1, frame_count), 2)
        }
        print(f"\nProcessing Completed. Summary: {summary}")
        return summary

    def _render_hud(
        self,
        frame: np.ndarray,
        tracks: List[Any],
        aggregates: List[Dict[str, Any]],
        latency_ms: float
    ) -> np.ndarray:
        """
        Renders HUD overlays matching Detection Slide 5 Schematic:
        - Off-White Bounding Boxes with CAR_xx [Tx]
        - Peach Historical Trajectory Lines
        - Red Motion Vectors
        - Stop Lines & Corridor Metrics HUD Panel
        """
        out = frame.copy()
        h, w, _ = out.shape

        # 1. Draw Lane Polygons and Stoplines
        for lane_id, lane in self.geometry.lanes.items():
            cv2.polylines(out, [lane.polygon], isClosed=True, color=(60, 60, 60), thickness=1)
            # Stopline
            pt1, pt2 = lane.stopline
            cv2.line(out, pt1, pt2, COLOR_STOPLINE, 3)

        # 2. Draw Vehicle Trajectories, BBoxes, and Motion Vectors
        for t in tracks:
            # Trajectory Trail (Peach)
            pts = list(t.history)
            for i in range(1, len(pts)):
                cv2.line(out, (int(pts[i - 1][0]), int(pts[i - 1][1])),
                              (int(pts[i][0]), int(pts[i][1])),
                              COLOR_TRAJECTORY, 2)

            # Predicted Motion Vector (Red arrow)
            mvec = t.get_motion_vector()
            if mvec and len(pts) > 0:
                cx, cy = int(pts[-1][0]), int(pts[-1][1])
                vx, vy = mvec
                end_x = int(cx + vx * 0.4)
                end_y = int(cy + vy * 0.4)
                cv2.arrowedLine(out, (cx, cy), (end_x, end_y), COLOR_MOTION_VEC, 2, tipLength=0.3)

            # Bounding Box (Off-white)
            if t.last_bbox:
                x1, y1, x2, y2 = map(int, t.last_bbox)
                box_color = (0, 165, 255) if t.is_queued else COLOR_BBOX  # Amber if queued, white if moving
                cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)

                # Label badge: e.g. CAR_01 [T3] 32km/h
                label = f"{t.class_name.upper()}_{t.track_id} | {t.speed_kmh:.0f}km/h | PCU:{t.pcu_weight}"
                if t.is_queued:
                    label += " [QUEUED]"

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 6, y1), (30, 30, 30), -1)
                cv2.putText(out, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # 3. Detection Telemetry Heads-Up Display (Top-Left overlay)
        panel_h, panel_w = 160, 380
        overlay = out.copy()
        cv2.rectangle(overlay, (15, 15), (15 + panel_w, 15 + panel_h), COLOR_PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)
        cv2.rectangle(out, (15, 15), (15 + panel_w, 15 + panel_h), (80, 80, 90), 1)

        cv2.putText(out, "DETECTION PERCEPTION SUBSYSTEM", (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 215, 0), 2)
        cv2.putText(out, f"Latency: {latency_ms:.1f}ms | Active Tracks: {len(tracks)}", (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        y_offset = 84
        for agg in aggregates[:4]:
            lane_text = f"{agg['lane_id']}: Q_Len={agg['queue_length_m']}m | PCU={agg['total_pcu']} | Spd={agg['avg_approach_speed']}km/h"
            cv2.putText(out, lane_text, (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 200), 1)
            y_offset += 20

        return out
