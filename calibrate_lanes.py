"""
=============================================================================
DETECTION - INTERACTIVE LANE & STOP-LINE CALIBRATION UTILITY
=============================================================================
Allows users to visually define approach polygons, virtual stop lines, and
pixel-to-meter scaling for any camera or video angle using mouse clicks.

Controls:
  - Left Click : Place point for current step
  - 'z' or 'u' : Undo last point
  - 't'        : Load standard 4-way template
  - 'r'        : Reset current approach
  - 's' / Enter: Save calibration to JSON
  - 'q' / ESC  : Exit
=============================================================================
"""
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np

# Ensure root in path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from detection.perception.lane_geometry import Lane, IntersectionGeometry

APPROACH_COLORS = {
    "North": (255, 200, 50),   # Cyan/Light Blue
    "South": (80, 240, 100),   # Bright Green
    "East":  (50, 180, 255),   # Amber/Orange
    "West":  (220, 80, 255)    # Magenta
}

APPROACH_DIRECTIONS = {
    "North": "Southbound",
    "South": "Northbound",
    "East":  "Westbound",
    "West":  "Eastbound"
}

class InteractiveLaneCalibrator:
    def __init__(self, frame: np.ndarray, output_path: str = r"c:\g\lane_calibration.json"):
        self.raw_frame = frame.copy()
        self.h, self.w = frame.shape[:2]
        self.output_path = output_path

        self.approaches = ["North", "South", "East", "West"]
        self.current_approach_idx = 0

        # State machine: "POLYGON" (4 clicks) -> "STOPLINE" (2 clicks)
        self.sub_step = "POLYGON"

        self.calibrated_lanes: Dict[str, Dict[str, Any]] = {}
        self.current_points: List[Tuple[int, int]] = []

        # Scale calibration (2 points)
        self.scale_points: List[Tuple[int, int]] = []
        self.scale_done = False
        self.pixels_per_meter = 12.0

        self.window_name = "Detection Interactive Lane Calibration"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_approach_idx < len(self.approaches):
                self._handle_approach_click(x, y)
            elif not self.scale_done:
                self._handle_scale_click(x, y)

    def _handle_approach_click(self, x: int, y: int):
        app_name = self.approaches[self.current_approach_idx]
        self.current_points.append((x, y))

        if self.sub_step == "POLYGON":
            if len(self.current_points) == 4:
                # Polygon complete, move to stopline
                if app_name not in self.calibrated_lanes:
                    self.calibrated_lanes[app_name] = {}
                self.calibrated_lanes[app_name]["polygon"] = list(self.current_points)
                self.current_points = []
                self.sub_step = "STOPLINE"
                print(f"[{app_name}] Polygon defined. Now click 2 points to define the STOP LINE.")

        elif self.sub_step == "STOPLINE":
            if len(self.current_points) == 2:
                self.calibrated_lanes[app_name]["stopline"] = (self.current_points[0], self.current_points[1])
                self.current_points = []
                self.sub_step = "POLYGON"
                self.current_approach_idx += 1
                if self.current_approach_idx < len(self.approaches):
                    next_app = self.approaches[self.current_approach_idx]
                    print(f"[{app_name}] Done! Next: Click 4 points for {next_app} approach polygon.")
                else:
                    print("\nAll 4 approaches defined! Optional: Click 2 points across a known 3.5m lane width for scale (or press 's' to save).")

    def _handle_scale_click(self, x: int, y: int):
        self.scale_points.append((x, y))
        if len(self.scale_points) == 2:
            dx = self.scale_points[1][0] - self.scale_points[0][0]
            dy = self.scale_points[1][1] - self.scale_points[0][1]
            dist_px = float(np.hypot(dx, dy))
            self.pixels_per_meter = max(1.0, round(dist_px / 3.5, 2))  # Assume 3.5m standard lane width
            self.scale_done = True
            print(f"[Scale] Calibrated scale: {self.pixels_per_meter:.2f} px/meter (based on {dist_px:.1f}px = 3.5m).")

    def load_standard_template(self):
        """Loads a proportional standard 4-way template."""
        geom = IntersectionGeometry.create_standard_4way(self.w, self.h, px_per_m=self.w / 100.0)
        for lid, lane in geom.lanes.items():
            app = lane.approach
            self.calibrated_lanes[app] = {
                "polygon": lane.polygon.tolist(),
                "stopline": lane.stopline
            }
        self.pixels_per_meter = round(self.w / 100.0, 2)
        self.current_approach_idx = len(self.approaches)
        self.scale_done = True
        print("[Template] Standard 4-way template loaded! Press 's' to save or adjust.")

    def run(self) -> bool:
        print("\n" + "=" * 65)
        print("   INTERACTIVE LANE & STOP-LINE CALIBRATION   ")
        print("=" * 65)
        print(f"Step 1: Click 4 points for {self.approaches[0]} approach polygon.")
        print("Hotkeys: [Z]=Undo  [T]=Template  [R]=Reset  [S]/[Enter]=Save  [Q]=Quit\n")

        saved = False
        while True:
            display = self.raw_frame.copy()
            self._render_overlay(display)
            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(30) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                print("\nCalibration cancelled without saving.")
                break
            elif key in [ord('s'), ord('S'), 13]:  # Enter
                if len(self.calibrated_lanes) == 4:
                    self.save()
                    saved = True
                    break
                else:
                    print(f"\n[Warning] Please finish all 4 approaches first ({len(self.calibrated_lanes)}/4 done) or press 'T' for template.")
            elif key in [ord('z'), ord('u'), ord('Z'), ord('U')]:
                if self.current_points:
                    self.current_points.pop()
                    print("[Undo] Removed last clicked point.")
            elif key in [ord('t'), ord('T')]:
                self.load_standard_template()
            elif key in [ord('r'), ord('R')]:
                self.calibrated_lanes.clear()
                self.current_points.clear()
                self.scale_points.clear()
                self.scale_done = False
                self.current_approach_idx = 0
                self.sub_step = "POLYGON"
                print("\n[Reset] Calibration cleared. Restarting from North approach.")

        cv2.destroyAllWindows()
        return saved

    def _render_overlay(self, frame: np.ndarray):
        # 1. Draw top instruction header
        cv2.rectangle(frame, (0, 0), (self.w, 48), (15, 15, 22), -1)
        cv2.line(frame, (0, 48), (self.w, 48), (0, 180, 255), 2)

        if self.current_approach_idx < len(self.approaches):
            curr_app = self.approaches[self.current_approach_idx]
            msg = f"Step: Define {curr_app.upper()} Approach | Need: {self.sub_step} ({len(self.current_points)} points)"
            header_col = APPROACH_COLORS[curr_app]
        else:
            if not self.scale_done:
                msg = f"Approaches Ready! Click 2 points across 3.5m lane for scale, or press 'S' to Save."
                header_col = (0, 255, 180)
            else:
                msg = f"Calibration Complete ({self.pixels_per_meter:.1f} px/m)! Press [S] or [Enter] to Save."
                header_col = (50, 255, 100)

        cv2.putText(frame, msg, (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.55, header_col, 1, cv2.LINE_AA)
        cv2.putText(frame, "[Z]=Undo  [T]=Template  [R]=Reset  [S]=Save  [Q]=Quit", (self.w - 450, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 200, 215), 1, cv2.LINE_AA)

        # 2. Render already calibrated approaches
        for app_name, ldata in self.calibrated_lanes.items():
            col = APPROACH_COLORS.get(app_name, (200, 200, 200))
            poly = np.array(ldata["polygon"], dtype=np.int32)

            # Shaded fill
            overlay = frame.copy()
            cv2.fillPoly(overlay, [poly], col)
            cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
            cv2.polylines(frame, [poly], True, col, 2, cv2.LINE_AA)

            # Center label
            cx = int(np.mean(poly[:, 0]))
            cy = int(np.mean(poly[:, 1]))
            cv2.putText(frame, app_name.upper(), (cx - 25, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            # Stop line
            if "stopline" in ldata:
                p1, p2 = ldata["stopline"]
                cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 3, cv2.LINE_AA)
                cv2.circle(frame, (int(p1[0]), int(p1[1])), 4, (0, 255, 255), -1)
                cv2.circle(frame, (int(p2[0]), int(p2[1])), 4, (0, 255, 255), -1)

        # 3. Render in-progress points
        for i, pt in enumerate(self.current_points):
            cv2.circle(frame, pt, 5, (0, 255, 255), -1)
            cv2.putText(frame, str(i + 1), (pt[0] + 6, pt[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)

        if len(self.current_points) >= 2:
            pts_arr = np.array(self.current_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts_arr], False, (0, 255, 255), 1, cv2.LINE_AA)

        # 4. Render scale points
        for pt in self.scale_points:
            cv2.circle(frame, pt, 5, (255, 100, 255), -1)
        if len(self.scale_points) == 2:
            cv2.line(frame, self.scale_points[0], self.scale_points[1], (255, 100, 255), 2)
            cv2.putText(frame, f"3.5m ({self.pixels_per_meter:.1f} px/m)",
                        ((self.scale_points[0][0] + self.scale_points[1][0]) // 2,
                         (self.scale_points[0][1] + self.scale_points[1][1]) // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 255), 1, cv2.LINE_AA)

    def save(self):
        geom = IntersectionGeometry(pixels_per_meter=self.pixels_per_meter, resolution=(self.w, self.h))
        for app_name, ldata in self.calibrated_lanes.items():
            lid = f"{app_name}_L1"
            dir_str = APPROACH_DIRECTIONS.get(app_name, "Southbound")
            lane = Lane(
                lane_id=lid,
                polygon=ldata["polygon"],
                stopline=ldata["stopline"],
                direction=dir_str,
                approach=app_name
            )
            geom.add_lane(lane)

        geom.save_to_json(self.output_path)
        print(f"\n[SUCCESS] Calibration saved to: {self.output_path}")

def main():
    parser = argparse.ArgumentParser(description="Detection Interactive Lane Calibration Utility")
    parser.add_argument("--source", type=str, default=r"c:\g\tests\real_traffic_video.mp4",
                        help="Video file path OR camera index")
    parser.add_argument("--output", type=str, default=r"c:\g\lane_calibration.json",
                        help="Path to save calibrated JSON file")
    args = parser.parse_args()

    try:
        src = int(args.source)
    except ValueError:
        src = str(args.source)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[Error] Could not open video source: {src}")
        sys.exit(1)

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("[Error] Failed to read frame from video source.")
        sys.exit(1)

    calibrator = InteractiveLaneCalibrator(frame, output_path=args.output)
    calibrator.run()

if __name__ == "__main__":
    main()
