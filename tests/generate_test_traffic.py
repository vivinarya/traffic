"""
Detection Test Traffic Video Generator
Synthesizes a realistic multi-lane intersection video clip with heterogeneous traffic:
- Cars (PCU: 1.0)
- Buses (PCU: 3.0)
- Trucks (PCU: 3.0)
- Motorcycles (PCU: 0.5)
Features vehicles approaching the intersection, decelerating, queueing at red lights,
and departing on green lights.
"""
import math
import random
from pathlib import Path
import cv2
import numpy as np

OUTPUT_VIDEO = Path(r"c:\g\tests\sample_traffic.mp4")

# Vehicle visual properties for realistic synthetic generation
VEHICLE_SPECS = {
    "car": {"size": (40, 24), "color": (220, 180, 80), "pcu": 1.0, "max_speed": 4.5},
    "bus": {"size": (75, 30), "color": (60, 120, 220), "pcu": 3.0, "max_speed": 3.0},
    "truck": {"size": (80, 32), "color": (100, 150, 130), "pcu": 3.0, "max_speed": 3.0},
    "motorcycle": {"size": (22, 14), "color": (200, 70, 150), "pcu": 0.5, "max_speed": 5.0},
}

class SimVehicle:
    def __init__(self, v_type: str, approach: str, lane_pos: int, start_d: float, target_stop: float):
        self.v_type = v_type
        self.spec = VEHICLE_SPECS[v_type]
        self.approach = approach
        self.lane_pos = lane_pos
        self.d = start_d
        self.speed = self.spec["max_speed"]
        self.target_stop = target_stop
        self.w, self.h = self.spec["size"]
        self.color = self.spec["color"]

    def update(self, red_light: bool, leader_d: float = -999.0):
        # Desired stop location
        effective_stop = self.target_stop
        if leader_d > 0:
            effective_stop = min(effective_stop, leader_d - self.h - 15)

        dist_to_stop = effective_stop - self.d

        if red_light and dist_to_stop > 0:
            if dist_to_stop < 150:
                # Decelerate smoothly
                self.speed = max(0.0, self.speed - 0.12)
            if dist_to_stop < 5:
                self.speed = 0.0
        else:
            # Accelerate up to max speed
            self.speed = min(self.spec["max_speed"], self.speed + 0.1)

        self.d += self.speed

    def get_bbox_center(self, width: int, height: int):
        cx, cy = width // 2, height // 2
        lane_w = 60

        if self.approach == "North":
            # Moving top to bottom towards center
            x = cx - lane_w // 2
            y = self.d
            return x, y, self.w, self.h
        elif self.approach == "South":
            # Moving bottom to top towards center
            x = cx + lane_w // 2
            y = height - self.d
            return x, y, self.w, self.h
        elif self.approach == "East":
            # Moving right to left towards center
            x = width - self.d
            y = cy - lane_w // 2
            return x, y, self.h, self.w
        else: # West
            x = self.d
            y = cy + lane_w // 2
            return x, y, self.h, self.w

def generate_traffic_clip(output_file: Path = OUTPUT_VIDEO, duration_sec: int = 10, fps: int = 25):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1280, 720
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

    cx, cy = width // 2, height // 2
    lane_w = 60

    # Initialize vehicles across 4 approaches
    vehicles = []
    approaches = ["North", "South", "East", "West"]
    v_types = ["car", "bus", "car", "truck", "motorcycle", "car"]

    for i, app in enumerate(approaches):
        stop_d = (cy - 90) if app in ["North", "South"] else (cx - 90)
        # 3 vehicles in platoon per approach
        for k in range(3):
            vt = random.choice(v_types)
            start_dist = 50 + k * 80
            vehicles.append(SimVehicle(vt, app, 0, start_d=start_dist, target_stop=stop_d))

    print(f"Generating synthetic traffic video: {output_file} ({total_frames} frames)...")

    for frame_idx in range(total_frames):
        # Canvas background (Asphalt road & intersection)
        frame = np.full((height, width, 3), (35, 35, 40), dtype=np.uint8)

        # Draw Road Corridors
        # North-South Road
        cv2.rectangle(frame, (cx - lane_w, 0), (cx + lane_w, height), (50, 50, 55), -1)
        # East-West Road
        cv2.rectangle(frame, (0, cy - lane_w), (width, cy + lane_w), (50, 50, 55), -1)

        # Intersection Box
        cv2.rectangle(frame, (cx - lane_w, cy - lane_w), (cx + lane_w, cy + lane_w), (55, 55, 60), -1)

        # Road Markings (Lane dividers, dashed lines)
        cv2.line(frame, (cx, 0), (cx, cy - lane_w), (200, 200, 200), 2)
        cv2.line(frame, (cx, cy + lane_w), (cx, height), (200, 200, 200), 2)
        cv2.line(frame, (0, cy), (cx - lane_w, cy), (200, 200, 200), 2)
        cv2.line(frame, (cx + lane_w, cy), (width, cy), (200, 200, 200), 2)

        # Stoplines (Solid white lines)
        cv2.line(frame, (cx - lane_w, cy - lane_w - 5), (cx, cy - lane_w - 5), (255, 255, 255), 3) # North
        cv2.line(frame, (cx, cy + lane_w + 5), (cx + lane_w, cy + lane_w + 5), (255, 255, 255), 3) # South
        cv2.line(frame, (cx + lane_w + 5, cy - lane_w), (cx + lane_w + 5, cy), (255, 255, 255), 3) # East
        cv2.line(frame, (cx - lane_w - 5, cy), (cx - lane_w - 5, cy + lane_w), (255, 255, 255), 3) # West

        # Traffic Signal Phase (Simulate Red light on North/South for first 6 seconds, then green)
        ns_red = frame_idx < (fps * 6)
        ew_red = not ns_red

        # Draw Signal Status Indicators
        signal_color_ns = (0, 0, 255) if ns_red else (0, 255, 0)
        signal_color_ew = (0, 0, 255) if ew_red else (0, 255, 0)
        cv2.circle(frame, (cx - lane_w - 15, cy - lane_w - 15), 8, signal_color_ns, -1)
        cv2.circle(frame, (cx + lane_w + 15, cy + lane_w + 15), 8, signal_color_ew, -1)

        # Update and render vehicles
        # Group by approach and sort by d to update leader-follower
        for app in approaches:
            app_vehicles = [v for v in vehicles if v.approach == app]
            app_vehicles.sort(key=lambda v: v.d, reverse=True) # leader first
            is_red = ns_red if app in ["North", "South"] else ew_red

            leader_d = -999.0
            for v in app_vehicles:
                v.update(red_light=is_red, leader_d=leader_d)
                leader_d = v.d

                # Render vehicle body
                vx, vy, vw, vh = v.get_bbox_center(width, height)
                x1 = int(vx - vw // 2)
                y1 = int(vy - vh // 2)
                x2 = int(vx + vw // 2)
                y2 = int(vy + vh // 2)

                # Vehicle shadow
                cv2.rectangle(frame, (x1 + 3, y1 + 3), (x2 + 3, y2 + 3), (20, 20, 20), -1)
                # Vehicle body
                cv2.rectangle(frame, (x1, y1), (x2, y2), v.color, -1)
                # Windshield & details
                cv2.rectangle(frame, (x1 + 4, y1 + 4), (x2 - 4, y2 - 4), (60, 60, 70), 1)

        writer.write(frame)

    writer.release()
    print(f"Synthetic traffic video saved: {output_file} ({output_file.stat().st_size / (1024*1024):.2f} MB)")
    return output_file

if __name__ == "__main__":
    generate_traffic_clip()
