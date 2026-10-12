"""
Detection Lane Geometry & Spatial Calibration
Defines approach corridors, lane polygons, stop lines, and pixel-to-meter calibration
to accurately partition traffic flow across intersection arms.
Supports interactive JSON serialization for custom camera calibration.
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2

def _ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def _segments_intersect(A: Tuple[float, float], B: Tuple[float, float],
                        C: Tuple[float, float], D: Tuple[float, float]) -> bool:
    """Returns True if line segment AB intersects line segment CD."""
    return (_ccw(A, C, D) != _ccw(B, C, D)) and (_ccw(A, B, C) != _ccw(A, B, D))

class Lane:
    def __init__(
        self,
        lane_id: str,
        polygon: List[Tuple[int, int]],
        stopline: Tuple[Tuple[int, int], Tuple[int, int]],
        direction: str = "Southbound",
        approach: Optional[str] = None
    ):
        self.lane_id = lane_id
        self.polygon = np.array(polygon, dtype=np.int32)
        self.stopline = stopline  # ((x1, y1), (x2, y2))
        self.direction = direction
        # Approach: "North", "South", "East", "West"
        if approach:
            self.approach = approach
        else:
            self.approach = lane_id.split("_")[0] if "_" in lane_id else lane_id

    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Returns True if point (x, y) is inside the lane polygon."""
        pt = (float(point[0]), float(point[1]))
        return cv2.pointPolygonTest(self.polygon, pt, False) >= 0

    def distance_to_stopline_px(self, point: Tuple[float, float]) -> float:
        """Calculates Euclidean distance from point to stopline center."""
        sl_cx = (self.stopline[0][0] + self.stopline[1][0]) / 2.0
        sl_cy = (self.stopline[0][1] + self.stopline[1][1]) / 2.0
        return float(np.hypot(point[0] - sl_cx, point[1] - sl_cy))

    def has_crossed_stopline(self, pt_prev: Tuple[float, float], pt_curr: Tuple[float, float]) -> bool:
        """Checks if vehicle displacement vector from pt_prev to pt_curr crossed the stopline."""
        sl_p1 = (float(self.stopline[0][0]), float(self.stopline[0][1]))
        sl_p2 = (float(self.stopline[1][0]), float(self.stopline[1][1]))
        return _segments_intersect(pt_prev, pt_curr, sl_p1, sl_p2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "polygon": self.polygon.tolist(),
            "stopline": [list(self.stopline[0]), list(self.stopline[1])],
            "direction": self.direction,
            "approach": self.approach
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Lane":
        poly = [tuple(p) for p in data["polygon"]]
        sl = (tuple(data["stopline"][0]), tuple(data["stopline"][1]))
        return cls(
            lane_id=data["lane_id"],
            polygon=poly,
            stopline=sl,
            direction=data.get("direction", "Southbound"),
            approach=data.get("approach")
        )

class IntersectionGeometry:
    def __init__(self, pixels_per_meter: float = 12.0, resolution: Tuple[int, int] = (1280, 720)):
        self.pixels_per_meter = pixels_per_meter
        self.resolution = resolution
        self.lanes: Dict[str, Lane] = {}

    def add_lane(self, lane: Lane):
        self.lanes[lane.lane_id] = lane

    def assign_lane(self, center_point: Tuple[float, float]) -> Optional[str]:
        for lane_id, lane in self.lanes.items():
            if lane.contains_point(center_point):
                return lane_id
        return None

    def px_to_meters(self, pixels: float) -> float:
        return pixels / max(1.0, self.pixels_per_meter)

    def meters_to_px(self, meters: float) -> float:
        return meters * self.pixels_per_meter

    def save_to_json(self, filepath: str):
        """Serializes calibrated intersection geometry to a JSON file."""
        data = {
            "version": "2.0",
            "resolution": list(self.resolution),
            "pixels_per_meter": self.pixels_per_meter,
            "lanes": [lane.to_dict() for lane in self.lanes.values()]
        }
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str) -> "IntersectionGeometry":
        """Loads calibrated intersection geometry from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        res = tuple(data.get("resolution", [1280, 720]))
        ppm = float(data.get("pixels_per_meter", 12.0))
        geom = cls(pixels_per_meter=ppm, resolution=res)

        for l_data in data.get("lanes", []):
            lane = Lane.from_dict(l_data)
            geom.add_lane(lane)

        return geom

    @classmethod
    def create_standard_4way(cls, width: int = 1280, height: int = 720, px_per_m: float = 12.0) -> "IntersectionGeometry":
        """
        Creates a standard 4-way intersection layout for top-down corridor monitoring.
        """
        geom = cls(pixels_per_meter=px_per_m, resolution=(width, height))
        cx, cy = width // 2, height // 2
        lane_w = 60

        # North Approach (Incoming from top towards center stopline)
        poly_north_l1 = [(cx - lane_w, 0), (cx, 0), (cx, cy - 80), (cx - lane_w, cy - 80)]
        stop_north = ((cx - lane_w, cy - 80), (cx, cy - 80))
        geom.add_lane(Lane("North_L1", poly_north_l1, stop_north, direction="Southbound", approach="North"))

        # South Approach (Incoming from bottom towards center stopline)
        poly_south_l1 = [(cx, cy + 80), (cx + lane_w, cy + 80), (cx + lane_w, height), (cx, height)]
        stop_south = ((cx, cy + 80), (cx + lane_w, cy + 80))
        geom.add_lane(Lane("South_L1", poly_south_l1, stop_south, direction="Northbound", approach="South"))

        # East Approach (Incoming from right towards center stopline)
        poly_east_l1 = [(cx + 80, cy - lane_w), (width, cy - lane_w), (width, cy), (cx + 80, cy)]
        stop_east = ((cx + 80, cy - lane_w), (cx + 80, cy))
        geom.add_lane(Lane("East_L1", poly_east_l1, stop_east, direction="Westbound", approach="East"))

        # West Approach (Incoming from left towards center stopline)
        poly_west_l1 = [(0, cy), (cx - 80, cy), (cx - 80, cy + lane_w), (0, cy + lane_w)]
        stop_west = ((0, cy), (cx - 80, cy))
        geom.add_lane(Lane("West_L1", poly_west_l1, stop_west, direction="Eastbound", approach="West"))

        return geom
