"""
Traffic Light Vehicle Density Detector - Tracking Layer
Integrates ByteTrack for continuous vehicle tracking across frames.
Maintains historical trajectories, calculates motion vectors, and estimates queue states.
"""
import time
from collections import deque
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from detection.perception.detector import PCU_WEIGHTS, TRAFFIC_NAME_CANONICAL

class VehicleTrack:
    def __init__(self, track_id: int, class_id: int, class_name: str, pcu_weight: float):
        self.track_id = track_id
        self.class_id = class_id
        self.class_name = class_name
        self.pcu_weight = pcu_weight
        self.history = deque(maxlen=30)  # Last 30 points (peach trajectory trail)
        self.timestamps = deque(maxlen=30)
        self.last_bbox: List[float] = []
        self.confidence: float = 0.0
        self.lane_id: str = "Approach_L1"
        self.speed_kmh: float = 0.0
        self.headway_m: float = 0.0
        self.is_queued: bool = False
        self.missed_frames: int = 0

    def update(self, bbox: List[float], conf: float, timestamp: float, lane_id: str = "Approach_L1"):
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.history.append((cx, cy))
        self.timestamps.append(timestamp)
        self.last_bbox = bbox
        self.confidence = conf
        self.lane_id = lane_id
        self.missed_frames = 0

    def get_motion_vector(self) -> Optional[Tuple[float, float]]:
        """Returns (dx, dy) motion vector predicted from recent positions."""
        if len(self.history) < 2:
            return None
        p1 = self.history[-1]
        p0 = self.history[-min(5, len(self.history))]
        dt = max(0.001, self.timestamps[-1] - self.timestamps[-min(5, len(self.timestamps))])
        vx = (p1[0] - p0[0]) / dt
        vy = (p1[1] - p0[1]) / dt
        return (vx, vy)

class ByteTrackManager:
    """
    Manages active vehicle tracks, associations, trajectory trails, and kinematics.
    """
    def __init__(self, max_missed_frames: int = 15):
        self.active_tracks: Dict[int, VehicleTrack] = {}
        self.max_missed_frames = max_missed_frames

    def update_with_ultralytics_results(
        self,
        results,
        timestamp: float,
        is_visdrone: bool = False,
        names_dict: Optional[Dict[int, str]] = None
    ) -> List[VehicleTrack]:
        """
        Updates tracks from Ultralytics track results.
        """
        seen_ids = set()
        active_list = []

        if not results or len(results) == 0 or results[0].boxes is None:
            self._age_tracks(seen_ids)
            return []

        boxes = results[0].boxes
        if boxes.id is None:
            self._age_tracks(seen_ids)
            return []

        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        # Fallback names map
        res_names = names_dict or getattr(results[0], "names", {})

        for i, track_id in enumerate(ids):
            seen_ids.add(track_id)
            c_id = int(clss[i])
            raw_name = res_names.get(c_id, "car").lower().strip()
            c_name = TRAFFIC_NAME_CANONICAL.get(raw_name, "car")
            pcu = PCU_WEIGHTS.get(c_name, 1.0)
            bbox = [float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])]
            conf = float(confs[i])

            if track_id not in self.active_tracks:
                self.active_tracks[track_id] = VehicleTrack(track_id, c_id, c_name, pcu)

            track = self.active_tracks[track_id]
            track.update(bbox, conf, timestamp)
            active_list.append(track)

        self._age_tracks(seen_ids)
        return active_list

    def _age_tracks(self, seen_ids: set):
        to_delete = []
        for tid, track in self.active_tracks.items():
            if tid not in seen_ids:
                track.missed_frames += 1
                if track.missed_frames > self.max_missed_frames:
                    to_delete.append(tid)
        for tid in to_delete:
            del self.active_tracks[tid]
