"""
Traffic Light Vehicle Density Detector - Traffic State Representation
Computes the dynamic traffic state S(t) per intersection approach:
S(t) = [Vehicle_Count, Total_PCU, Queue_Length_m, Avg_Approach_Speed, Red_Wait_Time]
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np

@dataclass
class ApproachState:
    approach_id: str                      # e.g. "North", "South", "East", "West"
    vehicle_count: int = 0                # Total vehicles currently detected
    total_pcu: float = 0.0                # Sum of PCU weights
    queued_vehicles: int = 0              # Vehicles with speed < threshold
    queue_length_m: float = 0.0           # Physical queue extent in meters
    avg_speed_kmh: float = 0.0            # Average velocity of approaching vehicles
    waiting_time_sec: float = 0.0         # Continuous waiting duration under Red phase
    current_signal: str = "RED"           # "GREEN", "YELLOW", "RED"
    last_updated: float = field(default_factory=time.time)

    @property
    def density_level(self) -> str:
        if self.total_pcu >= 12.0 or self.queue_length_m >= 35.0:
            return "CRITICAL"
        elif self.total_pcu >= 6.0 or self.queue_length_m >= 18.0:
            return "HIGH"
        elif self.total_pcu >= 2.0:
            return "MEDIUM"
        else:
            return "LOW"

@dataclass
class TrafficStateSnapshot:
    timestamp: float
    approaches: Dict[str, ApproachState] = field(default_factory=dict)
    total_intersection_pcu: float = 0.0
    active_phase: str = "North-South"     # Current active green phase
    critical_approach: str = "North"      # Approach with highest pressure/queue

class TrafficStateManager:
    """
    Maintains and updates the operational traffic state S(t) across all approaches.
    """
    def __init__(self, approach_ids: Optional[List[str]] = None):
        self.approach_ids = approach_ids or ["North", "South", "East", "West"]
        self.approaches: Dict[str, ApproachState] = {
            aid: ApproachState(approach_id=aid) for aid in self.approach_ids
        }
        self.phase_start_time: float = time.time()
        self.current_green_phase: str = "North-South"

    def update_from_tracks(
        self,
        active_tracks: List[Any],
        timestamp: float,
        current_signals: Dict[str, str],
        px_per_m: float = 12.0
    ) -> TrafficStateSnapshot:
        """
        Updates S(t) using active tracks from ByteTrack.
        """
        # Reset counters per approach
        counts = {aid: 0 for aid in self.approach_ids}
        pcus = {aid: 0.0 for aid in self.approach_ids}
        queued = {aid: 0 for aid in self.approach_ids}
        speeds = {aid: [] for aid in self.approach_ids}
        max_ys = {aid: 0.0 for aid in self.approach_ids}
        min_ys = {aid: 999999.0 for aid in self.approach_ids}

        for t in active_tracks:
            # Map lane/direction to approach
            aid = self._resolve_approach(t)
            if aid in counts:
                counts[aid] += 1
                pcus[aid] += getattr(t, "pcu_weight", 1.0)
                spd = getattr(t, "speed_kmh", 0.0)
                speeds[aid].append(spd)

                # Check queued condition (speed < 4 km/h)
                if spd < 4.0:
                    queued[aid] += 1

                # Bbox vertical extent for queue length
                if hasattr(t, "last_bbox") and len(t.last_bbox) == 4:
                    cy = (t.last_bbox[1] + t.last_bbox[3]) / 2.0
                    max_ys[aid] = max(max_ys[aid], cy)
                    min_ys[aid] = min(min_ys[aid], cy)

        total_pcu = sum(pcus.values())
        critical_app = max(pcus.keys(), key=lambda k: pcus[k]) if pcus else self.approach_ids[0]

        # Update approach states
        for aid in self.approach_ids:
            app = self.approaches[aid]
            app.vehicle_count = counts[aid]
            app.total_pcu = round(pcus[aid], 1)
            app.queued_vehicles = queued[aid]
            app.avg_speed_kmh = round(float(np.mean(speeds[aid])), 1) if speeds[aid] else 0.0

            # Queue length calculation (span between front and rear queued vehicles in meters)
            if queued[aid] > 0 and min_ys[aid] < max_ys[aid]:
                app.queue_length_m = round(float(max_ys[aid] - min_ys[aid]) / px_per_m, 1)
            else:
                app.queue_length_m = 0.0

            # Signal state and wait time accumulation
            sig = current_signals.get(aid, "RED")
            app.current_signal = sig
            if sig == "RED":
                app.waiting_time_sec += (timestamp - app.last_updated)
            else:
                app.waiting_time_sec = 0.0  # Reset upon green

            app.last_updated = timestamp

        return TrafficStateSnapshot(
            timestamp=timestamp,
            approaches={aid: self.approaches[aid] for aid in self.approach_ids},
            total_intersection_pcu=round(total_pcu, 1),
            critical_approach=critical_app
        )

    def _resolve_approach(self, track: Any) -> str:
        """Resolves track position or lane string to one of the 4 intersection approaches."""
        lane = str(getattr(track, "lane_id", "")).lower()
        if "north" in lane:
            return "North"
        elif "south" in lane:
            return "South"
        elif "east" in lane:
            return "East"
        elif "west" in lane:
            return "West"

        # Spatial coordinate fallback based on centroid if lane string not assigned
        if hasattr(track, "last_bbox") and len(track.last_bbox) == 4:
            cx = (track.last_bbox[0] + track.last_bbox[2]) / 2.0
            cy = (track.last_bbox[1] + track.last_bbox[3]) / 2.0
            # Standard single camera overhead perspective: top half is North approach, bottom half is South approach
            if cy < 360:
                return "North"
            else:
                return "South"

        return "North"
