"""
Detection Kinematics & State Variable Estimation
Calculates approach velocity (km/h), inter-vehicle spatial headway (meters),
physical queue length behind the stop line, and aggregated PCU weights.
"""
from typing import List, Dict, Any, Tuple
import numpy as np
from detection.perception.tracker import VehicleTrack
from detection.perception.lane_geometry import IntersectionGeometry, Lane

class KinematicsEstimator:
    def __init__(self, geometry: IntersectionGeometry, queue_speed_threshold_kmh: float = 4.0):
        self.geometry = geometry
        self.queue_speed_threshold_kmh = queue_speed_threshold_kmh

    def update_kinematics(self, tracks: List[VehicleTrack]):
        """
        Updates velocity, lane assignment, and queue state for all active tracks.
        """
        for track in tracks:
            if len(track.history) < 2:
                continue

            # Assign lane based on current position
            curr_pos = track.history[-1]
            lane_id = self.geometry.assign_lane(curr_pos)
            if lane_id:
                track.lane_id = lane_id

            # Calculate velocity in km/h
            dt = max(0.001, track.timestamps[-1] - track.timestamps[-min(6, len(track.timestamps))])
            p_now = track.history[-1]
            p_past = track.history[-min(6, len(track.history))]

            dist_px = np.hypot(p_now[0] - p_past[0], p_now[1] - p_past[1])
            dist_meters = self.geometry.px_to_meters(dist_px)
            speed_mps = dist_meters / dt
            speed_kmh = speed_mps * 3.6

            # Smooth speed
            track.speed_kmh = 0.7 * track.speed_kmh + 0.3 * speed_kmh

            # Queue determination
            lane = self.geometry.lanes.get(track.lane_id)
            if lane:
                dist_to_sl = self.geometry.px_to_meters(lane.distance_to_stopline_px(curr_pos))
                # If speed < threshold and within 60 meters of stopline
                track.is_queued = (track.speed_kmh < self.queue_speed_threshold_kmh) and (dist_to_sl < 60.0)
            else:
                track.is_queued = False

    def compute_lane_aggregates(self, tracks: List[VehicleTrack], timestamp: float) -> List[Dict[str, Any]]:
        """
        Computes aggregate metrics per lane:
        - vehicle_count: Total vehicles in corridor
        - total_pcu: Sum of PCU weights in corridor
        - queue_length_m: Distance from stopline to furthest queued vehicle
        - avg_approach_speed: Mean velocity of moving approaching vehicles
        - lead_vehicle_headway: Distance of nearest vehicle to stopline
        """
        # Group tracks by lane
        lane_groups: Dict[str, List[VehicleTrack]] = {lane_id: [] for lane_id in self.geometry.lanes}
        for track in tracks:
            if track.lane_id in lane_groups:
                lane_groups[track.lane_id].append(track)

        aggregates = []

        for lane_id, lane_tracks in lane_groups.items():
            lane = self.geometry.lanes[lane_id]
            if not lane_tracks:
                aggregates.append({
                    "timestamp": timestamp,
                    "lane_id": lane_id,
                    "vehicle_count": 0,
                    "total_pcu": 0.0,
                    "queue_length_m": 0.0,
                    "avg_approach_speed": 0.0,
                    "lead_vehicle_headway": 0.0
                })
                continue

            # Total PCU
            total_pcu = sum(t.pcu_weight for t in lane_tracks)

            # Sort tracks by distance to stop line
            dists = []
            for t in lane_tracks:
                if len(t.history) > 0:
                    d_m = self.geometry.px_to_meters(lane.distance_to_stopline_px(t.history[-1]))
                    dists.append((d_m, t))
            dists.sort(key=lambda x: x[0])  # closest to stopline first

            # Inter-vehicle headways
            for i in range(len(dists)):
                if i == 0:
                    dists[i][1].headway_m = dists[i][0]  # distance to stop line
                else:
                    dists[i][1].headway_m = max(0.0, dists[i][0] - dists[i - 1][0])

            # Queue length: furthest queued vehicle distance
            queued_dists = [d for d, t in dists if t.is_queued]
            queue_length_m = max(queued_dists) if queued_dists else 0.0

            # Average approach speed (for vehicles currently moving)
            moving_speeds = [t.speed_kmh for _, t in dists if not t.is_queued]
            avg_speed = float(np.mean(moving_speeds)) if moving_speeds else 0.0

            lead_headway = dists[0][0] if dists else 0.0

            aggregates.append({
                "timestamp": timestamp,
                "lane_id": lane_id,
                "vehicle_count": len(lane_tracks),
                "total_pcu": round(total_pcu, 2),
                "queue_length_m": round(queue_length_m, 2),
                "avg_approach_speed": round(avg_speed, 2),
                "lead_vehicle_headway": round(lead_headway, 2)
            })

        return aggregates
