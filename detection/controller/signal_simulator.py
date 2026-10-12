"""
Traffic Light Vehicle Density Detector - Software Traffic Signal Simulator
Renders real-time graphical traffic lights (RED / YELLOW / GREEN) on video frames
with glowing lenses, live countdown timers, and approach density meters.
"""
import cv2
import numpy as np
import time
from typing import Dict, Any, Tuple, Optional
from detection.controller.adaptive_controller import ControllerDecision
from detection.controller.traffic_state import TrafficStateSnapshot

# Color Palette (BGR)
COLOR_RED_ACTIVE = (30, 30, 255)
COLOR_RED_DIM = (15, 15, 60)
COLOR_YELLOW_ACTIVE = (0, 220, 255)
COLOR_YELLOW_DIM = (0, 60, 70)
COLOR_GREEN_ACTIVE = (50, 255, 60)
COLOR_GREEN_DIM = (10, 60, 15)

COLOR_HOUSING = (20, 20, 25)
COLOR_BORDER = (80, 80, 95)
COLOR_TEXT_MAIN = (255, 255, 255)
COLOR_TEXT_DIM = (180, 180, 190)

class SoftwareTrafficSignalSimulator:
    """
    Renders the Software Traffic Signal Simulator overlay onto the video stream.
    Displays physical-style 3-aspect traffic light heads with countdown timers.
    """
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height

    def render_overlay(
        self,
        frame: np.ndarray,
        decision: ControllerDecision,
        snapshot: TrafficStateSnapshot,
        arduino_status: str = "SIMULATION MODE",
        inference_fps: float = 30.0,
        geometry: Optional[Any] = None
    ) -> np.ndarray:
        """
        Draws the complete HUD including header, signal heads, countdowns, density gauges,
        calibrated approach polygons, stop lines, and emergency/gap-out alerts.
        """
        h, w = frame.shape[:2]

        # 0. Draw Calibrated Lane Polygons and Stop Lines (if geometry provided)
        if geometry and hasattr(geometry, "lanes"):
            self._render_geometry_overlays(frame, geometry)

        # 1. Top Branding Header Bar / Emergency Alert Banner
        if decision.is_emergency_preemption:
            # Flashing Emergency Bar (Red/Amber)
            bar_color = (25, 20, 180) if int(time.time() * 3) % 2 == 0 else (10, 140, 240)
            cv2.rectangle(frame, (0, 0), (w, 55), bar_color, -1)
            cv2.line(frame, (0, 55), (w, 55), (255, 255, 255), 2)
            cv2.putText(frame, f"[EMERGENCY PREEMPTION] {decision.emergency_corridor} PRIORITY GREEN WAVE",
                        (20, 36), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"HOLD: {decision.countdown_seconds:02d}s", (w - 180, 36),
                        cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (0, 0), (w, 55), (15, 15, 22), -1)
            cv2.line(frame, (0, 55), (w, 55), (0, 160, 255), 2)  # Amber accent line

            # Title
            cv2.putText(frame, "TRAFFIC LIGHT VEHICLE DENSITY DETECTOR", (20, 34),
                        cv2.FONT_HERSHEY_DUPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

            # Telemetry info on right of top bar
            gap_tag = " | [GAP-OUT]" if decision.is_gapout else ""
            telemetry_str = f"FPS: {inference_fps:.1f} | Phase: {decision.active_phase.value}{gap_tag}"
            cv2.putText(frame, telemetry_str, (w - 560, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 220, 240), 1, cv2.LINE_AA)

        # 2. Main Traffic Signal Panel (Top Right Floating HUD)
        panel_w = 440
        panel_h = 240
        px = w - panel_w - 20
        py = 70

        # Draw semi-transparent background card
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (20, 18, 25), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (60, 60, 75), 1)

        # Panel Header
        if decision.is_emergency_preemption:
            cv2.putText(frame, "PRIORITY EMERGENCY CONTROL", (px + 15, py + 26),
                        cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 100, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "SOFTWARE TRAFFIC SIGNAL SIMULATOR", (px + 15, py + 26),
                        cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 210, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, f"[{arduino_status}]", (px + panel_w - 180, py + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 255, 120) if "CONNECTED" in arduino_status else (160, 160, 170), 1, cv2.LINE_AA)
        cv2.line(frame, (px + 15, py + 34), (px + panel_w - 15, py + 34), (50, 50, 65), 1)

        # Render 2 Master Signal Heads (North-South Approach vs East-West Approach)
        self._draw_signal_head(
            frame,
            x=px + 25,
            y=py + 45,
            label="N-S APPROACH",
            current_state=decision.signals.get("North", "RED"),
            countdown=decision.countdown_seconds if decision.active_direction in ["NS", "ALL_RED"] else 0,
            density_pcu=(snapshot.approaches.get("North", {}).total_pcu if "North" in snapshot.approaches else 0.0) +
                        (snapshot.approaches.get("South", {}).total_pcu if "South" in snapshot.approaches else 0.0),
            queue_len=max(snapshot.approaches.get("North", {}).queue_length_m if "North" in snapshot.approaches else 0.0,
                          snapshot.approaches.get("South", {}).queue_length_m if "South" in snapshot.approaches else 0.0)
        )

        self._draw_signal_head(
            frame,
            x=px + 235,
            y=py + 45,
            label="E-W APPROACH",
            current_state=decision.signals.get("East", "RED"),
            countdown=decision.countdown_seconds if decision.active_direction in ["EW", "ALL_RED"] else 0,
            density_pcu=(snapshot.approaches.get("East", {}).total_pcu if "East" in snapshot.approaches else 0.0) +
                        (snapshot.approaches.get("West", {}).total_pcu if "West" in snapshot.approaches else 0.0),
            queue_len=max(snapshot.approaches.get("East", {}).queue_length_m if "East" in snapshot.approaches else 0.0,
                          snapshot.approaches.get("West", {}).queue_length_m if "West" in snapshot.approaches else 0.0)
        )

        # 3. Bottom Controller Decision Bar (Includes Gap-Out / Headway telemetry)
        d_bar_y = py + panel_h - 32
        cv2.line(frame, (px + 15, d_bar_y), (px + panel_w - 15, d_bar_y), (50, 50, 65), 1)

        if decision.is_gapout:
            cv2.putText(frame, "[GAP-OUT] Queue Cleared Early -> Next Phase",
                        (px + 15, d_bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1, cv2.LINE_AA)
        elif decision.headway_seconds > 0:
            cv2.putText(frame, f"Headway: {decision.headway_seconds:.1f}s (Gap Limit {decision.gap_threshold_seconds:.1f}s) | {decision.decision_reason[:25]}...",
                        (px + 15, d_bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 220, 230), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"Decision: {decision.decision_reason[:42]}...",
                        (px + 15, d_bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

        # 4. Floating Intersection Density Summary (Bottom Left)
        self._render_density_card(frame, snapshot, decision)

        return frame

    def _render_geometry_overlays(self, frame: np.ndarray, geometry: Any):
        """Renders subtle calibrated lane polygons, direction arrows, and stop lines."""
        app_cols = {
            "North": (255, 180, 40),
            "South": (80, 230, 90),
            "East":  (40, 180, 255),
            "West":  (220, 80, 240)
        }
        for lane_id, lane in geometry.lanes.items():
            app = getattr(lane, "approach", lane_id.split("_")[0])
            col = app_cols.get(app, (180, 180, 180))
            poly = lane.polygon

            # Translucent shaded lane corridor
            overlay = frame.copy()
            cv2.fillPoly(overlay, [poly], col)
            cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
            cv2.polylines(frame, [poly], True, col, 1, cv2.LINE_AA)

            # Prominent Virtual Stop Line
            sl_p1 = (int(lane.stopline[0][0]), int(lane.stopline[0][1]))
            sl_p2 = (int(lane.stopline[1][0]), int(lane.stopline[1][1]))
            cv2.line(frame, sl_p1, sl_p2, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, sl_p1, 3, (0, 255, 255), -1)
            cv2.circle(frame, sl_p2, 3, (0, 255, 255), -1)

    def _draw_signal_head(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        label: str,
        current_state: str,
        countdown: int,
        density_pcu: float,
        queue_len: float
    ):
        """Draws a vertical 3-lens traffic signal with housing, countdown, and metrics."""
        box_w = 46
        box_h = 120
        radius = 14

        # Housing
        cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), COLOR_HOUSING, -1)
        cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), COLOR_BORDER, 2)

        # Red Lens (Top)
        c_red = COLOR_RED_ACTIVE if current_state == "RED" else COLOR_RED_DIM
        cv2.circle(frame, (x + box_w // 2, y + 22), radius, c_red, -1)
        if current_state == "RED":
            cv2.circle(frame, (x + box_w // 2, y + 22), radius + 2, (80, 80, 255), 1)

        # Yellow Lens (Middle)
        c_yel = COLOR_YELLOW_ACTIVE if current_state == "YELLOW" else COLOR_YELLOW_DIM
        cv2.circle(frame, (x + box_w // 2, y + 60), radius, c_yel, -1)
        if current_state == "YELLOW":
            cv2.circle(frame, (x + box_w // 2, y + 60), radius + 2, (100, 240, 255), 1)

        # Green Lens (Bottom)
        c_grn = COLOR_GREEN_ACTIVE if current_state == "GREEN" else COLOR_GREEN_DIM
        cv2.circle(frame, (x + box_w // 2, y + 98), radius, c_grn, -1)
        if current_state == "GREEN":
            cv2.circle(frame, (x + box_w // 2, y + 98), radius + 2, (100, 255, 120), 1)

        # Side Information (Label, Countdown, Density)
        sx = x + box_w + 12
        cv2.putText(frame, label, (sx, y + 16), cv2.FONT_HERSHEY_DUPLEX, 0.42, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)

        # Countdown Timer Box
        timer_text = f"{countdown:02d}s" if countdown > 0 else "--"
        t_color = COLOR_GREEN_ACTIVE if current_state == "GREEN" else (COLOR_YELLOW_ACTIVE if current_state == "YELLOW" else COLOR_RED_ACTIVE)
        cv2.putText(frame, timer_text, (sx, y + 48), cv2.FONT_HERSHEY_DUPLEX, 0.85, t_color, 2, cv2.LINE_AA)

        # Density stats
        cv2.putText(frame, f"Density: {density_pcu:.1f} PCU", (sx, y + 74), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 200, 215), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Queue: {queue_len:.1f} m", (sx, y + 94), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 200, 215), 1, cv2.LINE_AA)

        # Mini Density Bar
        bar_w = 100
        bar_h = 6
        bar_fill = int(min(1.0, density_pcu / 15.0) * bar_w)
        cv2.rectangle(frame, (sx, y + 104), (sx + bar_w, y + 104 + bar_h), (40, 40, 50), -1)
        bar_color = (60, 230, 80) if density_pcu < 5.0 else ((0, 200, 255) if density_pcu < 10.0 else (40, 40, 240))
        if bar_fill > 0:
            cv2.rectangle(frame, (sx, y + 104), (sx + bar_fill, y + 104 + bar_h), bar_color, -1)

    def _render_density_card(self, frame: np.ndarray, snapshot: TrafficStateSnapshot, decision: ControllerDecision):
        """Renders 4-approach summary in bottom left."""
        h, w = frame.shape[:2]
        cw = 310
        ch = 135
        cx = 20
        cy = h - ch - 38  # Leave room for bottom control bar

        overlay = frame.copy()
        cv2.rectangle(overlay, (cx, cy), (cx + cw, cy + ch), (20, 18, 25), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (60, 60, 75), 1)

        cv2.putText(frame, "APPROACH DENSITY METRICS", (cx + 12, cy + 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, (0, 210, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (cx + 12, cy + 28), (cx + cw - 12, cy + 28), (50, 50, 65), 1)

        row_y = cy + 48
        for aid in ["North", "South", "East", "West"]:
            app = snapshot.approaches.get(aid)
            if not app:
                continue
            sig = app.current_signal
            sig_col = COLOR_GREEN_ACTIVE if sig == "GREEN" else (COLOR_YELLOW_ACTIVE if sig == "YELLOW" else COLOR_RED_ACTIVE)
            cv2.circle(frame, (cx + 18, row_y - 4), 5, sig_col, -1)

            line_str = f"{aid:<5} | PCU: {app.total_pcu:4.1f} | Q: {app.queue_length_m:4.1f}m | {app.density_level}"
            cv2.putText(frame, line_str, (cx + 30, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 235), 1, cv2.LINE_AA)
            row_y += 24
