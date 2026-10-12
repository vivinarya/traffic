"""
Traffic Light Vehicle Density Detector - Adaptive Traffic Controller
Implements Webster & Pressure-Based dynamic phase and green-time allocation.
Features:
  - Green Gap-Out Logic (early green termination when vehicle headway drops)
  - Emergency Vehicle Preemption (EVP: instant priority green wave for ambulances/police)
  - Dynamic Webster Green Allocation: min 10s to max 45s based on live PCU pressure
"""
import time
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
from detection.controller.traffic_state import TrafficStateSnapshot

class SignalPhase(Enum):
    PHASE_NS_GREEN = "North-South GREEN"
    PHASE_NS_YELLOW = "North-South YELLOW"
    PHASE_ALL_RED = "ALL RED (Clearance)"
    PHASE_EW_GREEN = "East-West GREEN"
    PHASE_EW_YELLOW = "East-West YELLOW"

@dataclass
class ControllerDecision:
    timestamp: float
    active_phase: SignalPhase
    active_direction: str                 # "NS", "EW", "ALL_RED"
    signals: Dict[str, str]               # {"North": "GREEN", "South": "GREEN", "East": "RED", "West": "RED"}
    countdown_seconds: int                # Remaining seconds in active state
    allocated_green_seconds: int          # Target green time allocated for this phase
    decision_reason: str                  # Explanation of controller decision
    phase_progress_ratio: float           # 0.0 to 1.0
    is_gapout: bool = False               # True if currently triggered by gap-out
    is_emergency_preemption: bool = False # True if emergency green wave active
    emergency_corridor: Optional[str] = None
    headway_seconds: float = 0.0
    gap_threshold_seconds: float = 2.5

class AdaptiveTrafficSignalController:
    """
    Closed-loop adaptive controller computing real-time green allocations
    based on live PCU density, queue lengths, predicted arrivals, and emergency triggers.
    """
    def __init__(
        self,
        min_green_sec: int = 10,
        max_green_sec: int = 45,
        yellow_sec: int = 3,
        all_red_sec: int = 2,
        gap_threshold_sec: float = 2.5,
        min_green_before_gapout: float = 6.0
    ):
        self.min_green = min_green_sec
        self.max_green = max_green_sec
        self.yellow_time = yellow_sec
        self.all_red_time = all_red_sec

        # Gap-Out Parameters
        self.gap_threshold_sec = gap_threshold_sec
        self.min_green_before_gapout = min_green_before_gapout
        self.last_departure_time: Dict[str, float] = {"NS": time.time(), "EW": time.time()}
        self.is_gapout_triggered = False
        self.last_gapout_reason = ""
        self.current_headway = 0.0

        # Emergency Vehicle Preemption (EVP) Parameters
        self.emergency_preemption_active = False
        self.emergency_corridor: Optional[str] = None
        self.emergency_expiry_time = 0.0

        # State machine variables
        self.current_phase = SignalPhase.PHASE_NS_GREEN
        self.phase_start_time = time.time()
        self.current_allocated_duration = 15.0  # Default initial green
        self.next_target_phase = SignalPhase.PHASE_EW_GREEN

    def notify_vehicle_departure(self, approach: str, timestamp: Optional[float] = None):
        """Called whenever a vehicle crosses the stop line on an approach."""
        ts = timestamp or time.time()
        corridor = "NS" if approach in ["North", "South"] else "EW"
        self.last_departure_time[corridor] = ts

    def trigger_emergency_preemption(self, direction: str = "NS", hold_duration_sec: float = 14.0):
        """Immediately grants high-priority green wave to the emergency corridor."""
        self.emergency_preemption_active = True
        self.emergency_corridor = direction.upper()
        self.emergency_expiry_time = time.time() + hold_duration_sec
        print(f"\n[ALERT] [CONTROLLER] EMERGENCY PREEMPTION TRIGGERED for {self.emergency_corridor} corridor ({hold_duration_sec}s hold)")

    def clear_emergency_preemption(self):
        """Cancels emergency preemption and resumes standard adaptive control."""
        if self.emergency_preemption_active:
            print("\n[CONTROLLER] Emergency corridor clear. Resuming standard adaptive cycle.")
            self.emergency_preemption_active = False
            self.emergency_corridor = None

    def compute_decision(
        self,
        snapshot: TrafficStateSnapshot,
        predictions: Dict[str, Dict[str, float]]
    ) -> ControllerDecision:
        """
        Evaluates current traffic state, emergency preemption, and gap-out thresholds.
        """
        now = snapshot.timestamp
        elapsed = now - self.phase_start_time
        remaining = max(0.0, self.current_allocated_duration - elapsed)

        # -------------------------------------------------------------
        # 1. Evaluate Emergency Vehicle Preemption
        # -------------------------------------------------------------
        if self.emergency_preemption_active:
            if now >= self.emergency_expiry_time:
                self.clear_emergency_preemption()
            else:
                target_green = SignalPhase.PHASE_NS_GREEN if self.emergency_corridor == "NS" else SignalPhase.PHASE_EW_GREEN
                if self.current_phase == target_green:
                    # Maintain Green for emergency corridor
                    remaining = max(2.0, self.emergency_expiry_time - now)
                    self.current_allocated_duration = elapsed + remaining
                else:
                    # Switch rapidly to yellow -> all-red -> emergency green
                    if self.current_phase in [SignalPhase.PHASE_NS_GREEN, SignalPhase.PHASE_EW_GREEN]:
                        self.current_phase = SignalPhase.PHASE_NS_YELLOW if self.current_phase == SignalPhase.PHASE_NS_GREEN else SignalPhase.PHASE_EW_YELLOW
                        self.current_allocated_duration = 2.0
                        self.phase_start_time = now
                        elapsed = 0.0
                        remaining = 2.0
                    elif self.current_phase in [SignalPhase.PHASE_NS_YELLOW, SignalPhase.PHASE_EW_YELLOW] and remaining <= 0.0:
                        self.current_phase = SignalPhase.PHASE_ALL_RED
                        self.current_allocated_duration = 1.5
                        self.phase_start_time = now
                        elapsed = 0.0
                        remaining = 1.5
                    elif self.current_phase == SignalPhase.PHASE_ALL_RED and remaining <= 0.0:
                        self.current_phase = target_green
                        self.current_allocated_duration = max(5.0, self.emergency_expiry_time - now)
                        self.phase_start_time = now
                        elapsed = 0.0
                        remaining = self.current_allocated_duration

        # -------------------------------------------------------------
        # 2. Evaluate Green Gap-Out (Only if not in emergency mode)
        # -------------------------------------------------------------
        elif self.current_phase in [SignalPhase.PHASE_NS_GREEN, SignalPhase.PHASE_EW_GREEN]:
            curr_corridor = "NS" if self.current_phase == SignalPhase.PHASE_NS_GREEN else "EW"
            comp_corridor = "EW" if curr_corridor == "NS" else "NS"

            # Compute time since last departure
            last_dep = self.last_departure_time.get(curr_corridor, self.phase_start_time)
            headway = max(0.0, now - last_dep)
            self.current_headway = headway

            # Competing traffic demand
            apps = snapshot.approaches
            if comp_corridor == "EW":
                comp_pcu = (apps.get("East", {}).total_pcu if "East" in apps else 0.0) + \
                           (apps.get("West", {}).total_pcu if "West" in apps else 0.0)
                active_pcu = (apps.get("North", {}).total_pcu if "North" in apps else 0.0) + \
                             (apps.get("South", {}).total_pcu if "South" in apps else 0.0)
            else:
                comp_pcu = (apps.get("North", {}).total_pcu if "North" in apps else 0.0) + \
                           (apps.get("South", {}).total_pcu if "South" in apps else 0.0)
                active_pcu = (apps.get("East", {}).total_pcu if "East" in apps else 0.0) + \
                             (apps.get("West", {}).total_pcu if "West" in apps else 0.0)

            # Trigger gap-out if min_green has elapsed and traffic has gapped out while opposing queue waits
            if elapsed >= self.min_green_before_gapout:
                if (headway >= self.gap_threshold_sec or active_pcu < 0.5) and comp_pcu > 0.5:
                    self.is_gapout_triggered = True
                    self.last_gapout_reason = (f"Gap-Out Triggered: Headway {headway:.1f}s >= {self.gap_threshold_sec}s "
                                              f"(Active {active_pcu:.1f} PCU, Competing {comp_pcu:.1f} PCU waiting)")
                    print(f"\n[GAP-OUT] {self.last_gapout_reason}")
                    self._transition_to_next_phase(snapshot, predictions)
                    elapsed = 0.0
                    remaining = self.current_allocated_duration
        else:
            self.current_headway = 0.0

        # -------------------------------------------------------------
        # 3. Regular Timer Expiry Transitions
        # -------------------------------------------------------------
        if remaining <= 0.0 and not self.emergency_preemption_active:
            self.is_gapout_triggered = False
            self._transition_to_next_phase(snapshot, predictions)
            elapsed = 0.0
            remaining = self.current_allocated_duration

        # 4. Derive individual approach light states
        signals, active_dir = self._resolve_light_colors(self.current_phase)

        # 5. Build Decision Rationale
        if self.emergency_preemption_active:
            reason = f"[EMERGENCY PREEMPTION] Priority Green Wave for {self.emergency_corridor} Corridor"
        elif self.is_gapout_triggered:
            reason = self.last_gapout_reason
        else:
            reason = self._build_decision_rationale(self.current_phase, snapshot, predictions)

        progress = min(1.0, elapsed / max(0.1, self.current_allocated_duration))

        return ControllerDecision(
            timestamp=now,
            active_phase=self.current_phase,
            active_direction=active_dir,
            signals=signals,
            countdown_seconds=int(round(remaining)),
            allocated_green_seconds=int(self.current_allocated_duration),
            decision_reason=reason,
            phase_progress_ratio=round(progress, 2),
            is_gapout=self.is_gapout_triggered,
            is_emergency_preemption=self.emergency_preemption_active,
            emergency_corridor=self.emergency_corridor,
            headway_seconds=round(self.current_headway, 1),
            gap_threshold_seconds=self.gap_threshold_sec
        )

    def _transition_to_next_phase(
        self,
        snapshot: TrafficStateSnapshot,
        predictions: Dict[str, Dict[str, float]]
    ):
        """Advances the signal state machine."""
        now = time.time()

        if self.current_phase == SignalPhase.PHASE_NS_GREEN:
            self.current_phase = SignalPhase.PHASE_NS_YELLOW
            self.current_allocated_duration = float(self.yellow_time)
            self.next_target_phase = SignalPhase.PHASE_EW_GREEN

        elif self.current_phase == SignalPhase.PHASE_NS_YELLOW:
            self.current_phase = SignalPhase.PHASE_ALL_RED
            self.current_allocated_duration = float(self.all_red_time)
            self.next_target_phase = SignalPhase.PHASE_EW_GREEN

        elif self.current_phase == SignalPhase.PHASE_EW_GREEN:
            self.current_phase = SignalPhase.PHASE_EW_YELLOW
            self.current_allocated_duration = float(self.yellow_time)
            self.next_target_phase = SignalPhase.PHASE_NS_GREEN

        elif self.current_phase == SignalPhase.PHASE_EW_YELLOW:
            self.current_phase = SignalPhase.PHASE_ALL_RED
            self.current_allocated_duration = float(self.all_red_time)
            self.next_target_phase = SignalPhase.PHASE_NS_GREEN

        elif self.current_phase == SignalPhase.PHASE_ALL_RED:
            # Entering new green phase: compute dynamic green allocation!
            target = self.next_target_phase
            self.current_phase = target
            allocated_green = self._calculate_dynamic_green(target, snapshot, predictions)
            self.current_allocated_duration = float(allocated_green)
            # Reset departure time for newly active corridor
            active_corr = "NS" if target == SignalPhase.PHASE_NS_GREEN else "EW"
            self.last_departure_time[active_corr] = now

        self.phase_start_time = now

    def _calculate_dynamic_green(
        self,
        target_phase: SignalPhase,
        snapshot: TrafficStateSnapshot,
        predictions: Dict[str, Dict[str, float]]
    ) -> int:
        """
        GREEN TIME DECISION ALGORITHM:
        Dynamically calculates optimal green time based on approach pressure:
        Pressure P = PCU_active + 0.8 * Predicted_PCU + 0.12 * Red_Wait_Time
        """
        apps = snapshot.approaches

        # NS pressure
        ns_pcu = (apps.get("North", {}).total_pcu if "North" in apps else 0.0) + \
                 (apps.get("South", {}).total_pcu if "South" in apps else 0.0)
        ns_wait = max(apps.get("North", {}).waiting_time_sec if "North" in apps else 0.0,
                      apps.get("South", {}).waiting_time_sec if "South" in apps else 0.0)
        ns_pred = predictions.get("North", {}).get("predicted_pcu", 0.0) + predictions.get("South", {}).get("predicted_pcu", 0.0)
        p_ns = ns_pcu + 0.8 * ns_pred + 0.12 * ns_wait

        # EW pressure
        ew_pcu = (apps.get("East", {}).total_pcu if "East" in apps else 0.0) + \
                 (apps.get("West", {}).total_pcu if "West" in apps else 0.0)
        ew_wait = max(apps.get("East", {}).waiting_time_sec if "East" in apps else 0.0,
                      apps.get("West", {}).waiting_time_sec if "West" in apps else 0.0)
        ew_pred = predictions.get("East", {}).get("predicted_pcu", 0.0) + predictions.get("West", {}).get("predicted_pcu", 0.0)
        p_ew = ew_pcu + 0.8 * ew_pred + 0.12 * ew_wait

        if target_phase == SignalPhase.PHASE_NS_GREEN:
            target_pressure = p_ns
            competing_pressure = max(1.0, p_ew)
        else:
            target_pressure = p_ew
            competing_pressure = max(1.0, p_ns)

        # Baseline allocation proportional to demand pressure
        pressure_ratio = target_pressure / (target_pressure + competing_pressure)
        raw_green = self.min_green + (self.max_green - self.min_green) * (pressure_ratio * 1.6)

        # Clamp between min and max green
        final_green = int(round(max(float(self.min_green), min(float(self.max_green), raw_green))))
        return final_green

    def _resolve_light_colors(self, phase: SignalPhase) -> Tuple[Dict[str, str], str]:
        """Maps phase enum to 4 individual approach light states."""
        if phase == SignalPhase.PHASE_NS_GREEN:
            return {"North": "GREEN", "South": "GREEN", "East": "RED", "West": "RED"}, "NS"
        elif phase == SignalPhase.PHASE_NS_YELLOW:
            return {"North": "YELLOW", "South": "YELLOW", "East": "RED", "West": "RED"}, "NS"
        elif phase == SignalPhase.PHASE_EW_GREEN:
            return {"North": "RED", "South": "RED", "East": "GREEN", "West": "GREEN"}, "EW"
        elif phase == SignalPhase.PHASE_EW_YELLOW:
            return {"North": "RED", "South": "RED", "East": "YELLOW", "West": "YELLOW"}, "EW"
        else: # ALL_RED
            return {"North": "RED", "South": "RED", "East": "RED", "West": "RED"}, "ALL_RED"

    def _build_decision_rationale(
        self,
        phase: SignalPhase,
        snapshot: TrafficStateSnapshot,
        predictions: Dict[str, Dict[str, float]]
    ) -> str:
        if phase in [SignalPhase.PHASE_NS_GREEN, SignalPhase.PHASE_EW_GREEN]:
            p_name = "North-South" if phase == SignalPhase.PHASE_NS_GREEN else "East-West"
            return f"Adaptive Green Active for {p_name} approach (Density Demand={snapshot.total_intersection_pcu:.1f} PCU)"
        elif phase in [SignalPhase.PHASE_NS_YELLOW, SignalPhase.PHASE_EW_YELLOW]:
            return "Yellow Clearance interval active (Preparing phase switch)"
        else:
            return "All-Red intersection clearance safety interval"
