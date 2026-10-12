"""
Traffic Light Vehicle Density Detector - Short-Term Traffic Prediction
Forecasts vehicle arrival rates and queue growth over a rolling horizon (5s - 15s)
using Exponentially Weighted Moving Average (EWMA) and approach kinematics.
"""
import time
from typing import Dict, List, Any
from detection.controller.traffic_state import TrafficStateSnapshot

class ShortTermTrafficPredictor:
    """
    Predicts near-future traffic demand S(t + delta_t) for each intersection approach.
    """
    def __init__(self, alpha: float = 0.35, forecast_horizon_sec: float = 10.0):
        self.alpha = alpha  # Smoothing factor for EWMA
        self.horizon_sec = forecast_horizon_sec
        self.arrival_rates_pcu_per_sec: Dict[str, float] = {}
        self.last_pcu: Dict[str, float] = {}
        self.last_timestamp: float = time.time()
        # Saturation flow rate per green lane: ~0.5 PCU/sec (1800 PCU/hr)
        self.saturation_discharge_rate = 0.5

    def predict(self, current_snapshot: TrafficStateSnapshot) -> Dict[str, Dict[str, float]]:
        """
        Calculates predicted traffic state S(t + horizon_sec).
        Returns:
            Dict mapping approach_id to predicted arrival rate, predicted PCU, and queue growth.
        """
        now = current_snapshot.timestamp
        dt = max(0.2, now - self.last_timestamp)
        predictions = {}

        for aid, app in current_snapshot.approaches.items():
            curr_pcu = app.total_pcu
            prev_pcu = self.last_pcu.get(aid, curr_pcu)

            # Instantaneous rate of vehicle change
            delta_pcu = max(0.0, curr_pcu - prev_pcu)
            inst_arrival_rate = delta_pcu / dt

            # Update EWMA arrival rate
            prev_rate = self.arrival_rates_pcu_per_sec.get(aid, 0.2)
            smoothed_rate = self.alpha * inst_arrival_rate + (1.0 - self.alpha) * prev_rate
            self.arrival_rates_pcu_per_sec[aid] = smoothed_rate
            self.last_pcu[aid] = curr_pcu

            # Discharge rate if signal is GREEN
            discharge = self.saturation_discharge_rate if app.current_signal == "GREEN" else 0.0

            # Predicted PCU at t + horizon_sec
            predicted_pcu = max(0.0, curr_pcu + (smoothed_rate - discharge) * self.horizon_sec)

            # Predicted queue spillover index
            queue_growth_rate = smoothed_rate if app.current_signal == "RED" else (smoothed_rate - discharge)

            predictions[aid] = {
                "arrival_rate_pcu_sec": round(smoothed_rate, 2),
                "predicted_pcu": round(predicted_pcu, 1),
                "queue_growth_rate": round(queue_growth_rate, 2),
                "is_approaching_spillover": bool(predicted_pcu > 12.0)
            }

        self.last_timestamp = now
        return predictions
