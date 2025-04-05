import numpy as np
from ..utils.cycle_buffer import CycleBuffer # Relative import
import time

class PDDetector:
    """
    Assesses Parkinson's Disease risk based on ocular biomarkers and trends.
    """
    def __init__(self, history_days_velocity=7, history_days_fixation=30):
        """
        Initializes the PDDetector.

        Args:
            history_days_velocity (int): Number of days to track for velocity trends.
            history_days_fixation (int): Number of days to track for fixation trends.
        """
        # --- Tunable Weights for Biomarkers ---
        # These weights should be determined based on clinical research/data.
        # Example weights (summing doesn't necessarily have to be 1):
        self.weights = {
            'saccade_velocity_inv': 0.3,  # Inverse: Lower velocity might indicate higher risk
            'fixation_stability': 0.4,    # Higher instability might indicate higher risk
            'blink_rate_deviation': 0.2,  # Deviation from a baseline norm
            'vertical_saccade_ratio': 0.1 # Ratio of vertical to horizontal velocity (example)
            # Add other relevant weighted metrics
        }
        # Define a 'normal' or baseline blink rate (e.g., blinks per minute)
        self.baseline_blink_rate = 15 # Example baseline

        # --- Longitudinal Trend Tracking ---
        # Store daily averages. Maxlen assumes one entry per day.
        # In a real application, these would be loaded/saved from storage.
        self.daily_avg_saccade_velocity = CycleBuffer(maxlen=history_days_velocity)
        self.daily_avg_fixation_stability = CycleBuffer(maxlen=history_days_fixation)

        # Temporary storage for current day's metrics before averaging
        self._today_velocity_readings = []
        self._today_fixation_readings = []
        self._last_aggregation_timestamp = time.time()

    def _normalize_metric(self, value, min_val, max_val, invert=False):
        """Normalizes a metric value to a 0-1 range."""
        if max_val == min_val: return 0.5 # Avoid division by zero, return neutral value
        normalized = (value - min_val) / (max_val - min_val)
        normalized = np.clip(normalized, 0.0, 1.0) # Ensure value stays within [0, 1]
        return 1.0 - normalized if invert else normalized

    def _aggregate_daily_metrics(self):
        """Calculates and stores the average metrics for the completed day."""
        # This function would typically be called once daily or when loading the app
        # For simplicity here, we'll call it periodically if a day has passed.
        current_time = time.time()
        # Check if roughly a day (86400 seconds) has passed since last aggregation
        if current_time - self._last_aggregation_timestamp >= 86400:
            if self._today_velocity_readings:
                avg_vel = np.mean(self._today_velocity_readings)
                self.daily_avg_saccade_velocity.append(avg_vel)
                self._today_velocity_readings = [] # Reset for the new day

            if self._today_fixation_readings:
                avg_fix = np.mean(self._today_fixation_readings)
                self.daily_avg_fixation_stability.append(avg_fix)
                self._today_fixation_readings = [] # Reset for the new day

            self._last_aggregation_timestamp = current_time
            # In a real app: Save updated buffers to storage here
            print("Aggregated daily metrics and updated trend buffers.")


    def predict(self, metrics, ethnicity=None):
        """
        Calculates the PD risk score based on current metrics and historical trends.

        Args:
            metrics (dict): A dictionary of eye metrics from EyeTracker.
                            Expected keys depend on self.weights.
            ethnicity (str, optional): Patient's ethnicity (passed for potential future use
                                       or downstream components like BioNeMo).

        Returns:
            tuple: (risk_level, contributing_factors)
                   - risk_level (float): A score between 0.0 and 1.0 indicating risk.
                   - contributing_factors (dict): Metrics and their contribution to the score.
        """
        if not metrics:
            return 0.0, {} # No metrics, no risk

        # --- Aggregate daily metrics if needed ---
        # self._aggregate_daily_metrics() # Call this based on time passing

        # --- Store current readings for daily aggregation ---
        # (Only store if values are reasonable)
        if 'saccade_velocity' in metrics and metrics['saccade_velocity'] > 0:
             self._today_velocity_readings.append(metrics['saccade_velocity'])
        if 'fixation_stability' in metrics and metrics['fixation_stability'] >= 0:
             self._today_fixation_readings.append(metrics['fixation_stability'])


        # --- Calculate Score from Current Metrics ---
        risk_score = 0.0
        contributing_factors = {}

        # Define expected ranges for normalization (these need careful calibration)
        # Example ranges:
        VELOCITY_RANGE = (50, 400) # degrees/sec (example)
        STABILITY_RANGE = (0.01, 1.0) # Pixel std dev (example)
        BLINK_RATE_RANGE = (5, 30) # Blinks per minute (example)
        V_SACCADE_RATIO_RANGE = (0.1, 1.0) # Ratio (example)

        # 1. Saccade Velocity (Inverse relationship assumed: lower velocity -> higher risk)
        if 'saccade_velocity' in metrics:
            norm_vel_inv = self._normalize_metric(metrics['saccade_velocity'], VELOCITY_RANGE[0], VELOCITY_RANGE[1], invert=True)
            contribution = norm_vel_inv * self.weights['saccade_velocity_inv']
            risk_score += contribution
            contributing_factors['Saccade Velocity (Inv)'] = contribution

        # 2. Fixation Stability (Direct relationship assumed: higher instability -> higher risk)
        if 'fixation_stability' in metrics:
            norm_stab = self._normalize_metric(metrics['fixation_stability'], STABILITY_RANGE[0], STABILITY_RANGE[1], invert=False)
            contribution = norm_stab * self.weights['fixation_stability']
            risk_score += contribution
            contributing_factors['Fixation Stability'] = contribution

        # 3. Blink Rate Deviation (Deviation from baseline -> higher risk)
        if 'blink_rate' in metrics:
            deviation = abs(metrics['blink_rate'] - self.baseline_blink_rate)
            # Normalize deviation based on how far it *could* deviate within the range
            max_deviation = max(abs(BLINK_RATE_RANGE[0] - self.baseline_blink_rate), abs(BLINK_RATE_RANGE[1] - self.baseline_blink_rate))
            norm_blink_dev = self._normalize_metric(deviation, 0, max_deviation, invert=False)
            contribution = norm_blink_dev * self.weights['blink_rate_deviation']
            risk_score += contribution
            contributing_factors['Blink Rate Deviation'] = contribution

        # 4. Vertical Saccade Ratio (Example: Ratio of vertical to horizontal velocity)
        if 'vertical_saccade_velocity' in metrics and 'horizontal_saccade_velocity' in metrics and metrics['horizontal_saccade_velocity'] > 0:
            ratio = metrics['vertical_saccade_velocity'] / metrics['horizontal_saccade_velocity']
            norm_ratio = self._normalize_metric(ratio, V_SACCADE_RATIO_RANGE[0], V_SACCADE_RATIO_RANGE[1], invert=False) # Assuming higher ratio might be risk? Needs clinical basis.
            contribution = norm_ratio * self.weights['vertical_saccade_ratio']
            risk_score += contribution
            contributing_factors['Vertical Saccade Ratio'] = contribution


        # --- Incorporate Longitudinal Trends (Example Logic) ---
        # Compare current metrics to historical averages.
        # This logic is basic and needs refinement based on clinical understanding.

        # Velocity Trend: If current velocity is significantly lower than 7-day avg
        if len(self.daily_avg_saccade_velocity) > 0 and 'saccade_velocity' in metrics:
            avg_vel_hist = self.daily_avg_saccade_velocity.mean()
            if metrics['saccade_velocity'] < avg_vel_hist * 0.8: # e.g., 20% lower
                trend_factor = 0.1 # Add a small risk factor for negative trend
                risk_score += trend_factor
                contributing_factors['Velocity Trend (Low)'] = trend_factor

        # Stability Trend: If current stability is significantly higher than 30-day avg
        if len(self.daily_avg_fixation_stability) > 0 and 'fixation_stability' in metrics:
             avg_stab_hist = self.daily_avg_fixation_stability.mean()
             if metrics['fixation_stability'] > avg_stab_hist * 1.2: # e.g., 20% higher
                 trend_factor = 0.1 # Add a small risk factor for negative trend
                 risk_score += trend_factor
                 contributing_factors['Stability Trend (High)'] = trend_factor


        # --- Final Risk Score Calculation ---
        # Normalize the total score based on the maximum possible score from weights
        max_possible_score = sum(self.weights.values()) # Simplistic max score
        # Add potential trend factors to max possible if they are additive constants
        max_possible_score += 0.1 + 0.1 # Add max possible trend contributions

        final_risk = np.clip(risk_score / max_possible_score, 0.0, 1.0) if max_possible_score > 0 else 0.0

        # Sort factors by contribution for clarity
        sorted_factors = dict(sorted(contributing_factors.items(), key=lambda item: item[1], reverse=True))

        return final_risk, sorted_factors

    def load_history(self, velocity_history, fixation_history):
        """Loads historical data into the buffers."""
        # In a real app, this would load from a file/database via storage.py
        print(f"Loading history: {len(velocity_history)} velocity, {len(fixation_history)} fixation points.")
        for v in velocity_history:
            self.daily_avg_saccade_velocity.append(v)
        for f in fixation_history:
            self.daily_avg_fixation_stability.append(f)

    def get_history(self):
        """Returns the current historical data."""
        return {
            'velocity': self.daily_avg_saccade_velocity.get_all(),
            'fixation': self.daily_avg_fixation_stability.get_all()
        }


# Example Usage
if __name__ == '__main__':
    detector = PDDetector()

    # Simulate loading some history
    detector.load_history(
        velocity_history=[200, 210, 195, 205, 190, 185, 198], # Example 7 days velocity
        fixation_history=[0.3, 0.35, 0.4, 0.33] * 7 + [0.31, 0.36] # Example 30 days fixation
    )

    # Simulate receiving metrics from EyeTracker
    sample_metrics_low_risk = {
        'saccade_velocity': 250,
        'fixation_stability': 0.2,
        'blink_rate': 16,
        'vertical_saccade_velocity': 80,
        'horizontal_saccade_velocity': 240
    }

    sample_metrics_high_risk = {
        'saccade_velocity': 100, # Low velocity
        'fixation_stability': 0.8, # High instability
        'blink_rate': 5, # Low blink rate (large deviation)
        'vertical_saccade_velocity': 100,
        'horizontal_saccade_velocity': 110 # High V/H ratio
    }

    # Simulate metrics showing a negative trend compared to history
    sample_metrics_trend = {
        'saccade_velocity': 150, # Lower than historical avg (~197)
        'fixation_stability': 0.5, # Higher than historical avg (~0.34)
        'blink_rate': 15,
        'vertical_saccade_velocity': 60,
        'horizontal_saccade_velocity': 140
    }


    risk1, factors1 = detector.predict(sample_metrics_low_risk)
    print(f"\nLow Risk Scenario:")
    print(f"Calculated Risk: {risk1:.3f}")
    print("Contributing Factors:")
    for factor, value in factors1.items():
        print(f"  - {factor}: {value:.3f}")

    risk2, factors2 = detector.predict(sample_metrics_high_risk)
    print(f"\nHigh Risk Scenario:")
    print(f"Calculated Risk: {risk2:.3f}")
    print("Contributing Factors:")
    for factor, value in factors2.items():
        print(f"  - {factor}: {value:.3f}")

    risk3, factors3 = detector.predict(sample_metrics_trend)
    print(f"\nNegative Trend Scenario:")
    print(f"Calculated Risk: {risk3:.3f}")
    print("Contributing Factors:")
    for factor, value in factors3.items():
        print(f"  - {factor}: {value:.3f}")

    # print("\nCurrent History Buffers:")
    # print(f"  Velocity (last {detector.daily_avg_saccade_velocity.buffer.maxlen}): {detector.get_history()['velocity']}")
    # print(f"  Fixation (last {detector.daily_avg_fixation_stability.buffer.maxlen}): {detector.get_history()['fixation']}")
