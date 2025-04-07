import numpy as np
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RiskLevel(Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"

class PDDetector:
    """
    Assesses Parkinson's Disease risk by combining ocular biomarkers
    and genetic risk scores based on specified thresholds and weights.
    """
    def __init__(self, baseline_blink_rate=18, # Typical average blink rate
                       saccade_norm_range=(50, 600), # deg/s, wider range for normalization
                       stability_norm_range=(0.1, 3.0), # deg, wider range for normalization
                       blink_norm_range=(5, 35) # bpm, wider range for normalization
                       ):
        """
        Initializes the PDDetector.

        Args:
            baseline_blink_rate (float): Expected normal blink rate (blinks per minute) for deviation calculation.
            saccade_norm_range (tuple): Min/Max expected saccade velocity (deg/s) for normalization.
            stability_norm_range (tuple): Min/Max expected fixation stability (deg) for normalization.
            blink_norm_range (tuple): Min/Max expected blink rate (bpm) for normalization.
        """
        # --- Weights for Ocular Risk Score Component ---
        # As specified: Saccade Vel (Inv): 0.6, Fixation Stab: 0.25, Blink Rate Dev: 0.15
        self.weights = {
            'saccade_velocity_inv': 0.6,
            'fixation_stability': 0.25,
            'blink_rate_deviation': 0.15,
        }
        self.baseline_blink_rate = baseline_blink_rate

        # --- Normalization Ranges ---
        # Used to scale metrics before applying weights for the ocular score component
        self.metric_ranges = {
            'saccade_velocity': saccade_norm_range,
            'fixation_stability': stability_norm_range,
            'blink_rate': blink_norm_range
        }

        # --- Thresholds for Final Classification ---
        # Based directly on the provided requirements table
        self.thresholds = {
            'saccade_velocity_moderate': 400, # Below this is moderate or high risk (ocular)
            'saccade_velocity_high': 300,     # Below this is high risk (ocular)
            'genetic_score_moderate': 1.5,    # Above this is moderate or high risk (genetic)
            'genetic_score_high': 2.5,        # Above this is high risk (genetic)
        }

        logging.info(f"PDDetector initialized. Weights: {self.weights}. Thresholds: {self.thresholds}")

    def _normalize_metric(self, value, metric_key, invert=False):
        """Normalizes a metric value to a 0-1 range based on predefined ranges."""
        if value is None:
            logging.warning(f"Normalization skipped for {metric_key}: Received None value. Returning 0.5 (neutral).")
            return 0.5 # Neutral value for missing data

        if metric_key not in self.metric_ranges:
            logging.warning(f"Normalization range not defined for metric: {metric_key}. Returning 0.5.")
            return 0.5

        min_val, max_val = self.metric_ranges[metric_key]

        if max_val == min_val:
            logging.warning(f"Min and Max range values are equal for {metric_key}. Returning 0.5.")
            return 0.5 # Avoid division by zero

        try:
            value_float = float(value)
            normalized = (value_float - min_val) / (max_val - min_val)
            normalized = np.clip(normalized, 0.0, 1.0) # Ensure value stays within [0, 1]
            result = 1.0 - normalized if invert else normalized
            # logging.debug(f"Normalized {metric_key}: {value} -> {result:.3f} (Invert: {invert})")
            return result
        except (ValueError, TypeError) as e:
            logging.error(f"Error normalizing {metric_key} with value '{value}': {e}. Returning 0.5.")
            return 0.5


    def _calculate_ocular_risk_score(self, ocular_metrics):
        """
        Calculates a combined score based *only* on the weighted ocular metrics.
        This score is primarily for understanding the contribution of eye movements,
        the final classification uses the threshold table.
        """
        if not ocular_metrics:
            return 0.0, {}

        ocular_score = 0.0
        contributing_factors = {}

        # Metric keys expected from EyeTracker
        vel_key = 'saccade_velocity_deg_s'
        stab_key = 'fixation_stability_deg' # Expecting stability in degrees now
        blink_key = 'blink_rate_bpm'

        # 1. Saccade Velocity (Inverse relationship: lower velocity -> higher risk contribution)
        norm_vel_inv = self._normalize_metric(ocular_metrics.get(vel_key), 'saccade_velocity', invert=True)
        contribution_vel = norm_vel_inv * self.weights['saccade_velocity_inv']
        ocular_score += contribution_vel
        contributing_factors['Saccade Velocity (Inv)'] = contribution_vel

        # 2. Fixation Stability (Direct relationship: higher instability -> higher risk contribution)
        norm_stab = self._normalize_metric(ocular_metrics.get(stab_key), 'fixation_stability', invert=False)
        contribution_stab = norm_stab * self.weights['fixation_stability']
        ocular_score += contribution_stab
        contributing_factors['Fixation Stability'] = contribution_stab

        # 3. Blink Rate Deviation (Deviation from baseline -> higher risk contribution)
        blink_rate = ocular_metrics.get(blink_key)
        if blink_rate is not None:
            deviation = abs(blink_rate - self.baseline_blink_rate)
            # Normalize the deviation itself relative to max possible deviation from baseline
            min_rate, max_rate = self.metric_ranges['blink_rate']
            max_deviation = max(abs(min_rate - self.baseline_blink_rate), abs(max_rate - self.baseline_blink_rate))
            if max_deviation > 0:
                 norm_blink_dev = min(deviation / max_deviation, 1.0) # Clip at 1.0
            else:
                 norm_blink_dev = 0.0
            contribution_blink = norm_blink_dev * self.weights['blink_rate_deviation']
            ocular_score += contribution_blink
            contributing_factors['Blink Rate Deviation'] = contribution_blink
        else:
            contributing_factors['Blink Rate Deviation'] = 0.0 # No contribution if missing

        # Normalize the final ocular score based on the sum of weights
        max_possible_score = sum(self.weights.values())
        normalized_ocular_score = np.clip(ocular_score / max_possible_score, 0.0, 1.0) if max_possible_score > 0 else 0.0

        # Sort factors by contribution
        sorted_factors = dict(sorted(contributing_factors.items(), key=lambda item: item[1], reverse=True))

        return normalized_ocular_score, sorted_factors

    def assess_risk(self, ocular_metrics, genetic_risk_score):
        """
        Assesses the PD risk level based on ocular metrics and genetic score
        using the defined threshold table.

        Args:
            ocular_metrics (dict): Dictionary of metrics from EyeTracker.
                                   Must contain 'saccade_velocity_deg_s'.
                                   Optionally 'fixation_stability_deg', 'blink_rate_bpm'.
            genetic_risk_score (float): The risk score multiplier from BioNeMo (e.g., 1.0, 1.5, 2.5).

        Returns:
            tuple: (risk_level, classification_reason, ocular_score, contributing_factors)
                   - risk_level (RiskLevel): Enum indicating Low, Moderate, or High risk.
                   - classification_reason (str): Explanation of why the level was chosen.
                   - ocular_score (float): The calculated score based on weighted ocular metrics (0-1).
                   - contributing_factors (dict): Ocular metrics and their contribution score.
        """
        if ocular_metrics is None:
            logging.warning("Assess risk called with no ocular metrics.")
            # Cannot classify without saccade velocity
            return RiskLevel.LOW, "Insufficient ocular data", 0.0, {}
        if genetic_risk_score is None:
             logging.warning("Assess risk called with no genetic risk score. Treating as baseline (1.0).")
             genetic_risk_score = 1.0 # Assume baseline if missing? Or return error?

        # Calculate the ocular score component (for contributing factors info)
        ocular_score, contributing_factors = self._calculate_ocular_risk_score(ocular_metrics)

        # Get the key metric for classification: Saccade Velocity
        saccade_velocity = ocular_metrics.get('saccade_velocity_deg_s')

        if saccade_velocity is None:
            logging.warning("Saccade velocity missing from ocular metrics. Cannot classify risk.")
            return RiskLevel.LOW, "Missing saccade velocity data", ocular_score, contributing_factors

        # --- Apply Threshold Logic ---
        sacc_thresh_high = self.thresholds['saccade_velocity_high']
        sacc_thresh_mod = self.thresholds['saccade_velocity_moderate']
        gen_thresh_high = self.thresholds['genetic_score_high']
        gen_thresh_mod = self.thresholds['genetic_score_moderate']

        risk_level = RiskLevel.LOW # Default to low
        reason = ""

        # Check for High Risk conditions first
        is_high_saccade = saccade_velocity < sacc_thresh_high
        is_high_genetic = genetic_risk_score > gen_thresh_high

        if is_high_saccade or is_high_genetic:
            risk_level = RiskLevel.HIGH
            reasons = []
            if is_high_saccade: reasons.append(f"Saccade velocity ({saccade_velocity:.1f} deg/s) < {sacc_thresh_high}")
            if is_high_genetic: reasons.append(f"Genetic score ({genetic_risk_score:.2f}x) > {gen_thresh_high}")
            reason = "High Risk: " + " and ".join(reasons)
        else:
            # Check for Moderate Risk conditions
            is_moderate_saccade = saccade_velocity < sacc_thresh_mod
            is_moderate_genetic = genetic_risk_score > gen_thresh_mod

            if is_moderate_saccade or is_moderate_genetic:
                risk_level = RiskLevel.MODERATE
                reasons = []
                if is_moderate_saccade: reasons.append(f"Saccade velocity ({saccade_velocity:.1f} deg/s) < {sacc_thresh_mod}")
                if is_moderate_genetic: reasons.append(f"Genetic score ({genetic_risk_score:.2f}x) > {gen_thresh_mod}")
                reason = "Moderate Risk: " + " and ".join(reasons)
            else:
                # If neither High nor Moderate conditions met, it's Low Risk
                risk_level = RiskLevel.LOW
                reason = f"Low Risk: Saccade velocity ({saccade_velocity:.1f} deg/s) >= {sacc_thresh_mod} and Genetic score ({genetic_risk_score:.2f}x) <= {gen_thresh_mod}"

        logging.info(f"Risk Assessment: Level={risk_level.value}. Reason: {reason}")
        return risk_level, reason, ocular_score, contributing_factors


# Example Usage
if __name__ == '__main__':
    detector = PDDetector()

    print("\n--- Test Cases ---")

    # Low Risk Scenario
    metrics_low = {'saccade_velocity_deg_s': 450, 'fixation_stability_deg': 0.5, 'blink_rate_bpm': 20}
    genetic_low = 1.2
    level, reason, ocular, factors = detector.assess_risk(metrics_low, genetic_low)
    print(f"Scenario: Low Ocular, Low Genetic")
    print(f"  Metrics: {metrics_low}")
    print(f"  Genetic Score: {genetic_low}")
    print(f"  Result: Level={level.value}, OcularScore={ocular:.3f}")
    print(f"  Reason: {reason}")
    # print(f"  Factors: {factors}")

    # Moderate Risk (Low Saccade)
    metrics_mod_sacc = {'saccade_velocity_deg_s': 350, 'fixation_stability_deg': 0.6, 'blink_rate_bpm': 18}
    genetic_mod_sacc = 1.3
    level, reason, ocular, factors = detector.assess_risk(metrics_mod_sacc, genetic_mod_sacc)
    print(f"\nScenario: Moderate Ocular (Saccade), Low Genetic")
    print(f"  Metrics: {metrics_mod_sacc}")
    print(f"  Genetic Score: {genetic_mod_sacc}")
    print(f"  Result: Level={level.value}, OcularScore={ocular:.3f}")
    print(f"  Reason: {reason}")

    # Moderate Risk (High Genetic)
    metrics_mod_gen = {'saccade_velocity_deg_s': 410, 'fixation_stability_deg': 0.7, 'blink_rate_bpm': 22}
    genetic_mod_gen = 1.8
    level, reason, ocular, factors = detector.assess_risk(metrics_mod_gen, genetic_mod_gen)
    print(f"\nScenario: Low Ocular, Moderate Genetic")
    print(f"  Metrics: {metrics_mod_gen}")
    print(f"  Genetic Score: {genetic_mod_gen}")
    print(f"  Result: Level={level.value}, OcularScore={ocular:.3f}")
    print(f"  Reason: {reason}")

    # High Risk (Very Low Saccade)
    metrics_high_sacc = {'saccade_velocity_deg_s': 250, 'fixation_stability_deg': 1.0, 'blink_rate_bpm': 15}
    genetic_high_sacc = 1.4 # Genetic score doesn't need to be high if saccade is very low
    level, reason, ocular, factors = detector.assess_risk(metrics_high_sacc, genetic_high_sacc)
    print(f"\nScenario: High Ocular (Saccade), Low Genetic")
    print(f"  Metrics: {metrics_high_sacc}")
    print(f"  Genetic Score: {genetic_high_sacc}")
    print(f"  Result: Level={level.value}, OcularScore={ocular:.3f}")
    print(f"  Reason: {reason}")

    # High Risk (Very High Genetic)
    metrics_high_gen = {'saccade_velocity_deg_s': 380, 'fixation_stability_deg': 0.8, 'blink_rate_bpm': 12}
    genetic_high_gen = 2.8
    level, reason, ocular, factors = detector.assess_risk(metrics_high_gen, genetic_high_gen)
    print(f"\nScenario: Moderate Ocular (Saccade), High Genetic")
    print(f"  Metrics: {metrics_high_gen}")
    print(f"  Genetic Score: {genetic_high_gen}")
    print(f"  Result: Level={level.value}, OcularScore={ocular:.3f}")
    print(f"  Reason: {reason}")

    # High Risk (Both Moderate/High)
    metrics_high_both = {'saccade_velocity_deg_s': 320, 'fixation_stability_deg': 1.5, 'blink_rate_bpm': 28}
    genetic_high_both = 1.9
    level, reason, ocular, factors = detector.assess_risk(metrics_high_both, genetic_high_both)
    print(f"\nScenario: Moderate Ocular (Saccade), Moderate Genetic")
    print(f"  Metrics: {metrics_high_both}")
    print(f"  Genetic Score: {genetic_high_both}")
    print(f"  Result: Level={level.value}, OcularScore={ocular:.3f}")
    print(f"  Reason: {reason}")

    # Missing Saccade Velocity
    metrics_missing = {'fixation_stability_deg': 0.5, 'blink_rate_bpm': 20}
    genetic_missing = 1.2
    level, reason, ocular, factors = detector.assess_risk(metrics_missing, genetic_missing)
    print(f"\nScenario: Missing Saccade Velocity")
    print(f"  Metrics: {metrics_missing}")
    print(f"  Genetic Score: {genetic_missing}")
    print(f"  Result: Level={level.value}, OcularScore={ocular:.3f}")
    print(f"  Reason: {reason}")
