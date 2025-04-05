import threading

class SingaporeValidator:
    """
    Validates eye metrics against Singapore-specific, ethnicity-based thresholds
    in a thread-safe manner.
    """
    def __init__(self):
        """Initializes the validator with predefined thresholds and a lock."""
        # Thresholds based on provided information (adjust if necessary)
        # Using vertical saccade velocity as the example metric
        self._thresholds = {
            'chinese': {'vertical_saccade_velocity': 190}, # Example threshold
            'malay': {'vertical_saccade_velocity': 185},   # Example threshold
            'indian': {'vertical_saccade_velocity': 188},  # Example threshold
            'default': {'vertical_saccade_velocity': 200} # Default if ethnicity unknown/other
        }
        self._lock = threading.Lock()

    def validate(self, metrics, ethnicity):
        """
        Validates metrics against Singapore-specific thresholds for a given ethnicity.

        Args:
            metrics (dict): A dictionary containing calculated eye metrics.
                            Expected to have 'vertical_saccade_velocity'.
            ethnicity (str): The ethnicity of the patient (e.g., 'chinese', 'malay', 'indian').
                             Case-insensitive matching is performed.

        Returns:
            bool: True if the metric is within the acceptable range for the ethnicity,
                  False otherwise. Returns False if the required metric is missing.
        """
        if 'vertical_saccade_velocity' not in metrics:
            # Log this potentially?
            print("Warning: 'vertical_saccade_velocity' not found in metrics for validation.")
            return False

        metric_value = metrics['vertical_saccade_velocity']
        ethnicity_lower = ethnicity.lower() if ethnicity else 'default'

        with self._lock:
            # Get the specific threshold dictionary for the ethnicity, or default
            ethnic_thresholds = self._thresholds.get(ethnicity_lower, self._thresholds['default'])
            # Get the specific threshold value for the metric
            threshold_value = ethnic_thresholds.get('vertical_saccade_velocity', self._thresholds['default']['vertical_saccade_velocity'])

        # Assuming lower velocity is indicative of potential issues based on prompt context
        # (e.g., "vertical_saccade < threshold")
        is_valid = metric_value < threshold_value
        return is_valid

    def get_threshold(self, ethnicity, metric_key='vertical_saccade_velocity'):
        """
        Retrieves the threshold value for a specific ethnicity and metric.

        Args:
            ethnicity (str): The ethnicity.
            metric_key (str): The metric key (e.g., 'vertical_saccade_velocity').

        Returns:
            float or None: The threshold value, or None if not found.
        """
        ethnicity_lower = ethnicity.lower() if ethnicity else 'default'
        with self._lock:
            ethnic_thresholds = self._thresholds.get(ethnicity_lower, self._thresholds['default'])
            return ethnic_thresholds.get(metric_key)

# Example Usage (can be removed or kept for testing)
if __name__ == '__main__':
    validator = SingaporeValidator()
    sample_metrics_chinese_ok = {'vertical_saccade_velocity': 180}
    sample_metrics_chinese_fail = {'vertical_saccade_velocity': 195}
    sample_metrics_malay_ok = {'vertical_saccade_velocity': 170}
    sample_metrics_other = {'vertical_saccade_velocity': 199} # Should use default

    print(f"Chinese OK: {validator.validate(sample_metrics_chinese_ok, 'Chinese')}")
    print(f"Chinese Fail: {validator.validate(sample_metrics_chinese_fail, 'chinese')}")
    print(f"Malay OK: {validator.validate(sample_metrics_malay_ok, 'Malay')}")
    print(f"Other OK (default): {validator.validate(sample_metrics_other, 'caucasian')}")
    print(f"Missing Metric: {validator.validate({}, 'indian')}")

    print(f"Chinese Threshold: {validator.get_threshold('chinese')}")
    print(f"Default Threshold: {validator.get_threshold('unknown')}")
