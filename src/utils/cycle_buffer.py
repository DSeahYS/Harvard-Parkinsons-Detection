import time
from collections import deque
import numpy as np

class CycleBuffer:
    """
    Manages data collection over fixed time cycles (e.g., 15 seconds).
    Placeholder implementation.
    """
    def __init__(self, cycle_duration=15):
        self.cycle_duration = cycle_duration
        self.current_cycle_data = []
        self.last_cycle_start_time = time.time()
        print(f"Initialized CycleBuffer with duration: {self.cycle_duration}s")

    def add_data(self, metrics):
        """Adds new metrics data to the current cycle."""
        current_time = time.time()
        if metrics:
            self.current_cycle_data.append(metrics)

        # Check if cycle duration has passed
        if current_time - self.last_cycle_start_time >= self.cycle_duration:
            # Process the completed cycle
            analysis = self.process_cycle()
            # Reset for the next cycle
            self.current_cycle_data = []
            self.last_cycle_start_time = current_time
            return analysis
        return None

    def process_cycle(self):
        """Analyzes the data collected over the completed cycle."""
        if not self.current_cycle_data:
            return {"message": "No data in cycle"}

        print(f"Processing cycle with {len(self.current_cycle_data)} data points.")
        # Placeholder: Calculate average metrics over the cycle
        avg_metrics = {}
        keys_to_average = ['avg_saccade_velocity', 'avg_vertical_saccade_velocity', 'fixation_stability', 'avg_ear']

        for key in keys_to_average:
            values = [m.get(key) for m in self.current_cycle_data if m and key in m]
            if values:
                avg_metrics[f"cycle_{key}"] = np.mean(values)

        # Add blink rate calculation if needed (similar to pd_detector)
        # ...

        print(f"Cycle analysis results: {avg_metrics}")
        return avg_metrics

    def is_cycle_analysis_enabled(self):
        """Checks if cycle analysis should be active (e.g., based on UI checkbox)."""
        # This might need to be linked to the dashboard's cycle_var
        # For now, assume it's always potentially active if initialized.
        return True
