import numpy as np
from scipy.signal import welch # Assuming welch is used for hippus calculation

class MedicationDetector:
    """
    Analyzes pupil dynamics (hippus) to infer Levodopa medication efficacy (ON/OFF state),
    with adjustments for ethnicity based on Singapore clinical observations.
    """
    def __init__(self, ethnicity='default', fs=120):
        """
        Initializes the detector.

        Args:
            ethnicity (str): The patient's ethnicity ('chinese', 'malay', 'indian', 'default').
                             Used to adjust hippus thresholds.
            fs (int): Sampling frequency of the pupil data in Hz.
        """
        self.ethnicity = ethnicity.lower() if ethnicity else 'default'
        self.fs = fs # Sampling frequency needed for Welch method
        print(f"Initialized MedicationDetector for ethnicity: {self.ethnicity}, fs: {self.fs}Hz")

    def _calculate_hippus(self, pupil_diameter_ts):
        """
        Calculates the power of pupillary hippus in the 0.5-2Hz range.
        Placeholder implementation - requires actual pupil diameter time series data.

        Args:
            pupil_diameter_ts (np.array): Time series of pupil diameter measurements.

        Returns:
            float: Power spectral density in the target frequency band, or 0.0 if calculation fails.
        """
        if pupil_diameter_ts is None or len(pupil_diameter_ts) < self.fs * 2: # Need at least 2 seconds of data
             print("Warning: Insufficient data for hippus calculation.")
             return 0.0 # Not enough data

        try:
            # Calculate Power Spectral Density using Welch method
            freqs, psd = welch(pupil_diameter_ts, fs=self.fs, nperseg=self.fs*2) # Use 2-second segments

            # Find indices corresponding to the 0.5-2Hz band
            idx_band = np.where((freqs >= 0.5) & (freqs <= 2.0))[0]

            if len(idx_band) == 0:
                print("Warning: Target frequency band (0.5-2Hz) not found in PSD.")
                return 0.0

            # Calculate the average power in the band
            hippus_power = np.mean(psd[idx_band])
            print(f"Calculated Hippus Power (0.5-2Hz): {hippus_power:.4f}") # Debug print
            return hippus_power

        except Exception as e:
            print(f"Error calculating hippus power: {e}")
            return 0.0


    def detect_state(self, pupil_diameter_ts):
        """
        Detects medication efficacy ('Medication Effective' or 'Needs Dose')
        based on hippus power and ethnicity-specific thresholds.

        Args:
            pupil_diameter_ts (np.array): Time series of pupil diameter measurements.

        Returns:
            str: The detected medication state.
        """
        hippus_power = self._calculate_hippus(pupil_diameter_ts)

        # Ethnicity-specific threshold adjustment
        # From Search Result: Malay patients show 18% larger hippus amplitude (implies lower power when ON)
        # The feedback threshold logic seems reversed based on the research note (lower power = ON).
        # Assuming lower power indicates 'ON' state (effective medication).
        if self.ethnicity == 'malay':
            # Malay patients might have naturally higher hippus, so the 'ON' threshold might be slightly higher?
            # Or does the 18% larger amplitude mean the *reduction* when ON is less pronounced?
            # Sticking to feedback's direct threshold values for now, but noting potential ambiguity.
            on_threshold = 0.18 # Threshold below which state is considered 'ON'
        else:
            on_threshold = 0.15 # Default threshold

        print(f"Ethnicity: {self.ethnicity}, Hippus Power: {hippus_power:.4f}, Threshold: {on_threshold}") # Debug print

        if hippus_power < on_threshold and hippus_power > 0: # Check > 0 to ensure calculation was valid
            return "Medication Effective"
        else:
            return "Needs Dose / Ineffective" # More descriptive than just "Needs Dose"

# Example Usage (requires pupil data)
# detector = MedicationDetector(ethnicity='malay')
# pupil_data = np.random.rand(10 * 120) # 10 seconds of dummy data at 120Hz
# state = detector.detect_state(pupil_data)
# print(f"Detected State: {state}")