"""
Clinical thresholds for Parkinson's disease detection based on eye movement metrics.
These values are derived from scientific literature and clinical studies.
"""

PD_THRESHOLDS = {
    # Saccadic movement thresholds
    'saccade_velocity_min': 400.0,  # degrees/second - below this may indicate PD
    'saccade_velocity_max': 600.0,  # degrees/second - normal range upper limit
    'vertical_saccade_velocity_min': 200.0, # degrees/second - below this may indicate PD (Vertical)

    # Fixation stability threshold
    'fixation_stability': 0.3,      # variance in position - above this may indicate PD
    
    # Blink rate thresholds
    'blink_rate_min': 8,            # blinks per minute - below may indicate PD
    'blink_rate_max': 21,           # blinks per minute - above may indicate PD
    
    # Antisaccade error rate
    'antisaccade_error': 2.1,        # errors per second - above may indicate PD
}

# Singapore-specific vertical saccade thresholds (degrees/second)
# Lower values indicate potential abnormality (slower vertical saccades)
# Note: These are thresholds *below which* might indicate PD, consistent with PD_THRESHOLDS['vertical_saccade_velocity_min']
# However, the implementation in eye_tracker checks for velocity *above* a threshold.
# This dictionary defines the *threshold* value itself, which can be used for either comparison.
# The eye_tracker implementation uses these values as a *lower bound* for *normal* fast saccades.
VERTICAL_SACCADE_THRESHOLDS = {
    'chinese': 190.0,
    'malay': 185.0,
    'indian': 188.0,
    'default': 200.0  # Default threshold from PD_THRESHOLDS
}


# Clinical interpretation guidelines
RISK_LEVELS = {
    'low': (0.0, 0.3),
    'medium': (0.3, 0.7),
    'high': (0.7, 1.0)
}

# Recommendations based on risk levels
RECOMMENDATIONS = {
    'low': "No immediate action required. Consider follow-up in 12 months.",
    'medium': "Recommend follow-up in 6 months. Consider additional neurological testing.",
    'high': "Prompt referral to neurologist recommended. Consider DaTscan or other confirmatory tests."
}

