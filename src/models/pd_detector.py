import numpy as np
from collections import deque
from src.data.thresholds import PD_THRESHOLDS, RECOMMENDATIONS, RISK_LEVELS # Import more from thresholds

# Singapore-specific referral logic
SINGAPORE_REFERRALS = {
    'low': "Polyclinic follow-up recommended.",
    'medium': "Referral to NNI General Neurology recommended.",
    'high': "Urgent referral to NNI Movement Disorders Clinic recommended."
}

def get_referral(risk_level):
    """Determines the Singapore-specific referral pathway based on risk level."""
    if risk_level < RISK_LEVELS['low'][1]: # Use thresholds from imported RISK_LEVELS
        level = 'low'
    elif risk_level < RISK_LEVELS['medium'][1]:
        level = 'medium'
    else:
        level = 'high'
    return SINGAPORE_REFERRALS.get(level, SINGAPORE_REFERRALS['low']) # Default to low

class ParkinsonsDetector:
    def __init__(self, window_size=30, trend_window_velocity=7, trend_window_fixation=30):
        # Buffer for metrics history over the analysis window
        self.metrics_history = deque(maxlen=window_size)
        
        # Clinical thresholds for Parkinson's detection - Load from imported dict
        self.thresholds = PD_THRESHOLDS
        
        # Buffers for longitudinal trend analysis (storing window averages)
        self.trend_analyzer = {
            # Store avg velocity over the window for the last 7 analyses (e.g., weekly if run daily)
            'velocity': deque(maxlen=trend_window_velocity),
            # Store avg fixation stability over the window for the last 30 analyses (e.g., monthly if run daily)
            'fixation': deque(maxlen=trend_window_fixation)
        }
        print(f"Initialized ParkinsonsDetector with window: {window_size}, trend windows: V={trend_window_velocity}, F={trend_window_fixation}")

    def analyze_metrics(self, metrics):
        """Analyze eye metrics for Parkinson's indicators"""
        # Add current metrics to history
        if metrics:
            self.metrics_history.append(metrics)
        
        # Only analyze when we have enough data
        if len(self.metrics_history) < self.metrics_history.maxlen:
            return {
                'analysis_complete': False,
                'message': f"Collecting data: {len(self.metrics_history)}/{self.metrics_history.maxlen}"
            }
        
        # Extract features relevant to Parkinson's detection
        features = self._extract_features()
        
        # Analyze the features against clinical thresholds
        analysis_result = self._analyze_features(features)
        
        return analysis_result
    
    def _extract_features(self):
        """Extract features from metrics history"""
        features = {}
        
        # Extract saccade velocities
        saccade_velocities = [m.get('avg_saccade_velocity', 0) for m in self.metrics_history 
                             if 'avg_saccade_velocity' in m]
        if saccade_velocities:
            features['avg_saccade_velocity'] = np.mean(saccade_velocities)

        # Extract vertical saccade velocities
        vertical_saccade_velocities = [m.get('avg_vertical_saccade_velocity', 0) for m in self.metrics_history
                                     if 'avg_vertical_saccade_velocity' in m]
        if vertical_saccade_velocities:
            features['avg_vertical_saccade_velocity'] = np.mean(vertical_saccade_velocities)

        # Extract fixation stability
        fixation_stabilities = [m.get('fixation_stability', 0) for m in self.metrics_history 
                               if 'fixation_stability' in m]
        if fixation_stabilities:
            features['avg_fixation_stability'] = np.mean(fixation_stabilities)
        
        # Extract EAR values for blink detection
        ear_values = [m.get('avg_ear', 1.0) for m in self.metrics_history if 'avg_ear' in m]
        if ear_values:
            # Simple blink detection (EAR drops below 0.2)
            blink_count = 0
            for i in range(1, len(ear_values)):
                if ear_values[i-1] > 0.2 and ear_values[i] <= 0.2:
                    blink_count += 1
            
            # Convert to blinks per minute (assuming 30 FPS)
            # 30 frames * window_size = total frames
            # total_seconds = total_frames / 30
            # blinks_per_minute = blink_count * (60 / total_seconds)
            blinks_per_minute = blink_count * (60 / (self.metrics_history.maxlen / 30))
            features['blink_rate'] = blinks_per_minute
        
        return features
    
    def _analyze_features(self, features):
        """Analyze features against clinical thresholds"""
        risk_factors = []
        risk_level = 0.0
        # Define weights for each factor (adjust as needed, ensuring they sum close to 1)
        weights = {
            'saccade': 0.30,
            'vertical_saccade': 0.20,
            'fixation': 0.25,
            'blink': 0.25
        }


        # Check saccade velocity (lower in Parkinson's)
        if 'avg_saccade_velocity' in features:
            # Use the specific threshold key from PD_THRESHOLDS
            if features['avg_saccade_velocity'] < self.thresholds['saccade_velocity_min']:
                risk_factors.append(f"Reduced saccade velocity: {features['avg_saccade_velocity']:.2f}°/s (Threshold: <{self.thresholds['saccade_velocity_min']})")
                risk_level += weights['saccade']

        # Check vertical saccade velocity (lower in Parkinson's)
        if 'avg_vertical_saccade_velocity' in features:
             # Use the specific threshold key from PD_THRESHOLDS
            if features['avg_vertical_saccade_velocity'] < self.thresholds['vertical_saccade_velocity_min']:
                risk_factors.append(f"Reduced vertical saccade velocity: {features['avg_vertical_saccade_velocity']:.2f}°/s (Threshold: <{self.thresholds['vertical_saccade_velocity_min']})")
                risk_level += weights['vertical_saccade']
        
        # Check fixation stability (higher variance in Parkinson's)
        if 'avg_fixation_stability' in features:
            # Use the specific threshold key from PD_THRESHOLDS
            if features['avg_fixation_stability'] > self.thresholds['fixation_stability']:
                risk_factors.append(f"Reduced fixation stability: {features['avg_fixation_stability']:.4f} (Threshold: >{self.thresholds['fixation_stability']})")
                risk_level += weights['fixation']
        
        # Check blink rate (can be abnormal in Parkinson's)
        if 'blink_rate' in features:
            # Use the specific threshold keys from PD_THRESHOLDS
            if (features['blink_rate'] < self.thresholds['blink_rate_min'] or
                features['blink_rate'] > self.thresholds['blink_rate_max']):
                risk_factors.append(f"Abnormal blink rate: {features['blink_rate']:.2f}/min (Range: {self.thresholds['blink_rate_min']}-{self.thresholds['blink_rate_max']})")
                risk_level += weights['blink']
        
        # Ensure risk level is between 0 and 1
        risk_level = min(max(risk_level, 0.0), 1.0) # Also ensure >= 0

        # --- Longitudinal Trend Update ---
        # Store the calculated average features for this window into the trend deques
        if 'avg_saccade_velocity' in features:
            self.trend_analyzer['velocity'].append(features['avg_saccade_velocity'])
        if 'avg_fixation_stability' in features:
             self.trend_analyzer['fixation'].append(features['avg_fixation_stability'])
        # Note: Actual trend *calculation* (e.g., slope) is not implemented here yet, just data storage.

        # --- Singapore Specific Recommendations & Referral ---
        recommendation = RECOMMENDATIONS.get('low') # Default recommendation
        if risk_level >= RISK_LEVELS['high'][0]:
             recommendation = RECOMMENDATIONS.get('high')
             # Add Singapore-specific high-risk recommendation
             recommendation += " Consider referral to SGH Neurodegenerative Cohort Study."
        elif risk_level >= RISK_LEVELS['medium'][0]:
             recommendation = RECOMMENDATIONS.get('medium')

        referral = get_referral(risk_level) # Get SG referral pathway

        return {
            'analysis_complete': True,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'features': features, # Features calculated over the current window
            'recommendation': recommendation, # General + SG specific recommendation
            'referral': referral # SG specific referral pathway
            # 'trends': self._calculate_trends() # Could add trend calculation results here later
        }
        
    # Placeholder for future trend calculation logic
    # def _calculate_trends(self):
    #     trends = {}
    #     if len(self.trend_analyzer['velocity']) > 1:
    #         # Example: Calculate slope using linear regression or simple difference
    #         # trends['velocity_trend'] = ...
    #         pass
    #     if len(self.trend_analyzer['fixation']) > 1:
    #         # trends['fixation_trend'] = ...
    #         pass
    #     return trends

