"""
GenomeGuard - Parkinson's Detection System

A system that combines eye tracking and genomic analysis to detect early signs of Parkinson's disease.
"""
import os
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Suppress MediaPipe warnings
os.environ['GLOG_minloglevel'] = '2'
import numpy as np
from datetime import datetime
import logging

from src.models.eye_tracker import EyeTracker
from src.models.pd_detector import ParkinsonsDetector
from src.frontend.dashboard import Dashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='genomicguard.log'
)

def main():
    """Main function to initialize and run the application."""
    try:
        # Initialize components
        eye_tracker = EyeTracker()
        pd_detector = ParkinsonsDetector()
        
        # Initialize metrics history
        metrics_history = []
        
        # Create and run dashboard
        dashboard = Dashboard(eye_tracker, pd_detector, metrics_history)
        dashboard.run()
        
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main()
