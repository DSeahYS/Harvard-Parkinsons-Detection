import cv2
import numpy as np
import time
import os
import absl.logging
from collections import deque
from dotenv import load_dotenv # Added

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize absl logging to avoid feedback tensor warnings
absl.logging.set_verbosity(absl.logging.ERROR)

# Load environment variables from .env file
load_dotenv(dotenv_path='GenomeGuard/.env') # Specify path relative to project root

from src.models.eye_tracker import EyeTracker
from src.models.pd_detector import ParkinsonsDetector
from src.utils.visualization import create_metrics_visualization, create_risk_meter
from src.frontend.dashboard import Dashboard
from src.utils.cycle_buffer import CycleBuffer

def main():
    # Initialize components
    eye_tracker = EyeTracker()
    pd_detector = ParkinsonsDetector()
    metrics_history = deque(maxlen=90)  # 3 seconds at 30 FPS

    # Initialize cycle buffer
    cycle_buffer = CycleBuffer(cycle_duration=15)

    # Create dashboard with components
    dashboard = Dashboard(eye_tracker, pd_detector, metrics_history)

    # Run the dashboard
    dashboard.run()

    # Clean up
    if dashboard.cap:
        dashboard.cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
