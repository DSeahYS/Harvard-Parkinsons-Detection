import cv2
import numpy as np
import time
import tkinter as tk
from collections import deque
from PIL import Image, ImageTk
from src.models.eye_tracker import EyeTracker
from src.models.pd_detector import ParkinsonsDetector
from src.utils.visualization import create_metrics_visualization, create_risk_meter
from src.frontend.dashboard import Dashboard
from src.data.thresholds import PD_THRESHOLDS, RISK_LEVELS, RECOMMENDATIONS

def main():
    # Initialize components
    eye_tracker = EyeTracker()
    pd_detector = ParkinsonsDetector()
    dashboard = Dashboard()
    
    # Buffer for metrics history
    metrics_history = deque(maxlen=90)  # 3 seconds at 30 FPS
    
    # Set up capture
    cap = cv2.VideoCapture(0)
    
    # Start webcam processing
    def process_webcam():
        if dashboard.running and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Process frame with eye tracker
                if dashboard.debug_mode:
                    # Process with visualization on original frame
                    processed_frame, metrics = eye_tracker.process_frame(frame)
                else:
                    # Process on copy to preserve original frame
                    frame_copy = frame.copy()
                    processed_frame, metrics = eye_tracker.process_frame(frame_copy)
                    processed_frame = frame  # Keep original frame without indicators
                print(f"Generated metrics: {metrics}")  # Debug output
                
                # Update metrics history
                if metrics:
                    metrics_history.append(metrics)
                    print(f"Metrics history count: {len(metrics_history)}")  # Debug
                
                # Get Parkinson's analysis
                analysis = pd_detector.analyze_metrics(metrics)
                print(f"Analysis results: {analysis}")  # Debug
                
                # Create visualizations
                risk_meter_img = None
                metrics_vis_img = None
                
                if analysis.get('analysis_complete', False):
                    risk_level = analysis.get('risk_level', 0.0)
                    risk_meter_img = create_risk_meter(risk_level)
                    print(f"Created risk meter for level: {risk_level}")  # Debug
                
                if len(metrics_history) > 10:
                    metrics_vis_img = create_metrics_visualization(list(metrics_history))
                    print("Created metrics visualization")  # Debug
                
                # Update dashboard
                dashboard.update_metrics(metrics)
                dashboard.update_risk_assessment(analysis)
                dashboard.update_visualization(risk_meter_img, metrics_vis_img)
                
                # Convert frame to format suitable for tkinter
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(image=img)
                
                # Update canvas with proper reference keeping
                dashboard.canvas.delete("all")
                dashboard.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                dashboard.canvas.image = img_tk  # Keep reference!
                
                # Force update
                dashboard.window.update_idletasks()
            
            # Schedule next frame processing
            dashboard.window.after(33, process_webcam)  # ~30 FPS
    
    # Connect start button to webcam processing
    dashboard.on_start = lambda: (setattr(dashboard, 'running', True), process_webcam())
    
    # Run the dashboard
    dashboard.run()
    
    # Clean up resources
    cap.release()

if __name__ == "__main__":
    main()
