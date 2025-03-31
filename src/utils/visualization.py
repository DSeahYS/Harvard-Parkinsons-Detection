import cv2
import numpy as np
from src.models.eye_tracker import EyeTracker
import time

def create_risk_meter(risk_level):
    """Create a visual risk meter image"""
    width, height = 300, 50
    meter = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw meter background
    cv2.rectangle(meter, (0, 0), (width, height), (50, 50, 50), -1)
    
    # Calculate filled width based on risk level (0-1)
    filled_width = int(width * min(max(risk_level, 0), 1))
    
    # Determine color based on risk level
    if risk_level < 0.3:
        color = (0, 255, 0)  # Green
    elif risk_level < 0.7:
        color = (0, 255, 255)  # Yellow
    else:
        color = (0, 0, 255)  # Red
    
    # Draw filled portion
    cv2.rectangle(meter, (0, 0), (filled_width, height), color, -1)
    
    # Add text
    cv2.putText(meter, f"Risk: {risk_level*100:.1f}%",
                (10, height//2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return meter

def create_metrics_visualization(metrics_history):
    """Create a visualization of historical metrics"""
    height = 200
    width = 400
    vis = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw background
    cv2.rectangle(vis, (0, 0), (width, height), (50, 50, 50), -1)
    
    # Plot EAR (Eye Aspect Ratio) over time
    if len(metrics_history) > 1:
        ear_values = [m.get('avg_ear', 0) for m in metrics_history]
        max_ear = max(ear_values) if ear_values else 0.3
        scale = (height - 20) / (max_ear or 0.3)
        
        points = []
        for i, ear in enumerate(ear_values):
            x = int(i * width / len(ear_values))
            y = height - int(ear * scale) - 10
            points.append((x, y))
        
        if len(points) > 1:
            cv2.polylines(vis, [np.array(points)], False, (0, 255, 0), 2)
    
    # Add title and labels
    cv2.putText(vis, "Eye Metrics History", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return vis

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    # Initialize the eye tracker
    eye_tracker = EyeTracker()
    
    # Track frame processing times for performance metrics
    frame_times = []
    
    while cap.isOpened():
        # Read frame from webcam
        success, frame = cap.read()
        if not success:
            print("Failed to read from webcam")
            break
            
        # Start frame processing timer
        start_time = time.time()
        
        # Process the frame with the eye tracker
        processed_frame, metrics = eye_tracker.process_frame(frame)
        
        # Calculate frame processing time
        frame_time = time.time() - start_time
        frame_times.append(frame_time)
        
        # Display FPS
        fps = 1 / frame_time
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display metrics if available
        if metrics:
            y_pos = 70
            for key, value in metrics.items():
                text = f"{key}: {value}"
                cv2.putText(processed_frame, text, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_pos += 30
        
        # Display the frame
        cv2.imshow('MediaPipe Eye Tracking', processed_frame)
        
        # Exit on ESC key
        if cv2.waitKey(5) & 0xFF == 27:
            break
    
    # Calculate average frame processing time
    avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
    print(f"Average frame processing time: {avg_frame_time:.4f} seconds")
    print(f"Average FPS: {1/avg_frame_time:.2f}")
    
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
