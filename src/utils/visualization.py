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
        # Filter out None values which occur in 'eye' mode
        ear_values = [m.get('avg_ear') for m in metrics_history if m.get('avg_ear') is not None]
        max_ear = max(ear_values) if ear_values else 0.3  # Default to 0.3 if no valid EAR values
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

def create_neuro_mirror(frame, metrics, ethnicity):
    """
    Creates an AR visualization overlay on the frame representing estimated
    basal ganglia dopamine levels based on fixation stability and ethnicity.

    Args:
        frame (np.array): The input video frame (BGR).
        metrics (dict): Dictionary containing eye tracking metrics, expecting 'fixation_stability'.
        ethnicity (str): The detected or provided ethnicity ('chinese', 'malay', 'indian', 'default').

    Returns:
        np.array: The frame with the neuro-mirror overlay.
                  Returns the original frame if metrics are missing.
    """
    if not metrics or 'fixation_stability' not in metrics:
        # print("Warning: 'fixation_stability' not found in metrics for neuro_mirror.")
        return frame # Return original frame if data is missing

    # Ethnicity-based color mapping for dopamine level visualization
    ETHNIC_COLORS = {
        'chinese': (0, 255, 0),   # Green (higher stability -> more green)
        'malay': (0, 255, 255),   # Yellow
        'indian': (0, 165, 255),  # Orange
        'default': (0, 0, 255)    # Red (lower stability -> more red)
    }

    # Normalize fixation stability (assuming lower is better, max instability around 0.3 based on feedback)
    # Higher stability -> lower depletion value (closer to 0)
    # Lower stability -> higher depletion value (closer to 1 or more)
    max_instability_ref = 0.3 # Reference value for maximum instability
    stability = metrics.get('fixation_stability', max_instability_ref) # Default to max if missing
    depletion = min(max(stability / max_instability_ref, 0), 1.5) # Cap depletion factor somewhat

    # Get color based on ethnicity, default to red
    color = ETHNIC_COLORS.get(ethnicity.lower() if ethnicity else 'default', ETHNIC_COLORS['default'])

    # Create pulsating effect based on time
    # The size of the pulse could potentially be linked to another metric, e.g., saccade velocity?
    pulse_base_radius = 30
    pulse_amplitude = 20
    pulse = int(pulse_base_radius + pulse_amplitude * (1 + np.sin(time.time() * 3)) / 2) # Smoother pulse 0-1 range

    # Determine center - using fixed values assumes 640x480 frame. Better to calculate.
    h, w, _ = frame.shape
    center_x, center_y = w // 2, h // 2 # Center of the frame

    # Draw the pulsating circle representing dopamine level (inversely related to depletion)
    # Make the circle smaller/dimmer with higher depletion (lower dopamine)
    # We can adjust alpha (transparency) or radius based on depletion
    
    # Option 1: Adjust radius based on depletion (smaller = more depleted)
    # display_radius = int(pulse * (1 - depletion * 0.7)) # Reduce radius significantly with depletion
    
    # Option 2: Adjust color intensity/alpha (requires creating an overlay)
    overlay = frame.copy()
    cv2.circle(overlay, (center_x, center_y), pulse, color, -1)
    
    # Adjust alpha based on depletion (more transparent = more depleted)
    alpha = max(0.1, 1.0 - depletion * 0.6) # Ensure minimum visibility
    
    # Blend the overlay with the original frame
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Add a small text indicator (optional)
    cv2.putText(frame, f"Est. Dopamine Level ({ethnicity})", (center_x - 100, center_y - pulse - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return frame
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

