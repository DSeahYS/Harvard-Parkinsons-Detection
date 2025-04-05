import cv2
import numpy as np

def create_risk_meter(risk_level, width=200, height=100):
    """
    Creates a visual risk meter image based on the risk level.

    Args:
        risk_level (float): The risk level (e.g., 0.0 to 1.0).
        width (int): The width of the meter image.
        height (int): The height of the meter image.

    Returns:
        np.ndarray: An OpenCV image representing the risk meter.
    """
    meter = np.zeros((height, width, 3), dtype=np.uint8)
    fill_width = int(risk_level * width)

    # Color gradient from green to red
    for i in range(fill_width):
        ratio = i / width
        # Simple linear gradient: Green -> Yellow -> Red
        if ratio < 0.5:
            # Green to Yellow
            red = int(255 * (ratio * 2))
            green = 255
            blue = 0
        else:
            # Yellow to Red
            red = 255
            green = int(255 * (1 - (ratio - 0.5) * 2))
            blue = 0
        cv2.line(meter, (i, 0), (i, height), (blue, green, red), 1)

    # Add border and text
    cv2.rectangle(meter, (0, 0), (width - 1, height - 1), (255, 255, 255), 1)
    text = f"Risk: {risk_level:.2f}"
    cv2.putText(meter, text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return meter

def create_metrics_visualization(metrics_history):
    """
    Creates a visualization of historical metrics (placeholder).

    Args:
        metrics_history (dict): A dictionary containing lists of historical metrics.
                                Example: {'saccade_velocity': [1, 2, 3], 'fixation_stability': [4, 5, 6]}

    Returns:
        np.ndarray: An OpenCV image visualizing the metrics (or None if not implemented).
    """
    # Placeholder: This function needs a proper implementation
    # based on how you want to visualize the time-series data (e.g., line charts).
    # For now, return a simple placeholder image.
    height, width = 200, 400
    vis_image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(vis_image, "Metrics History Plot", (10, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    # In a real implementation, you'd use libraries like Matplotlib (rendered to numpy array)
    # or draw directly with OpenCV functions.
    return vis_image

def draw_debug_overlay(frame, eye_landmarks=None, face_landmarks=None, metrics=None, risk_info=None):
    """
    Draws debugging information onto a frame.

    Args:
        frame (np.ndarray): The input video frame.
        eye_landmarks (list, optional): List of eye landmark points.
        face_landmarks (list, optional): List of face landmark points.
        metrics (dict, optional): Dictionary of calculated eye metrics.
        risk_info (tuple, optional): Tuple containing (risk_level, contributing_factors).

    Returns:
        np.ndarray: The frame with the debug overlay drawn.
    """
    overlay_frame = frame.copy()
    y_offset = 30

    # Draw landmarks (if provided)
    if face_landmarks:
        for landmark in face_landmarks.landmark:
             x = int(landmark.x * frame.shape[1])
             y = int(landmark.y * frame.shape[0])
             cv2.circle(overlay_frame, (x, y), 1, (0, 255, 0), -1) # Green dots for face

    if eye_landmarks:
         # Example: Draw circles around specific eye landmarks if needed
         # This depends on the structure of eye_landmarks from MediaPipe
         pass # Add specific landmark drawing logic here if needed

    # Display metrics (if provided)
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, float):
                text = f"{key.replace('_', ' ').title()}: {value:.2f}"
            else:
                 text = f"{key.replace('_', ' ').title()}: {value}"
            cv2.putText(overlay_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_offset += 20

    # Display risk info (if provided)
    if risk_info:
        risk_level, factors = risk_info
        risk_text = f"PD Risk: {risk_level:.3f}"
        cv2.putText(overlay_frame, risk_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset += 25
        if factors:
             cv2.putText(overlay_frame, "Factors:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
             y_offset += 15
             for factor, value in factors.items():
                 factor_text = f"- {factor}: {value:.2f}"
                 cv2.putText(overlay_frame, factor_text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                 y_offset += 15


    return overlay_frame
