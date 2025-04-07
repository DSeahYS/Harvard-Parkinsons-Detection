import cv2
import numpy as np
import tkinter as tk # Needed for draw_risk_meter

def draw_risk_meter(canvas: tk.Canvas, risk_level: float):
    """
    Draws a risk meter directly onto a Tkinter Canvas.

    Args:
        canvas (tk.Canvas): The Tkinter canvas widget to draw on.
        risk_level (float): The risk level (0.0 to 1.0).
    """
    if not canvas or not canvas.winfo_exists():
        return # Don't draw if canvas doesn't exist

    canvas.delete("all") # Clear previous drawings
    width = canvas.winfo_width()
    height = canvas.winfo_height()

    if width <= 1 or height <= 1: # Canvas not ready
        canvas.create_text(10, 10, text="Loading...", anchor=tk.NW)
        return

    # Ensure risk_level is within bounds
    risk_level = np.clip(risk_level, 0.0, 1.0)

    fill_width = int(risk_level * width)

    # Draw background gradient (optional, can be slow)
    # for i in range(width):
    #     ratio = i / width
    #     # Simple linear gradient: Green -> Yellow -> Red
    #     # ... (color calculation logic from create_risk_meter)
    #     # color_hex = f'#{blue:02x}{green:02x}{red:02x}' # Incorrect order for hex
    #     color_hex = f'#{red:02x}{green:02x}{blue:02x}'
    #     canvas.create_line(i, 0, i, height, fill=color_hex)

    # Draw filled portion with color based on risk level
    if risk_level < 0.3:
        fill_color = "#00FF00" # Green
    elif risk_level < 0.6:
        fill_color = "#FFFF00" # Yellow
    else:
        fill_color = "#FF0000" # Red

    if fill_width > 0:
        canvas.create_rectangle(0, 0, fill_width, height, fill=fill_color, outline="")

    # Draw border
    canvas.create_rectangle(0, 0, width - 1, height - 1, outline="black", width=1)

    # Draw text centered
    text = f"Risk: {risk_level:.3f}"
    text_color = "black" # Ensure text is visible on all background colors
    canvas.create_text(width / 2, height / 2, text=text, fill=text_color, font=("Arial", 10))


def create_risk_meter_image(risk_level, width=200, height=30):
    """
    Creates a visual risk meter image based on the risk level.
    (Original function, kept for potential alternative use)

    Args:
        risk_level (float): The risk level (e.g., 0.0 to 1.0).
        width (int): The width of the meter image.
        height (int): The height of the meter image.

    Returns:
        np.ndarray: An OpenCV image representing the risk meter.
    """
    meter = np.full((height, width, 3), (200, 200, 200), dtype=np.uint8) # Light grey background
    risk_level = np.clip(risk_level, 0.0, 1.0) # Ensure bounds
    fill_width = int(risk_level * width)

    # Color gradient from green to red for the filled part
    for i in range(fill_width):
        ratio = i / width if width > 0 else 0
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
        # Draw vertical line for this color segment
        cv2.line(meter, (i, 0), (i, height), (blue, green, red), 1)

    # Add border and text
    cv2.rectangle(meter, (0, 0), (width - 1, height - 1), (50, 50, 50), 1) # Darker border
    text = f"Risk: {risk_level:.3f}"
    # Calculate text size to center it
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    text_x = (width - text_width) // 2
    text_y = (height + text_height) // 2
    cv2.putText(meter, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1) # Black text

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
    cv2.putText(vis_image, "Metrics History Plot (Not Implemented)", (10, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    # In a real implementation, you'd use libraries like Matplotlib (rendered to numpy array)
    # or draw directly with OpenCV functions.
    return vis_image

def draw_debug_overlay(frame, metrics=None, risk_info=None):
    """
    Draws debugging metrics and risk information onto a frame.
    (Simplified version, assumes landmarks are drawn by EyeTracker._draw_debug_overlay)

    Args:
        frame (np.ndarray): The input video frame (should be BGR).
        metrics (dict, optional): Dictionary of calculated eye metrics.
                                  Expected keys like 'saccade_velocity_deg_s', etc.
        risk_info (tuple, optional): Tuple containing (risk_level, contributing_factors).

    Returns:
        np.ndarray: The frame with the debug overlay drawn.
    """
    overlay_frame = frame # Draw directly on the frame passed in
    h, w = frame.shape[:2]
    y_offset = 30
    font_scale = 0.45
    font_color = (255, 255, 0) # Cyan
    font_thickness = 1

    # Display metrics (if provided)
    if metrics:
        # Define which metrics to display and their order
        display_order = ['saccade_velocity_deg_s', 'fixation_stability', 'blink_rate_bpm', 'estimated_distance_cm']
        for key in display_order:
            if key in metrics:
                 value = metrics[key]
                 if isinstance(value, float):
                     text = f"{key.replace('_deg_s','').replace('_bpm','').replace('_cm','').replace('_', ' ').title()}: {value:.2f}"
                 elif isinstance(value, tuple) and len(value) == 2: # Handle pupil pos
                      text = f"{key.replace('_', ' ').title()}: ({value[0]},{value[1]})" # Basic tuple display
                 elif value is None:
                      text = f"{key.replace('_', ' ').title()}: N/A"
                 else:
                      text = f"{key.replace('_', ' ').title()}: {value}"
                 cv2.putText(overlay_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
                 y_offset += 18 # Adjust spacing

    # Display risk info (if provided)
    if risk_info:
        risk_level, factors = risk_info
        if risk_level is not None:
            risk_text = f"PD Risk: {risk_level:.3f}"
            # Change color based on risk
            if risk_level < 0.3: risk_color = (0, 255, 0) # Green
            elif risk_level < 0.6: risk_color = (0, 255, 255) # Yellow
            else: risk_color = (0, 0, 255) # Red

            cv2.putText(overlay_frame, risk_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, risk_color, 2)
            y_offset += 25
            # Display factors below risk score
            if factors:
                 cv2.putText(overlay_frame, "Factors:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
                 y_offset += 15
                 # Sort factors by contribution for consistent display
                 sorted_factors = sorted(factors.items(), key=lambda item: abs(item[1]), reverse=True)
                 for factor, value in sorted_factors[:3]: # Show top 3 factors
                     factor_text = f"- {factor}: {value:.3f}"
                     cv2.putText(overlay_frame, factor_text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, font_color, font_thickness)
                     y_offset += 15


    return overlay_frame # Return modified frame
