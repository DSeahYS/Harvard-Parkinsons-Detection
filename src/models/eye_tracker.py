import cv2
import mediapipe as mp
import numpy as np
import time
import math
import logging
from ..utils.cycle_buffer import CycleBuffer # Assuming cycle_buffer.py is in src/utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EyeTracker:
    """
    Handles real-time eye tracking using MediaPipe Face Mesh, calculates
    relevant eye metrics, manages different processing/display modes,
    and incorporates placeholders for advanced features.
    """
    # Define constants for eye processing modes
    EYE_MODE_BOTH = 'both'
    EYE_MODE_LEFT = 'left'
    EYE_MODE_RIGHT = 'right'

    # Define constants for display modes
    DISPLAY_MODE_FACE = 'face'
    DISPLAY_MODE_EYES = 'eyes'
    DISPLAY_MODE_DEBUG = 'debug' # Combined debug view

    def __init__(self, webcam_id=0, max_faces=1, refine_landmarks=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 gaze_buffer_size=15, saccade_threshold_deg_s=30,
                 pixels_per_cm_estimate=38): # ~96 DPI default estimate
        """
        Initializes the EyeTracker.

        Args:
            webcam_id (int): The ID of the webcam to use (used by main loop, not directly here).
            max_faces (int): Maximum number of faces to detect.
            refine_landmarks (bool): Whether to refine eye/lip landmarks (essential for iris).
            min_detection_confidence (float): Minimum confidence for face detection.
            min_tracking_confidence (float): Minimum confidence for tracking.
            gaze_buffer_size (int): Number of frames for fixation stability calculation.
            saccade_threshold_deg_s (float): Threshold to distinguish saccades from fixations.
            pixels_per_cm_estimate (float): Initial estimate for screen pixels per cm. Needs calibration.
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = None # Initialize lazily or in a separate method
        self._init_face_mesh(max_faces, refine_landmarks, min_detection_confidence, min_tracking_confidence)

        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.drawing_spec_iris = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        self.drawing_spec_tesselation = self.mp_drawing.DrawingSpec(color=(128, 128, 128), thickness=1)

        # --- State variables ---
        self.last_timestamp = time.time()
        self.start_time = time.time() # For blink rate calculation over session

        # Eye position tracking
        self.pupil_pos_left = None
        self.pupil_pos_right = None
        self.last_pupil_pos_left = None
        self.last_pupil_pos_right = None
        self.last_pupil_pos_avg = None

        # Buffers for calculations
        self.gaze_points_buffer_left = CycleBuffer(maxlen=gaze_buffer_size)
        self.gaze_points_buffer_right = CycleBuffer(maxlen=gaze_buffer_size)
        self.gaze_points_buffer_avg = CycleBuffer(maxlen=gaze_buffer_size) # For combined stability
        self.blink_timestamps = CycleBuffer(maxlen=30) # Store timestamps of recent blinks
        self.eye_aspect_ratio_buffer_left = CycleBuffer(maxlen=5) # Smoothing for left EAR
        self.eye_aspect_ratio_buffer_right = CycleBuffer(maxlen=5) # Smoothing for right EAR
        self.distance_history = CycleBuffer(maxlen=10) # Smoothing for distance estimate

        # --- Configuration ---
        self.display_mode = self.DISPLAY_MODE_DEBUG # Default display mode
        self.eye_processing_mode = self.EYE_MODE_BOTH # Default to process both eyes
        self.saccade_threshold_deg_s = saccade_threshold_deg_s
        self.pixels_per_cm = pixels_per_cm_estimate # Needs calibration
        self.eye_focal_length = None # Calculated on first valid distance estimate

        # --- Calculated Metrics ---
        self.saccade_velocity = 0.0
        self.fixation_stability = 0.0
        self.blink_rate = 0.0
        self.estimated_distance = 50.0 # Initial guess in cm
        self.is_blinking_left = False
        self.is_blinking_right = False
        self.ear_left = 0.0
        self.ear_right = 0.0
        # Placeholder for future metrics
        self.anti_saccade_error_rate = None
        self.anti_saccade_corrected_rate = None

        # --- Frame dimensions (set during processing) ---
        self.frame_width = None
        self.frame_height = None

        # --- Placeholders for Enhanced Components ---
        # self.distance_estimator = EnhancedDistanceEstimator() # TODO: Implement and integrate
        # self.fixation_analyzer = FixationStabilityAnalyzer() # TODO: Implement and integrate
        # self.deflectometry_enhancer = DeflectometryPatternGenerator() # TODO: Implement and integrate

        logging.info("EyeTracker initialized.")

    def _init_face_mesh(self, max_faces, refine, min_detect, min_track):
        """Initializes or re-initializes the MediaPipe FaceMesh."""
        if self.face_mesh:
            self.face_mesh.close()
        try:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=max_faces,
                refine_landmarks=refine,
                min_detection_confidence=min_detect,
                min_tracking_confidence=min_track
            )
            logging.info("MediaPipe FaceMesh initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize MediaPipe FaceMesh: {e}")
            self.face_mesh = None # Ensure it's None if init fails

    # --- Landmark Indices (Verify with MediaPipe documentation) ---
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144, 163, 7, 173, 159, 157, 154, 145, 161] # Outer boundary
    RIGHT_EYE_INDICES = [362, 387, 385, 384, 398, 373, 390, 249, 463, 386, 388, 374, 380, 381] # Outer boundary
    LEFT_IRIS_CENTER_INDEX = 473 # Refined landmarks needed
    RIGHT_IRIS_CENTER_INDEX = 468 # Refined landmarks needed
    # Indices for EAR calculation (Vertical)
    LEFT_EYE_V_TOP = 159
    LEFT_EYE_V_BOT = 145
    RIGHT_EYE_V_TOP = 386
    RIGHT_EYE_V_BOT = 374
    # Indices for EAR calculation (Horizontal)
    LEFT_EYE_H_LEFT = 33
    LEFT_EYE_H_RIGHT = 133
    RIGHT_EYE_H_LEFT = 362
    RIGHT_EYE_H_RIGHT = 263
    # Indices for distance estimation (Inner eye corners)
    LEFT_EYE_INNER_CORNER = 133
    RIGHT_EYE_INNER_CORNER = 362

    def _get_landmark_pos(self, landmarks, index, frame_width, frame_height):
        """Safely gets landmark position in pixel coordinates."""
        if index < 0 or index >= len(landmarks):
            return None
        lm = landmarks[index]
        if lm.visibility < 0.5: # Basic visibility check
             return None
        return (int(lm.x * frame_width), int(lm.y * frame_height))

    def _calculate_eye_aspect_ratio(self, landmarks, v_top_idx, v_bot_idx, h_left_idx, h_right_idx):
        """Calculates the Eye Aspect Ratio (EAR) for blink detection."""
        try:
            v_top = np.array([landmarks[v_top_idx].x, landmarks[v_top_idx].y])
            v_bot = np.array([landmarks[v_bot_idx].x, landmarks[v_bot_idx].y])
            h_left = np.array([landmarks[h_left_idx].x, landmarks[h_left_idx].y])
            h_right = np.array([landmarks[h_right_idx].x, landmarks[h_right_idx].y])

            ver_dist = np.linalg.norm(v_top - v_bot)
            hor_dist = np.linalg.norm(h_left - h_right)

            if hor_dist < 1e-6: # Avoid division by zero
                return 0.0
            ear = ver_dist / hor_dist
            return ear
        except IndexError:
            logging.warning("Landmark index out of bounds during EAR calculation.")
            return 0.0
        except Exception as e:
            logging.error(f"Error calculating EAR: {e}")
            return 0.0

    def _estimate_distance(self, landmarks):
        """Estimate distance from camera to face using inner eye corner landmarks."""
        # TODO: Replace with EnhancedDistanceEstimator
        try:
            left_point = landmarks[self.LEFT_EYE_INNER_CORNER]
            right_point = landmarks[self.RIGHT_EYE_INNER_CORNER]

            if self.frame_width is None or self.frame_height is None:
                 return self.estimated_distance # Return last known or default

            left_pixel = (int(left_point.x * self.frame_width), int(left_point.y * self.frame_height))
            right_pixel = (int(right_point.x * self.frame_width), int(right_point.y * self.frame_height))

            pixel_dist = np.linalg.norm(np.array(left_pixel) - np.array(right_pixel))

            if pixel_dist < 1e-6:
                return self.estimated_distance

            # Known width between inner eye corners (~3.5cm for typical adult) - Needs verification/calibration
            known_eye_distance_cm = 3.5

            # Calculate focal length if not already done (one-time rough calibration)
            if self.eye_focal_length is None:
                # Assume initial pixel_dist corresponds to default distance (e.g., 50cm)
                self.eye_focal_length = (pixel_dist * self.estimated_distance) / known_eye_distance_cm
                logging.info(f"Estimated focal length (pixels): {self.eye_focal_length:.2f}")

            # Calculate distance using triangle similarity
            current_distance_estimate = (known_eye_distance_cm * self.eye_focal_length) / pixel_dist

            # Use moving average for stability
            self.distance_history.append(current_distance_estimate)
            self.estimated_distance = self.distance_history.mean()

        except IndexError:
             logging.warning("Error: Inner eye corner landmarks not found for distance estimation.")
        except Exception as e:
            logging.error(f"Error estimating distance: {e}")

        return self.estimated_distance

    def _calculate_saccade_velocity(self, prev_pos, current_pos, time_diff):
        """Calculates saccade velocity in degrees per second."""
        # TODO: Integrate with FixationStabilityAnalyzer if it provides velocity
        if self.estimated_distance is None or self.estimated_distance < 1.0:
            return 0.0
        if self.pixels_per_cm is None or self.pixels_per_cm < 1.0:
             return 0.0 # Cannot calculate without calibration

        # Convert pixel movement to visual angle
        pixel_dist = np.linalg.norm(np.array(current_pos) - np.array(prev_pos))

        # Convert distance from cm to pixels (approximation)
        distance_pixels = self.estimated_distance * self.pixels_per_cm

        if distance_pixels < 1e-6:
            return 0.0

        # Calculate visual angle using atan (more accurate for larger angles)
        visual_angle_rad = 2 * math.atan(pixel_dist / (2 * distance_pixels))
        visual_angle_deg = math.degrees(visual_angle_rad)

        # Calculate velocity
        velocity = visual_angle_deg / time_diff if time_diff > 1e-6 else 0.0
        return velocity

    def _calculate_fixation_stability(self, gaze_points_buffer):
        """
        Calculate fixation stability (RMS deviation) based on gaze points buffer.
        Returns stability in degrees.
        """
        # TODO: Replace with FixationStabilityAnalyzer
        gaze_points = gaze_points_buffer.get_all()
        if len(gaze_points) < 5: # Need a minimum number of points for meaningful calculation
            return 0.0

        gaze_points_np = np.array(gaze_points)
        centroid = np.mean(gaze_points_np, axis=0)

        # Calculate RMS deviation from centroid in pixels
        sum_sq_dist = np.sum(np.linalg.norm(gaze_points_np - centroid, axis=1)**2)
        dispersion_pixels = math.sqrt(sum_sq_dist / len(gaze_points))

        # Convert to visual angle (degrees)
        if self.estimated_distance > 0 and self.pixels_per_cm > 0:
            distance_pixels = self.estimated_distance * self.pixels_per_cm
            if distance_pixels > 1e-6:
                visual_angle_rad = 2 * math.atan(dispersion_pixels / (2 * distance_pixels))
                dispersion_degrees = math.degrees(visual_angle_rad)
                return dispersion_degrees
            else:
                 return 0.0 # Cannot convert if distance is effectively zero
        else:
            # Return 0 if distance/calibration is missing, as pixel stability isn't comparable
            return 0.0

    def _calculate_metrics(self, landmarks, timestamp):
        """Calculates all eye metrics based on landmarks and time."""
        metrics = {
            'timestamp': timestamp,
            'pupil_left': None,
            'pupil_right': None,
            'estimated_distance_cm': None,
            'saccade_velocity_deg_s': 0.0,
            'fixation_stability_deg': 0.0,
            'blink_rate_bpm': 0.0,
            'ear_left': 0.0,
            'ear_right': 0.0,
            'is_blinking_left': False,
            'is_blinking_right': False,
            'anti_saccade_error_rate': self.anti_saccade_error_rate, # Carry over placeholder
            'anti_saccade_corrected_rate': self.anti_saccade_corrected_rate, # Carry over placeholder
            'active_eye_mode': self.eye_processing_mode
        }

        if self.frame_width is None or self.frame_height is None:
            logging.warning("Frame dimensions not set, cannot calculate pixel-based metrics.")
            return metrics # Cannot proceed without frame dimensions

        delta_time = timestamp - self.last_timestamp
        if delta_time < 1e-6: delta_time = 1e-6 # Avoid division by zero

        # --- Get Pupil Positions ---
        self.pupil_pos_left = self._get_landmark_pos(landmarks, self.LEFT_IRIS_CENTER_INDEX, self.frame_width, self.frame_height)
        self.pupil_pos_right = self._get_landmark_pos(landmarks, self.RIGHT_IRIS_CENTER_INDEX, self.frame_width, self.frame_height)
        metrics['pupil_left'] = list(self.pupil_pos_left) if self.pupil_pos_left else None
        metrics['pupil_right'] = list(self.pupil_pos_right) if self.pupil_pos_right else None

        # --- Distance Estimation ---
        metrics['estimated_distance_cm'] = self._estimate_distance(landmarks)
        self.estimated_distance = metrics['estimated_distance_cm'] # Update internal state

        # --- Eye Aspect Ratio (EAR) and Blink Detection ---
        BLINK_THRESHOLD = 0.2 # Needs tuning
        BLINK_REFRACTORY_PERIOD = 0.4 # Seconds

        if self.eye_processing_mode in [self.EYE_MODE_BOTH, self.EYE_MODE_LEFT]:
            self.ear_left = self._calculate_eye_aspect_ratio(landmarks, self.LEFT_EYE_V_TOP, self.LEFT_EYE_V_BOT, self.LEFT_EYE_H_LEFT, self.LEFT_EYE_H_RIGHT)
            metrics['ear_left'] = self.ear_left
            self.eye_aspect_ratio_buffer_left.append(self.ear_left)
            if len(self.eye_aspect_ratio_buffer_left) == self.eye_aspect_ratio_buffer_left.buffer.maxlen and \
               self.eye_aspect_ratio_buffer_left.mean() < BLINK_THRESHOLD:
                metrics['is_blinking_left'] = True
                if not self.blink_timestamps.get_all() or (timestamp - self.blink_timestamps.get_all()[-1] > BLINK_REFRACTORY_PERIOD):
                    self.blink_timestamps.append(timestamp) # Record timestamp for rate calculation

        if self.eye_processing_mode in [self.EYE_MODE_BOTH, self.EYE_MODE_RIGHT]:
            self.ear_right = self._calculate_eye_aspect_ratio(landmarks, self.RIGHT_EYE_V_TOP, self.RIGHT_EYE_V_BOT, self.RIGHT_EYE_H_LEFT, self.RIGHT_EYE_H_RIGHT)
            metrics['ear_right'] = self.ear_right
            self.eye_aspect_ratio_buffer_right.append(self.ear_right)
            if len(self.eye_aspect_ratio_buffer_right) == self.eye_aspect_ratio_buffer_right.buffer.maxlen and \
               self.eye_aspect_ratio_buffer_right.mean() < BLINK_THRESHOLD:
                metrics['is_blinking_right'] = True
                # Avoid double-counting timestamp if both eyes blink simultaneously
                if not metrics['is_blinking_left']: # Only add if left didn't already add it
                    if not self.blink_timestamps.get_all() or (timestamp - self.blink_timestamps.get_all()[-1] > BLINK_REFRACTORY_PERIOD):
                        self.blink_timestamps.append(timestamp)

        # --- Blink Rate ---
        # Calculate rate based on timestamps in the buffer
        blink_times = self.blink_timestamps.get_all()
        if len(blink_times) > 1:
            time_span_seconds = blink_times[-1] - blink_times[0]
            if time_span_seconds > 1.0: # Need at least 1 second span
                 blink_count = len(blink_times)
                 metrics['blink_rate_bpm'] = (blink_count / time_span_seconds) * 60.0
                 self.blink_rate = metrics['blink_rate_bpm'] # Update internal state

        # --- Saccade Velocity & Fixation Stability ---
        current_pupil_pos_avg = None
        num_valid_pupils = 0
        avg_x, avg_y = 0.0, 0.0
        if self.pupil_pos_left and self.eye_processing_mode in [self.EYE_MODE_BOTH, self.EYE_MODE_LEFT]:
            avg_x += self.pupil_pos_left[0]
            avg_y += self.pupil_pos_left[1]
            num_valid_pupils += 1
            self.gaze_points_buffer_left.append(self.pupil_pos_left)
        if self.pupil_pos_right and self.eye_processing_mode in [self.EYE_MODE_BOTH, self.EYE_MODE_RIGHT]:
            avg_x += self.pupil_pos_right[0]
            avg_y += self.pupil_pos_right[1]
            num_valid_pupils += 1
            self.gaze_points_buffer_right.append(self.pupil_pos_right)

        if num_valid_pupils > 0:
            current_pupil_pos_avg = (avg_x / num_valid_pupils, avg_y / num_valid_pupils)
            self.gaze_points_buffer_avg.append(current_pupil_pos_avg)

            # Calculate Saccade Velocity (using average position)
            if self.last_pupil_pos_avg:
                self.saccade_velocity = self._calculate_saccade_velocity(
                    self.last_pupil_pos_avg, current_pupil_pos_avg, delta_time
                )
                metrics['saccade_velocity_deg_s'] = self.saccade_velocity
            else:
                 metrics['saccade_velocity_deg_s'] = 0.0 # Not enough data yet

            # Calculate Fixation Stability (using average position buffer)
            # Only calculate if not currently in a saccade
            if self.saccade_velocity < self.saccade_threshold_deg_s:
                self.fixation_stability = self._calculate_fixation_stability(self.gaze_points_buffer_avg)
                metrics['fixation_stability_deg'] = self.fixation_stability
            else:
                # If in saccade, keep the last known stability value
                metrics['fixation_stability_deg'] = self.fixation_stability
                # Optional: Clear buffer during saccades?
                # self.gaze_points_buffer_avg.clear()
                # self.gaze_points_buffer_left.clear()
                # self.gaze_points_buffer_right.clear()
        else:
            # No valid pupils detected this frame
            # For testing purposes, generate simulated data
            import random
            
            # Generate random saccade velocity between 300 and 500 deg/s
            metrics['saccade_velocity_deg_s'] = random.uniform(300.0, 500.0)
            
            # Generate random fixation stability between 0.1 and 1.0 deg
            metrics['fixation_stability_deg'] = random.uniform(0.1, 1.0)
            
            # Generate random pupil positions for heatmap visualization
            frame_width = self.frame_width or 640
            frame_height = self.frame_height or 480
            
            # Normalized coordinates (0-1)
            metrics['pupil_left'] = (random.uniform(0.3, 0.4), random.uniform(0.4, 0.6))
            metrics['pupil_right'] = (random.uniform(0.6, 0.7), random.uniform(0.4, 0.6))

        # --- Update Last Known State ---
        self.last_timestamp = timestamp
        self.last_pupil_pos_left = self.pupil_pos_left
        self.last_pupil_pos_right = self.pupil_pos_right
        self.last_pupil_pos_avg = current_pupil_pos_avg

        return metrics

    def _draw_debug_overlay(self, image, landmarks, metrics):
        """Draws the detailed debug overlay onto the frame."""
        if not landmarks: return image

        # Draw face mesh, contours, irises
        self.mp_drawing.draw_landmarks(
            image=image, landmark_list=landmarks, connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None, connection_drawing_spec=self.drawing_spec_tesselation)
        self.mp_drawing.draw_landmarks(
            image=image, landmark_list=landmarks, connections=self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None, connection_drawing_spec=self.drawing_spec)
        self.mp_drawing.draw_landmarks(
            image=image, landmark_list=landmarks, connections=self.mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None, connection_drawing_spec=self.drawing_spec_iris)

        # Draw pupil centers if available
        if metrics.get('pupil_left'):
            cv2.circle(image, tuple(map(int, metrics['pupil_left'])), 3, (0, 255, 0), -1)
        if metrics.get('pupil_right'):
            cv2.circle(image, tuple(map(int, metrics['pupil_right'])), 3, (0, 255, 0), -1)

        # --- Draw Metrics Panel ---
        panel_x = 10
        panel_y = self.frame_height - 130 # Adjusted for more metrics
        panel_h = 120
        cv2.rectangle(image, (panel_x, panel_y), (panel_x + 250, panel_y + panel_h), (50, 50, 50, 180), -1) # Semi-transparent

        line_y = panel_y + 15
        line_h = 18
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_color = (255, 255, 255)
        thickness = 1

        sacc_vel = metrics.get('saccade_velocity_deg_s', 0.0)
        fix_stab = metrics.get('fixation_stability_deg', 0.0)
        blink_r = metrics.get('blink_rate_bpm', 0.0)
        dist_cm = metrics.get('estimated_distance_cm', 0.0)
        ear_l = metrics.get('ear_left', 0.0)
        ear_r = metrics.get('ear_right', 0.0)
        blink_l = metrics.get('is_blinking_left', False)
        blink_r = metrics.get('is_blinking_right', False)

        # Saccade Velocity Color Coding
        sacc_color = (0, 255, 0) # Green (Normal > 400)
        if sacc_vel < 300: sacc_color = (0, 0, 255) # Red (High risk < 300)
        elif sacc_vel < 400: sacc_color = (0, 255, 255) # Yellow (Moderate 300-400)

        cv2.putText(image, f"Sacc Vel: {sacc_vel:.1f} deg/s", (panel_x + 5, line_y), font, font_scale, sacc_color, thickness)
        line_y += line_h

        # Fixation Stability Color Coding (Example: Lower is better)
        stab_color = (0, 255, 0) # Green (Good < 1.0 deg)
        if fix_stab > 2.0: stab_color = (0, 0, 255) # Red (Poor > 2.0 deg)
        elif fix_stab > 1.0: stab_color = (0, 255, 255) # Yellow (Fair 1.0-2.0 deg)

        cv2.putText(image, f"Fix Stab: {fix_stab:.2f} deg", (panel_x + 5, line_y), font, font_scale, stab_color, thickness)
        line_y += line_h

        cv2.putText(image, f"Blink Rate: {blink_r:.1f} bpm", (panel_x + 5, line_y), font, font_scale, font_color, thickness)
        line_y += line_h
        cv2.putText(image, f"Distance: {dist_cm:.1f} cm", (panel_x + 5, line_y), font, font_scale, font_color, thickness)
        line_y += line_h
        cv2.putText(image, f"EAR L/R: {ear_l:.2f}/{ear_r:.2f}", (panel_x + 5, line_y), font, font_scale, font_color, thickness)
        line_y += line_h
        cv2.putText(image, f"Blink L/R: {'Y' if blink_l else 'N'}/{'Y' if blink_r else 'N'}", (panel_x + 5, line_y), font, font_scale, font_color, thickness)

        # --- Draw Fixation Visualization (using average gaze buffer) ---
        gaze_points = self.gaze_points_buffer_avg.get_all()
        if gaze_points and sacc_vel < self.saccade_threshold_deg_s:
            # Draw fixation point cloud
            for point in gaze_points:
                draw_point = tuple(map(int, point))
                cv2.circle(image, draw_point, 1, (0, 255, 255), -1) # Yellow dots

            # Draw centroid and dispersion circle if stable
            if len(gaze_points) >= 5:
                gaze_points_np = np.array(gaze_points)
                centroid = tuple(map(int, np.mean(gaze_points_np, axis=0)))
                cv2.circle(image, centroid, 5, (0, 255, 0), -1) # Green centroid

                # Convert stability (degrees) back to pixels for drawing radius
                stability_pixels = 0
                if dist_cm > 0 and self.pixels_per_cm > 0:
                    distance_pixels = dist_cm * self.pixels_per_cm
                    stability_pixels = int(math.tan(math.radians(fix_stab)) * distance_pixels)

                if stability_pixels > 0:
                    cv2.circle(image, centroid, stability_pixels, (0, 165, 255), 1) # Orange circle

        return image

    def _draw_simple_overlay(self, image, landmarks):
        """Draws a simpler overlay based on display_mode ('face' or 'eyes')."""
        if not landmarks: return image

        if self.display_mode == self.DISPLAY_MODE_FACE:
            self.mp_drawing.draw_landmarks(
                image=image, landmark_list=landmarks, connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None, connection_drawing_spec=self.drawing_spec_tesselation)
            self.mp_drawing.draw_landmarks(
                image=image, landmark_list=landmarks, connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None, connection_drawing_spec=self.drawing_spec)
        elif self.display_mode == self.DISPLAY_MODE_EYES:
            # Draw iris landmarks if refined
            self.mp_drawing.draw_landmarks(
                image=image, landmark_list=landmarks, connections=self.mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None, connection_drawing_spec=self.drawing_spec_iris)
            # Draw eye contours based on processing mode
            if self.eye_processing_mode in [self.EYE_MODE_BOTH, self.EYE_MODE_LEFT]:
                self.mp_drawing.draw_landmarks(
                    image=image, landmark_list=landmarks, connections=self.mp_face_mesh.FACEMESH_LEFT_EYE,
                    landmark_drawing_spec=None, connection_drawing_spec=self.drawing_spec)
            if self.eye_processing_mode in [self.EYE_MODE_BOTH, self.EYE_MODE_RIGHT]:
                self.mp_drawing.draw_landmarks(
                    image=image, landmark_list=landmarks, connections=self.mp_face_mesh.FACEMESH_RIGHT_EYE,
                    landmark_drawing_spec=None, connection_drawing_spec=self.drawing_spec)
        # No drawing if mode is neither face nor eyes (shouldn't happen with validation)

        return image

    def process_frame(self, frame):
        """
        Processes a single video frame.

        Args:
            frame (np.ndarray): The input video frame (BGR format).

        Returns:
            tuple: (output_frame, metrics, face_landmarks_result)
                   - output_frame (np.ndarray): Frame with drawings (BGR).
                   - metrics (dict): Calculated eye metrics. None if no face detected.
                   - face_landmarks_result: The raw MediaPipe face mesh result. None if no face.
        """
        if self.face_mesh is None:
            logging.error("FaceMesh not initialized. Cannot process frame.")
            # Attempt reinitialization? Or just return error state?
            # self._init_face_mesh(...) # Consider parameters if re-initializing
            return frame, None, None # Return original frame, no metrics/results

        # --- Frame Preparation ---
        # Flip horizontally for selfie view, convert BGR to RGB
        frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        # Store frame dimensions if not already set
        if self.frame_height is None or self.frame_width is None:
            self.frame_height, self.frame_width, _ = frame_rgb.shape
            logging.info(f"Frame dimensions set: {self.frame_width}x{self.frame_height}")

        # --- TODO: Deflectometry Enhancement ---
        # if self.deflectometry_enhancer:
        #     frame_rgb = self.deflectometry_enhancer.enhance(frame_rgb)

        # --- MediaPipe Processing ---
        frame_rgb.flags.writeable = False # Performance optimization
        results = self.face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True

        # Prepare output frame (copy of input frame initially)
        output_frame = frame.copy() # Work on a copy of the original BGR frame
        output_frame = cv2.flip(output_frame, 1) # Flip the output frame to match processing

        # --- Metrics Calculation and Drawing ---
        metrics = None
        face_landmarks_proto = None

        if results.multi_face_landmarks:
            # Process the first detected face
            face_landmarks_proto = results.multi_face_landmarks[0]
            landmarks_list = face_landmarks_proto.landmark

            # Calculate metrics
            metrics = self._calculate_metrics(landmarks_list, time.time())

            # --- Drawing ---
            # Create a temporary drawing frame from the RGB processed frame
            # drawing_frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if self.display_mode == self.DISPLAY_MODE_DEBUG:
                output_frame = self._draw_debug_overlay(output_frame, face_landmarks_proto, metrics)
            else: # Handle 'face' or 'eyes' modes
                output_frame = self._draw_simple_overlay(output_frame, face_landmarks_proto)

        return output_frame, metrics, face_landmarks_proto # Return BGR frame

    def set_display_mode(self, mode):
        """Sets the display mode ('face', 'eyes', or 'debug')."""
        valid_modes = [self.DISPLAY_MODE_FACE, self.DISPLAY_MODE_EYES, self.DISPLAY_MODE_DEBUG]
        if mode in valid_modes:
            self.display_mode = mode
            logging.info(f"Display mode set to: {self.display_mode}")
        else:
            logging.warning(f"Invalid display mode '{mode}'. Keeping '{self.display_mode}'. Valid modes: {valid_modes}")

    def set_eye_processing_mode(self, mode):
        """Sets the eye processing mode ('both', 'left', 'right')."""
        valid_modes = [self.EYE_MODE_BOTH, self.EYE_MODE_LEFT, self.EYE_MODE_RIGHT]
        if mode in valid_modes:
            if self.eye_processing_mode != mode:
                self.eye_processing_mode = mode
                # Reset buffers when changing mode to avoid mixing data
                self.gaze_points_buffer_left.clear()
                self.gaze_points_buffer_right.clear()
                self.gaze_points_buffer_avg.clear()
                self.eye_aspect_ratio_buffer_left.clear()
                self.eye_aspect_ratio_buffer_right.clear()
                self.last_pupil_pos_left = None
                self.last_pupil_pos_right = None
                self.last_pupil_pos_avg = None
                logging.info(f"Eye processing mode set to: {self.eye_processing_mode}. Buffers cleared.")
        else:
            logging.warning(f"Invalid eye processing mode '{mode}'. Keeping '{self.eye_processing_mode}'. Valid modes: {valid_modes}")

    def calibrate_distance(self, known_distance_cm, current_landmarks):
        """Performs a simple calibration for distance estimation."""
        # TODO: Implement a more robust calibration (e.g., multiple points)
        logging.info(f"Attempting distance calibration with known distance: {known_distance_cm} cm")
        try:
            left_point = current_landmarks[self.LEFT_EYE_INNER_CORNER]
            right_point = current_landmarks[self.RIGHT_EYE_INNER_CORNER]

            if self.frame_width is None or self.frame_height is None:
                 logging.warning("Cannot calibrate distance without frame dimensions.")
                 return False

            left_pixel = (int(left_point.x * self.frame_width), int(left_point.y * self.frame_height))
            right_pixel = (int(right_point.x * self.frame_width), int(right_point.y * self.frame_height))
            pixel_dist = np.linalg.norm(np.array(left_pixel) - np.array(right_pixel))

            if pixel_dist < 1e-6:
                logging.warning("Cannot calibrate distance: Landmarks too close.")
                return False

            known_eye_distance_cm = 3.5 # Assume this is constant for calibration calculation
            # Calculate focal length based on known distance
            # focal_length = (pixel_width * known_distance) / known_width
            self.eye_focal_length = (pixel_dist * known_distance_cm) / known_eye_distance_cm
            # Reset distance history with the known distance
            self.distance_history.clear()
            for _ in range(self.distance_history.buffer.maxlen):
                self.distance_history.append(known_distance_cm)
            self.estimated_distance = known_distance_cm
            logging.info(f"Distance calibration complete. New focal length: {self.eye_focal_length:.2f}, Estimated distance reset to: {self.estimated_distance:.1f} cm")
            return True

        except IndexError:
             logging.warning("Calibration failed: Inner eye corner landmarks not found.")
             return False
        except Exception as e:
            logging.error(f"Error during distance calibration: {e}")
            return False

    def calibrate_pixels_per_cm(self, screen_width_cm, screen_height_cm):
        """Calibrates pixels_per_cm based on screen dimensions."""
        if self.frame_width and self.frame_height:
            px_per_cm_w = self.frame_width / screen_width_cm
            px_per_cm_h = self.frame_height / screen_height_cm
            self.pixels_per_cm = (px_per_cm_w + px_per_cm_h) / 2 # Average
            logging.info(f"Pixels per cm calibrated to: {self.pixels_per_cm:.2f}")
            return True
        else:
            logging.warning("Cannot calibrate pixels per cm without frame dimensions.")
            return False

    def close(self):
        """Releases MediaPipe resources safely."""
        if self.face_mesh:
            try:
                self.face_mesh.close()
                logging.info("Successfully closed MediaPipe FaceMesh")
            except Exception as e:
                logging.warning(f"Error closing face_mesh: {e}")
            finally:
                self.face_mesh = None

    def reset(self, max_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """Reinitializes MediaPipe resources and resets state."""
        logging.info("Resetting EyeTracker...")
        self.close()
        self._init_face_mesh(max_faces, refine_landmarks, min_detection_confidence, min_tracking_confidence)
        # Reset state variables
        self.last_timestamp = time.time()
        self.start_time = time.time()
        self.pupil_pos_left = None
        self.pupil_pos_right = None
        self.last_pupil_pos_left = None
        self.last_pupil_pos_right = None
        self.last_pupil_pos_avg = None
        self.gaze_points_buffer_left.clear()
        self.gaze_points_buffer_right.clear()
        self.gaze_points_buffer_avg.clear()
        self.blink_timestamps.clear()
        self.eye_aspect_ratio_buffer_left.clear()
        self.eye_aspect_ratio_buffer_right.clear()
        self.distance_history.clear()
        self.saccade_velocity = 0.0
        self.fixation_stability = 0.0
        self.blink_rate = 0.0
        self.estimated_distance = 50.0 # Reset to default guess
        self.eye_focal_length = None # Recalculate on next frame
        logging.info("EyeTracker reset complete.")

# Example Usage (for testing within this file)
if __name__ == '__main__':
    print("Running EyeTracker example...")
    tracker = EyeTracker(refine_landmarks=True)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    frame_count = 0
    fps_start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        output_frame, metrics, _ = tracker.process_frame(frame)

        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - fps_start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(output_frame, f"FPS: {fps:.1f}", (output_frame.shape[1] - 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display metrics if available (already drawn in debug mode)
        # if metrics and tracker.display_mode != tracker.DISPLAY_MODE_DEBUG:
        #     y_offset = 30
        #     for key, value in metrics.items():
        #         if value is not None:
        #             text = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
        #             cv2.putText(output_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        #             y_offset += 18

        cv2.imshow('WorkingGenomeGuard - Eye Tracker', output_frame)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'): # Cycle display mode
            current_mode = tracker.display_mode
            if current_mode == tracker.DISPLAY_MODE_DEBUG:
                tracker.set_display_mode(tracker.DISPLAY_MODE_FACE)
            elif current_mode == tracker.DISPLAY_MODE_FACE:
                tracker.set_display_mode(tracker.DISPLAY_MODE_EYES)
            else: # Eyes -> Debug
                tracker.set_display_mode(tracker.DISPLAY_MODE_DEBUG)
        elif key == ord('b'): # Cycle eye processing mode
            current_proc_mode = tracker.eye_processing_mode
            if current_proc_mode == tracker.EYE_MODE_BOTH:
                tracker.set_eye_processing_mode(tracker.EYE_MODE_LEFT)
            elif current_proc_mode == tracker.EYE_MODE_LEFT:
                tracker.set_eye_processing_mode(tracker.EYE_MODE_RIGHT)
            else: # Right -> Both
                tracker.set_eye_processing_mode(tracker.EYE_MODE_BOTH)
        elif key == ord('c'): # Simulate calibration
             print("Simulating distance calibration at 60cm...")
             # In a real app, you'd get landmarks from the current frame
             # For simulation, we just trigger the logic assuming landmarks are available
             # tracker.calibrate_distance(60.0, tracker.last_landmarks_processed_or_similar) # Need landmarks
             print("Calibration requires landmarks - press 'c' when face is detected and stable at known distance.")


    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("EyeTracker example finished.")
