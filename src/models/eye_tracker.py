import cv2
import logging
import mediapipe as mp
import numpy as np
import time
import math
from ..utils.cycle_buffer import CycleBuffer # Use relative import

class EyeTracker:
    """
    Handles real-time eye tracking using MediaPipe Face Mesh, calculates
    relevant eye metrics, and manages different processing and display modes.
    """
    # Define constants for eye processing modes
    EYE_MODE_BOTH = 'both'
    EYE_MODE_LEFT = 'left'
    EYE_MODE_RIGHT = 'right'
    def __init__(self, max_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initializes the EyeTracker with MediaPipe Face Mesh.

        Args:
            max_faces (int): Maximum number of faces to detect.
            refine_landmarks (bool): Whether to refine eye/lip landmarks.
            min_detection_confidence (float): Minimum confidence for face detection.
            min_tracking_confidence (float): Minimum confidence for tracking.
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        # State variables for metric calculation
        self.last_timestamp = time.time()
        self.last_left_eye_center = None
        self.last_right_eye_center = None
        self.left_eye_positions = CycleBuffer(maxlen=10) # Buffer for stability/saccade
        self.right_eye_positions = CycleBuffer(maxlen=10)
        self.blink_timestamps = CycleBuffer(maxlen=20) # Store timestamps of blinks
        self.eye_aspect_ratio_buffer = CycleBuffer(maxlen=5) # For blink detection smoothing

        # Display mode management ('face', 'eyes') - Controls what landmarks are drawn
        self.display_mode = 'face' # Default display mode

        # Processing mode management ('both', 'left', 'right') - Controls which eye(s) metrics are based on
        self.eye_processing_mode = self.EYE_MODE_BOTH # Default to process both eyes

        # Debug mode - If True, shows frame without overlay
        self.debug_mode = False

    def _calculate_eye_center(self, landmarks, eye_indices):
        """Calculates the geometric center of an eye based on specific landmarks."""
        points = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])
        center = np.mean(points, axis=0)
        return center

    def _calculate_eye_aspect_ratio(self, landmarks, eye_indices):
        """Calculates the Eye Aspect Ratio (EAR) for blink detection."""
        # Example indices for vertical and horizontal distances (adjust based on MediaPipe landmarks)
        # These indices need to be verified against the actual MediaPipe Face Mesh landmark map
        # Vertical landmarks (example: top/bottom points of the eye opening)
        v1 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y]) # Top point
        v2 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y]) # Point below top
        v3 = np.array([landmarks[eye_indices[8]].x, landmarks[eye_indices[8]].y]) # Point above bottom
        v4 = np.array([landmarks[eye_indices[12]].x, landmarks[eye_indices[12]].y]) # Bottom point
        # Horizontal landmarks (example: left/right corners)
        h1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y]) # Left corner
        h2 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y]) # Right corner

        # Calculate Euclidean distances
        ver_dist1 = np.linalg.norm(v1 - v4)
        ver_dist2 = np.linalg.norm(v2 - v3)
        hor_dist = np.linalg.norm(h1 - h2)

        # Avoid division by zero
        if hor_dist == 0:
            return 0.0

        # Compute EAR
        ear = (ver_dist1 + ver_dist2) / (2.0 * hor_dist)
        return ear

    def _calculate_metrics(self, landmarks, frame_shape):
        """
        Calculates various eye metrics based on landmarks and the current
        eye_processing_mode.
        """
        metrics = {
            'saccade_velocity': 0.0,
            'fixation_stability': 0.0,
            'blink_rate': 0.0,
            'vertical_saccade_velocity': 0.0,
            'horizontal_saccade_velocity': 0.0,
            'eye_aspect_ratio_left': None, # Use None to indicate not calculated/applicable
            'eye_aspect_ratio_right': None,
            'is_blinking': False,
            'active_eye_mode': self.eye_processing_mode # Include mode in metrics
        }
        timestamp = time.time()
        delta_time = timestamp - self.last_timestamp
        if delta_time == 0: delta_time = 1e-6 # Avoid division by zero

        # --- Define MediaPipe Landmark Indices (These are crucial and need verification) ---
        # You MUST verify these indices against the official MediaPipe Face Mesh documentation
        # https://developers.google.com/mediapipe/solutions/vision/face_mesh#face_mesh_connections_and_landmark_indices
        LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144, 163, 7, 173, 159, 157, 154, 145, 161] # Example set
        RIGHT_EYE_INDICES = [362, 387, 385, 384, 398, 373, 390, 249, 463, 386, 388, 374, 380, 381] # Example set
        LEFT_IRIS_INDICES = [474, 475, 476, 477] # Refined landmarks needed
        RIGHT_IRIS_INDICES = [469, 470, 471, 472] # Refined landmarks needed

        # Calculate Eye Centers based on mode
        left_eye_center = None
        right_eye_center = None
        if self.eye_processing_mode in [self.EYE_MODE_BOTH, self.EYE_MODE_LEFT]:
            left_eye_center = self._calculate_eye_center(landmarks, LEFT_EYE_INDICES)
        if self.eye_processing_mode in [self.EYE_MODE_BOTH, self.EYE_MODE_RIGHT]:
            right_eye_center = self._calculate_eye_center(landmarks, RIGHT_EYE_INDICES)

        # --- Saccade Velocity ---
        # --- Saccade Velocity (adjust based on available eyes) ---
        valid_last_centers = (self.last_left_eye_center is not None if self.eye_processing_mode != self.EYE_MODE_RIGHT else True) and \
                             (self.last_right_eye_center is not None if self.eye_processing_mode != self.EYE_MODE_LEFT else True)

        if valid_last_centers and (left_eye_center is not None or right_eye_center is not None):
            delta_pixels = 0.0
            delta_y_pix = 0.0
            delta_x_pix = 0.0
            num_eyes = 0

            if self.eye_processing_mode in [self.EYE_MODE_BOTH, self.EYE_MODE_LEFT] and left_eye_center is not None and self.last_left_eye_center is not None:
                delta_left = np.linalg.norm(left_eye_center - self.last_left_eye_center)
                delta_pixels += delta_left
                delta_y_pix += abs(left_eye_center[1] - self.last_left_eye_center[1])
                delta_x_pix += abs(left_eye_center[0] - self.last_left_eye_center[0])
                num_eyes += 1
            if self.eye_processing_mode in [self.EYE_MODE_BOTH, self.EYE_MODE_RIGHT] and right_eye_center is not None and self.last_right_eye_center is not None:
                delta_right = np.linalg.norm(right_eye_center - self.last_right_eye_center)
                delta_pixels += delta_right
                delta_y_pix += abs(right_eye_center[1] - self.last_right_eye_center[1])
                delta_x_pix += abs(right_eye_center[0] - self.last_right_eye_center[0])
                num_eyes += 1

            if num_eyes > 0:
                avg_delta_pixels = delta_pixels / num_eyes
                avg_delta_y = delta_y_pix / num_eyes
                avg_delta_x = delta_x_pix / num_eyes

            # Convert pixel distance to visual angle (requires camera calibration/estimation)
            # Placeholder: Assume a simple scaling factor for now
            pixels_per_degree = 50 # VERY ROUGH ESTIMATE - NEEDS CALIBRATION
            delta_angle = avg_delta_pixels / pixels_per_degree

            metrics['saccade_velocity'] = delta_angle / delta_time # Degrees per second

            # Vertical/Horizontal components (Move inside the 'if num_eyes > 0' block)
            metrics['vertical_saccade_velocity'] = (avg_delta_y / pixels_per_degree) / delta_time
            metrics['horizontal_saccade_velocity'] = (avg_delta_x / pixels_per_degree) / delta_time


        # --- Fixation Stability ---
        # --- Fixation Stability (adjust based on available eyes) ---
        stability = 0.0
        num_eyes_stability = 0
        if self.eye_processing_mode in [self.EYE_MODE_BOTH, self.EYE_MODE_LEFT] and left_eye_center is not None:
            self.left_eye_positions.append(left_eye_center)
            if len(self.left_eye_positions) > 1:
                left_std_dev = np.std(self.left_eye_positions.get_all(), axis=0)
                stability += np.linalg.norm(left_std_dev)
                num_eyes_stability += 1
        if self.eye_processing_mode in [self.EYE_MODE_BOTH, self.EYE_MODE_RIGHT] and right_eye_center is not None:
            self.right_eye_positions.append(right_eye_center)
            if len(self.right_eye_positions) > 1:
                right_std_dev = np.std(self.right_eye_positions.get_all(), axis=0)
                stability += np.linalg.norm(right_std_dev)
                num_eyes_stability += 1

        if num_eyes_stability > 0:
             metrics['fixation_stability'] = stability / num_eyes_stability

        # --- Blink Detection & Rate ---
        # --- Blink Detection & Rate (adjust based on available eyes) ---
        ear_sum = 0.0
        ear_count = 0
        if self.eye_processing_mode in [self.EYE_MODE_BOTH, self.EYE_MODE_LEFT]:
            metrics['eye_aspect_ratio_left'] = self._calculate_eye_aspect_ratio(landmarks, LEFT_EYE_INDICES)
            ear_sum += metrics['eye_aspect_ratio_left']
            ear_count += 1
        if self.eye_processing_mode in [self.EYE_MODE_BOTH, self.EYE_MODE_RIGHT]:
             metrics['eye_aspect_ratio_right'] = self._calculate_eye_aspect_ratio(landmarks, RIGHT_EYE_INDICES)
             ear_sum += metrics['eye_aspect_ratio_right']
             ear_count += 1

        if ear_count > 0:
            avg_ear = ear_sum / ear_count
            self.eye_aspect_ratio_buffer.append(avg_ear)

            # Simple blink detection: if EAR drops below a threshold
            EAR_THRESHOLD = 0.2 # Adjust based on testing
            # Check mean EAR only if buffer is full to avoid detecting blinks on startup
            if len(self.eye_aspect_ratio_buffer) == self.eye_aspect_ratio_buffer.buffer.maxlen and self.eye_aspect_ratio_buffer.mean() < EAR_THRESHOLD:
                 metrics['is_blinking'] = True
                 # Record blink timestamp only once per blink event (simple debounce)
                 if not self.blink_timestamps.get_all() or (timestamp - self.blink_timestamps.get_all()[-1] > 0.5): # 500ms refractory period
                     self.blink_timestamps.append(timestamp)


        # Calculate blink rate (blinks per minute)
        if len(self.blink_timestamps) > 1:
            time_span_seconds = self.blink_timestamps.get_all()[-1] - self.blink_timestamps.get_all()[0]
            if time_span_seconds > 1: # Need at least 1 second span
                 blink_count = len(self.blink_timestamps)
                 metrics['blink_rate'] = (blink_count / time_span_seconds) * 60.0


        # Update last known state
        # Update last known state only for the processed eyes
        self.last_timestamp = timestamp
        if left_eye_center is not None:
            self.last_left_eye_center = left_eye_center
        if right_eye_center is not None:
            self.last_right_eye_center = right_eye_center

        return metrics

    def set_display_mode(self, mode):
        """Sets the display mode ('face' or 'eyes')."""
        if mode in ['face', 'eyes']:
            self.display_mode = mode
            print(f"EyeTracker display mode set to: {self.display_mode}")
        else:
            print(f"Warning: Invalid display mode '{mode}'. Keeping mode '{self.display_mode}'.")

    def set_eye_processing_mode(self, mode):
        """Sets the eye processing mode ('both', 'left', 'right')."""
        if mode in [self.EYE_MODE_BOTH, self.EYE_MODE_LEFT, self.EYE_MODE_RIGHT]:
            self.eye_processing_mode = mode
            # Reset buffers when changing mode to avoid mixing data? Optional.
            self.left_eye_positions = CycleBuffer(maxlen=self.left_eye_positions.buffer.maxlen)
            self.right_eye_positions = CycleBuffer(maxlen=self.right_eye_positions.buffer.maxlen)
            self.last_left_eye_center = None
            self.last_right_eye_center = None
            print(f"EyeTracker processing mode set to: {self.eye_processing_mode}")
        else:
            print(f"Warning: Invalid eye processing mode '{mode}'. Keeping mode '{self.eye_processing_mode}'.")

    def set_debug_mode(self, enabled: bool):
        """Enables or disables debug mode (no overlay)."""
        self.debug_mode = enabled
        print(f"EyeTracker debug mode set to: {self.debug_mode}")

    def process_frame(self, frame):
        """
        Processes a single video frame to detect face/eyes and calculate metrics.
        Automatically reinitializes resources if needed.

        Args:
            frame (np.ndarray): The input video frame (BGR format).

        Returns:
            tuple: (output_frame, original_frame_rgb, metrics, face_landmarks_result)
                   - output_frame (np.ndarray): Frame with drawings (or original if debug_mode). BGR format.
                   - original_frame_rgb (np.ndarray): The frame after flipping and RGB conversion, before drawing.
                   - metrics (dict): Calculated eye metrics. None if no face detected.
                   - face_landmarks_result: The raw MediaPipe face mesh result. None if no face.
        """
        # Flip the frame horizontally for a later selfie-view display
        # and convert the BGR image to RGB.
        # Initialize face mesh if needed
        if not hasattr(self, 'face_mesh') or self.face_mesh is None:
            logging.debug("Reinitializing MediaPipe FaceMesh")
            self.__init_face_mesh()

        frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame_rgb.flags.writeable = False
        results = self.face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True

        # Convert the image back to BGR for drawing.
        # Keep a copy of the RGB frame before drawing for debug mode
        original_frame_rgb = frame_rgb.copy()
        # Convert the working frame back to BGR for drawing.
        drawing_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        metrics = None
        face_landmarks_result = None

        if results.multi_face_landmarks:
            face_landmarks_result = results.multi_face_landmarks[0] # Process first detected face
            landmarks_list = face_landmarks_result.landmark

            # --- Drawing based on display_mode (only if debug_mode is False) ---
            if not self.debug_mode:
                if self.display_mode == 'face':
                    # Draw face mesh connections
                    self.mp_drawing.draw_landmarks(
                        image=drawing_frame,
                        landmark_list=face_landmarks_result,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(128,128,128), thickness=1) # Gray lines
                    )
                    # Draw contours for face, eyes, lips etc.
                    self.mp_drawing.draw_landmarks(
                        image=drawing_frame,
                        landmark_list=face_landmarks_result,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.drawing_spec
                     )

                elif self.display_mode == 'eyes':
                     # Draw only eye-related landmarks/connections
                     # Draw iris landmarks if refined
                     if self.mp_face_mesh.FACEMESH_IRISES:
                         self.mp_drawing.draw_landmarks(
                             image=drawing_frame,
                             landmark_list=face_landmarks_result,
                             connections=self.mp_face_mesh.FACEMESH_IRISES,
                             landmark_drawing_spec=self.drawing_spec, # Draw iris points
                             connection_drawing_spec=self.drawing_spec
                         )
                     # Draw eye contours based on processing mode
                     if self.eye_processing_mode in [self.EYE_MODE_BOTH, self.EYE_MODE_LEFT]:
                         self.mp_drawing.draw_landmarks(
                             image=drawing_frame,
                             landmark_list=face_landmarks_result,
                             connections=self.mp_face_mesh.FACEMESH_LEFT_EYE,
                             landmark_drawing_spec=None,
                             connection_drawing_spec=self.drawing_spec
                         )
                     if self.eye_processing_mode in [self.EYE_MODE_BOTH, self.EYE_MODE_RIGHT]:
                         self.mp_drawing.draw_landmarks(
                             image=drawing_frame,
                             landmark_list=face_landmarks_result,
                             connections=self.mp_face_mesh.FACEMESH_RIGHT_EYE,
                             landmark_drawing_spec=None,
                             connection_drawing_spec=self.drawing_spec
                         )


            # --- Calculate Metrics ---
            metrics = self._calculate_metrics(landmarks_list, drawing_frame.shape)

        # Determine the output frame based on debug mode
        output_frame = cv2.cvtColor(original_frame_rgb, cv2.COLOR_RGB2BGR) if self.debug_mode else drawing_frame

        return output_frame, original_frame_rgb, metrics, face_landmarks_result
    def __init_face_mesh(self):
        """Initializes MediaPipe FaceMesh with current settings."""
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def close(self):
        """Releases MediaPipe resources safely."""
        if hasattr(self, 'face_mesh') and self.face_mesh is not None:
            try:
                self.face_mesh.close()
                logging.debug("Successfully closed MediaPipe FaceMesh")
            except Exception as e:
                logging.warning(f"Error closing face_mesh: {e}")
            finally:
                self.face_mesh = None

    def reset(self):
        """Reinitializes MediaPipe resources for restart."""
        self.close()
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
# Example Usage (for testing)
if __name__ == '__main__':
    tracker = EyeTracker(refine_landmarks=True) # Refine landmarks for iris tracking
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        output_frame, _, metrics, _ = tracker.process_frame(frame) # Get the final output frame

        # Display metrics on the output frame (even in debug mode)
        if metrics:
            y_offset = 30
            for key, value in metrics.items():
                 if value is not None: # Check if metric was calculated
                     if isinstance(value, float):
                         text = f"{key}: {value:.2f}"
                     else:
                         text = f"{key}: {value}"
                     cv2.putText(output_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                     y_offset += 20

        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(output_frame, f"FPS: {fps:.2f}", (output_frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


        cv2.imshow('GenomeGuard - Eye Tracker Test', output_frame)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'): # Toggle display mode
            current_mode = tracker.display_mode
            tracker.set_display_mode('eyes' if current_mode == 'face' else 'face')
        elif key == ord('d'): # Toggle debug mode
            tracker.set_debug_mode(not tracker.debug_mode)
        elif key == ord('b'): # Cycle eye processing mode
            current_proc_mode = tracker.eye_processing_mode
            if current_proc_mode == tracker.EYE_MODE_BOTH:
                tracker.set_eye_processing_mode(tracker.EYE_MODE_LEFT)
            elif current_proc_mode == tracker.EYE_MODE_LEFT:
                tracker.set_eye_processing_mode(tracker.EYE_MODE_RIGHT)
            else: # Right -> Both
                tracker.set_eye_processing_mode(tracker.EYE_MODE_BOTH)


    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
