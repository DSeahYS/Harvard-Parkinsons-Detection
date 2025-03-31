import cv2
import mediapipe as mp
import numpy as np
import time

class EyeTracker:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Required for iris detection
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("Initialized EyeTracker with iris detection enabled")  # Debug print
        self._log_count = 0  # Counter for logging frequency
        
        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Define eye landmarks indices
        # Left eye landmarks
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        # Right eye landmarks
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        # Iris landmarks
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        # Buffer for velocity calculations
        self.prev_landmarks = None
        self.prev_timestamp = None
        self.saccade_velocities = []
        self.fixation_positions = []
        
    def process_frame(self, frame):
        """Process a single frame and extract eye metrics"""
        current_timestamp = time.time()
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        # Add tracking indicator text
        cv2.putText(frame, "Eye Tracking Status", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        metrics = {}
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Display "Eyes Detected" in green
            cv2.putText(frame, "Eyes Detected", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Extract eye landmarks
            left_eye = self._extract_eye_landmarks(face_landmarks, self.LEFT_EYE, w, h)
            right_eye = self._extract_eye_landmarks(face_landmarks, self.RIGHT_EYE, w, h)
            left_iris = self._extract_eye_landmarks(face_landmarks, self.LEFT_IRIS, w, h)
            right_iris = self._extract_eye_landmarks(face_landmarks, self.RIGHT_IRIS, w, h)
            
            # Draw eye regions more prominently
            for point in left_eye:
                x, y = int(point[0]), int(point[1])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
            for point in right_eye:
                x, y = int(point[0]), int(point[1])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            # Highlight iris points with larger bright circles
            for point in left_iris:
                x, y = int(point[0]), int(point[1])
                cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
                
            for point in right_iris:
                x, y = int(point[0]), int(point[1])
                cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
            
            # Calculate eye metrics
            metrics = self._calculate_eye_metrics(left_eye, right_eye, left_iris, right_iris, current_timestamp)
            
            # Show active metrics on frame
            if 'avg_saccade_velocity' in metrics:
                vel = metrics['avg_saccade_velocity']
                cv2.putText(frame, f"Saccade: {vel:.1f}°/s", (20, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if 'fixation_stability' in metrics:
                stab = metrics['fixation_stability']
                cv2.putText(frame, f"Fixation: {stab:.4f}", (20, 120),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Draw landmarks on the frame for visualization
            self._draw_landmarks(frame, face_landmarks, w, h)
            
            # Store current landmarks for velocity calculation in next frame
            self.prev_landmarks = {
                'left_iris': left_iris,
                'right_iris': right_iris
            }
            self.prev_timestamp = current_timestamp
        else:
            # Display "No eyes detected" in red
            cv2.putText(frame, "No eyes detected", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add troubleshooting tips
            tips = ["Check lighting", "Face the camera", "Remove obstacles"]
            y_pos = 90
            for tip in tips:
                cv2.putText(frame, tip, (30, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # Log eye movements every 5 frames to avoid spamming console
        self._log_count += 1
        if self._log_count % 5 == 0:
            self._log_eye_movements(metrics)
        
            y_pos += 30
        
        # Add FPS counter
        if hasattr(self, 'prev_timestamp') and self.prev_timestamp:
            fps = 1.0 / (current_timestamp - self.prev_timestamp)
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame, metrics
    
    def _extract_eye_landmarks(self, face_landmarks, indices, frame_width, frame_height):
        """Extract and normalize landmark coordinates"""
        landmarks = []
        for idx in indices:
            landmark = face_landmarks.landmark[idx]
            # Convert normalized coordinates to pixel coordinates
            x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
            landmarks.append((x, y, landmark.z))
        return np.array(landmarks)
    
    def _calculate_eye_metrics(self, left_eye, right_eye, left_iris, right_iris, current_timestamp):
        """Calculate eye metrics relevant to Parkinson's detection"""
        metrics = {}
        
        # Calculate EAR (Eye Aspect Ratio)
        left_ear = self._calculate_ear(left_eye)
        right_ear = self._calculate_ear(right_eye)
        metrics['left_ear'] = left_ear
        metrics['right_ear'] = right_ear
        metrics['avg_ear'] = (left_ear + right_ear) / 2
        
        # Calculate iris positions
        left_iris_pos = np.mean(left_iris[:, :2], axis=0) if len(left_iris) > 0 else np.array([0, 0])
        right_iris_pos = np.mean(right_iris[:, :2], axis=0) if len(right_iris) > 0 else np.array([0, 0])
        metrics['left_iris_position'] = left_iris_pos
        metrics['right_iris_position'] = right_iris_pos
        
        # Add to fixation positions buffer
        self.fixation_positions.append(left_iris_pos)
        if len(self.fixation_positions) > 30:  # Keep 30 frames for fixation stability
            self.fixation_positions.pop(0)
        
        # Calculate fixation stability (variance of positions)
        if len(self.fixation_positions) > 5:
            fixation_array = np.array(self.fixation_positions)
            metrics['fixation_stability'] = np.mean(np.var(fixation_array, axis=0))
        
        # Calculate saccade velocity if we have previous landmarks
        if self.prev_landmarks is not None and self.prev_timestamp is not None:
            time_diff = current_timestamp - self.prev_timestamp
            
            # Calculate displacement
            prev_left_iris = np.mean(self.prev_landmarks['left_iris'][:, :2], axis=0)
            displacement = np.linalg.norm(left_iris_pos - prev_left_iris)
            
            # Convert to degrees (approximate)
            # Assuming 1 pixel ≈ 0.05 degrees of visual angle
            degrees_displacement = displacement * 0.05
            
            # Calculate velocity in degrees/second
            velocity = degrees_displacement / time_diff
            
            # Store velocity if it's likely a saccade (>50 deg/s)
            if velocity > 50:
                self.saccade_velocities.append(velocity)
                if len(self.saccade_velocities) > 10:  # Keep last 10 saccades
                    self.saccade_velocities.pop(0)
            
            metrics['current_velocity'] = velocity
            
            # Calculate average saccade velocity
            if self.saccade_velocities:
                metrics['avg_saccade_velocity'] = np.mean(self.saccade_velocities)
        
        return metrics
    
    def _calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio (EAR)"""
        if len(eye_landmarks) < 6:
            return 0
            
        # Use landmarks for vertical and horizontal measurements
        # This is a simplified version
        vertical_1 = np.linalg.norm(eye_landmarks[1][:2] - eye_landmarks[5][:2])
        vertical_2 = np.linalg.norm(eye_landmarks[2][:2] - eye_landmarks[4][:2])
        horizontal = np.linalg.norm(eye_landmarks[0][:2] - eye_landmarks[3][:2])
        
        # Avoid division by zero
        if horizontal == 0:
            return 0
            
        return (vertical_1 + vertical_2) / (2.0 * horizontal)
    
    def _draw_landmarks(self, frame, face_landmarks, width, height):
        """Draw facial landmarks on the frame"""
        # Draw mesh
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        
        # Highlight eye landmarks
        for idx in self.LEFT_EYE + self.RIGHT_EYE:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
        # Highlight iris landmarks in a different color
        for idx in self.LEFT_IRIS + self.RIGHT_IRIS:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
    def _log_eye_movements(self, metrics):
        """Log eye movement metrics to console"""
        if not metrics:
            return
            
        movement_log = "Eye Tracking: "
        if 'avg_saccade_velocity' in metrics:
            movement_log += f"Saccade: {metrics['avg_saccade_velocity']:.2f}°/s | "
        if 'fixation_stability' in metrics:
            movement_log += f"Fixation: {metrics['fixation_stability']:.4f} | "
        if 'avg_ear' in metrics:
            movement_log += f"EAR: {metrics['avg_ear']:.3f}"
            
        print(movement_log)

