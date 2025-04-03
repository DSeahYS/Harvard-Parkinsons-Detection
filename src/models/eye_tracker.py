import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from src.utils.cycle_buffer import CycleBuffer

class EyeTracker:
    def __init__(self, mode='face', buffer_size=60):
        """Initialize eye tracker with adaptive face mesh configuration"""
        self.current_mode = mode
        self.mode_history = deque(maxlen=10)
        self.metrics_history = []
        self.buffer_size = buffer_size
        self.initialize_buffers()
        
        # MediaPipe initialization
        self.mp_face_mesh = mp.solutions.face_mesh
        self._init_face_mesh()
        
        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Eye landmarks indices
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        # Buffer for velocity calculations
        self.prev_landmarks = None
        self.prev_timestamp = None
        self.saccade_velocities = []
        self.vertical_saccade_velocities = []
        self.fixation_positions = []
        self._log_count = 0
    
    def _init_face_mesh(self):
        """Reinitialize face mesh with current mode parameters"""
        if hasattr(self, 'face_mesh') and self.face_mesh:
            self.face_mesh.close()
        
        config = {
            'max_num_faces': 1,
            'min_detection_confidence': 0.3 if self.current_mode == 'eye' else 0.5,
            'min_tracking_confidence': 0.3 if self.current_mode == 'eye' else 0.5,
            'refine_landmarks': False if self.current_mode == 'eye' else True
        }
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(**config)
    
    def initialize_buffers(self):
        self.velocity_buffer = CycleBuffer(self.buffer_size)
        self.fixation_buffer = CycleBuffer(self.buffer_size)
        self.saccade_buffer = CycleBuffer(self.buffer_size)
    
    def process_frame(self, frame, debug_mode=False):
        """Process frame with adaptive mode switching"""
        try:
            # Check if eye mode needs activation
            if self._should_activate_eye_mode(frame):
                self.current_mode = 'eye'
                self._init_face_mesh()
            
            if self.current_mode == 'face':
                return self._process_face_mode(frame, debug_mode)
            else:
                return self._process_eye_mode(frame, debug_mode)
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame, {}
    
    def _should_activate_eye_mode(self, frame):
        """Detect if user is close enough for single-eye scanning"""
        h, w = frame.shape[:2]
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            min_y = min(lm.y * h for lm in face_landmarks.landmark)
            max_y = max(lm.y * h for lm in face_landmarks.landmark)
            face_height = max_y - min_y
            return face_height > h * 0.5
        return False
    
    def _process_face_mode(self, frame, debug_mode):
        """Process frame in face tracking mode"""
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        metrics = {}
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract eye landmarks
            left_eye = self._extract_eye_landmarks(face_landmarks, self.LEFT_EYE, w, h)
            right_eye = self._extract_eye_landmarks(face_landmarks, self.RIGHT_EYE, w, h)
            left_iris = self._extract_eye_landmarks(face_landmarks, self.LEFT_IRIS, w, h)
            right_iris = self._extract_eye_landmarks(face_landmarks, self.RIGHT_IRIS, w, h)
            
            # Calculate metrics
            metrics = self._calculate_eye_metrics(left_eye, right_eye, left_iris, right_iris, time.time())
            
            if debug_mode:
                self._draw_debug_info(frame, face_landmarks, w, h, metrics)
        
        return frame, metrics
    
    def _process_eye_mode(self, frame, debug_mode):
        """Process frame in single-eye tracking mode"""
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        metrics = {}
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract iris landmarks only
            left_iris = self._extract_eye_landmarks(face_landmarks, self.LEFT_IRIS, w, h)
            right_iris = self._extract_eye_landmarks(face_landmarks, self.RIGHT_IRIS, w, h)
            
            # Use dominant eye in eye mode
            primary_iris = left_iris if len(left_iris) > 0 else right_iris
            if len(primary_iris) >= 4:
                # Create normalized coordinates relative to iris center
                iris_center = np.mean(primary_iris[:, :2], axis=0)
                frame_center = np.array([w//2, h//2])
                
                # Calculate offset from frame center
                offset = iris_center - frame_center
                metrics['iris_offset_x'] = offset[0] / w  # Normalized
                metrics['iris_offset_y'] = offset[1] / h
                
                # Calculate metrics
                metrics.update(self._calculate_eye_metrics([], [], left_iris, right_iris, time.time()))
            
            if debug_mode:
                self._draw_eye_mode_debug(frame, w, h, metrics)
        
        return frame, metrics
    
    def _calculate_eye_metrics(self, left_eye, right_eye, left_iris, right_iris, current_timestamp):
        """Calculate eye metrics relevant to Parkinson's detection"""
        metrics = {
            'timestamp': current_timestamp,
            'mode': self.current_mode
        }
        
        # Calculate EAR (Eye Aspect Ratio) if in face mode
        if self.current_mode == 'face':
            left_ear = self._calculate_ear(left_eye)
            right_ear = self._calculate_ear(right_eye)
            metrics.update({
                'left_ear': left_ear,
                'right_ear': right_ear,
                'avg_ear': (left_ear + right_ear) / 2
            })
        
        # Calculate iris positions
        left_iris_pos = np.mean(left_iris[:, :2], axis=0) if len(left_iris) > 0 else np.array([0, 0])
        right_iris_pos = np.mean(right_iris[:, :2], axis=0) if len(right_iris) > 0 else np.array([0, 0])
        
        metrics.update({
            'left_iris_position': left_iris_pos.tolist(),
            'right_iris_position': right_iris_pos.tolist()
        })
        
        # Add to fixation positions buffer
        self.fixation_positions.append(left_iris_pos)
        if len(self.fixation_positions) > 30:
            self.fixation_positions.pop(0)
        
        # Calculate fixation stability (variance of positions)
        if len(self.fixation_positions) > 5:
            fixation_array = np.array(self.fixation_positions)
            metrics['fixation_stability'] = float(np.mean(np.var(fixation_array, axis=0)))
        
        # Calculate saccade velocity if we have previous landmarks
        if self.prev_landmarks is not None and self.prev_timestamp is not None:
            time_diff = current_timestamp - self.prev_timestamp
            
            if time_diff > 0:  # Prevent division by zero
                # Calculate overall and vertical displacement
                prev_left_iris = np.mean(self.prev_landmarks['left_iris'][:, :2], axis=0)
                displacement_vector = left_iris_pos - prev_left_iris
                displacement = np.linalg.norm(displacement_vector)
                displacement_y = abs(displacement_vector[1])  # Vertical component
                
                # Convert to degrees (approximate)
                degrees_displacement = displacement * 0.05
                degrees_displacement_y = displacement_y * 0.05
                
                # Calculate velocities
                velocity = degrees_displacement / time_diff
                velocity_y = degrees_displacement_y / time_diff
                
                # Store velocities if it's likely a saccade (>50 deg/s)
                if velocity > 50:
                    self.saccade_velocities.append(velocity)
                    if len(self.saccade_velocities) > 10:
                        self.saccade_velocities.pop(0)
                    
                    self.vertical_saccade_velocities.append(velocity_y)
                    if len(self.vertical_saccade_velocities) > 10:
                        self.vertical_saccade_velocities.pop(0)
                
                metrics.update({
                    'current_velocity': float(velocity),
                    'current_velocity_y': float(velocity_y)
                })
                
                # Calculate average velocities
                if self.saccade_velocities:
                    metrics['avg_saccade_velocity'] = float(np.mean(self.saccade_velocities))
                
                if self.vertical_saccade_velocities:
                    metrics['avg_vertical_saccade_velocity'] = float(np.mean(self.vertical_saccade_velocities))
        
        # Store current landmarks for next frame
        self.prev_landmarks = {
            'left_iris': left_iris,
            'right_iris': right_iris
        }
        
        self.prev_timestamp = current_timestamp
        
        return metrics
    
    def _draw_debug_info(self, frame, face_landmarks, w, h, metrics):
        """Draw debug information in face mode"""
        # Draw eye regions
        for idx in self.LEFT_EYE + self.RIGHT_EYE:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        # Highlight iris points
        for idx in self.LEFT_IRIS + self.RIGHT_IRIS:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
        
        # Show metrics
        if 'avg_saccade_velocity' in metrics:
            vel = metrics['avg_saccade_velocity']
            cv2.putText(frame, f"Saccade: {vel:.1f}°/s", (20, 90),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if 'fixation_stability' in metrics:
            stab = metrics['fixation_stability']
            cv2.putText(frame, f"Fixation: {stab:.4f}", (20, 120),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def _draw_eye_mode_debug(self, frame, w, h, metrics):
        """Draw debug information in eye mode"""
        # Draw targeting reticle
        cv2.circle(frame, (w//2, h//2), 30, (0, 255, 255), 2)
        cv2.line(frame, (w//2-50, h//2), (w//2+50, h//2), (0, 255, 255), 2)
        cv2.line(frame, (w//2, h//2-50), (w//2, h//2+50), (0, 255, 255), 2)
        
        # Show alignment guidance
        cv2.putText(frame, "Align one eye with center marker",
                  (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Show iris offset if available
        if 'iris_offset_x' in metrics and 'iris_offset_y' in metrics:
            offset_x = metrics['iris_offset_x']
            offset_y = metrics['iris_offset_y']
            cv2.putText(frame, f"Offset: X:{offset_x:.2f}, Y:{offset_y:.2f}",
                      (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def _extract_eye_landmarks(self, face_landmarks, indices, frame_width, frame_height):
        """Extract and normalize landmark coordinates"""
        landmarks = []
        for idx in indices:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
            landmarks.append((x, y, landmark.z))
        return np.array(landmarks)
    
    def _calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio (EAR)"""
        if len(eye_landmarks) < 6:
            return 0.0
            
        vertical_1 = np.linalg.norm(eye_landmarks[1][:2] - eye_landmarks[5][:2])
        vertical_2 = np.linalg.norm(eye_landmarks[2][:2] - eye_landmarks[4][:2])
        horizontal = np.linalg.norm(eye_landmarks[0][:2] - eye_landmarks[3][:2])
        
        return (vertical_1 + vertical_2) / (2.0 * horizontal) if horizontal != 0 else 0.0
    
    def _log_eye_movements(self, metrics):
        """Log eye movement metrics"""
        movement_log = "Eye Tracking: "
        
        if 'avg_saccade_velocity' in metrics:
            movement_log += f"Saccade: {metrics['avg_saccade_velocity']:.2f}°/s | "
        
        if 'avg_vertical_saccade_velocity' in metrics:
            movement_log += f"V-Saccade: {metrics['avg_vertical_saccade_velocity']:.2f}°/s | "
        
        if 'fixation_stability' in metrics:
            movement_log += f"Fixation: {metrics['fixation_stability']:.4f} | "
        
        if 'avg_ear' in metrics:
            movement_log += f"EAR: {metrics['avg_ear']:.3f}"
        
        print(movement_log)
