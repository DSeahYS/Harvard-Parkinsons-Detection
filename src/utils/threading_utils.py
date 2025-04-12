import cv2
import threading
import time
import queue
import logging
import numpy as np
import os # Added import

# --- Project Imports ---
# Use absolute imports assuming 'src' is in PYTHONPATH or run via `python -m src.main`
try:
    from ..models.eye_tracker import EyeTracker
    from ..models.pd_detector import PDDetector, RiskLevel
    from ..genomic.bionemo_client import BioNeMoRiskAssessor
    from ..data.storage import StorageManager # Added import
    from ..llm.openrouter_client import OpenRouterClient
except ImportError:
    # Fallback for direct execution or different project structure
    # This might indicate running the script directly instead of as part of the package
    print("Warning: Could not perform relative imports in threading_utils. Assuming models are in sibling directories.")
    # Add alternative import paths if necessary, or rely on PYTHONPATH
    # Example:
    # script_dir = os.path.dirname(__file__)
    # models_dir = os.path.join(script_dir, '..', 'models')
    # sys.path.append(models_dir)
    # from eye_tracker import EyeTracker
    # ... etc.
    # For simplicity, we'll assume the package structure is correct for now.
    # If errors persist, the execution context needs review.
    pass # Let it fail later if imports truly don't work


# Configure logging (might be overridden by main)
# log_level = os.environ.get('LOG_LEVEL', 'INFO').upper() # Moved config to main.py
# logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Camera Streaming Thread ---
class RTSPCameraStream:
    """
    Dedicated thread for reading frames from a camera source (like webcam or RTSP).
    Uses a queue to pass frames to the processing thread.
    """
    def __init__(self, source=0, max_queue_size=10):
        self.source = source
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._thread = None
        self.is_running = False

    def start(self):
        logger.info(f"Attempting to open camera source: {self.source}")
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise IOError(f"Cannot open camera source: {self.source}")
            logger.info(f"Camera source {self.source} opened successfully.")
            self.is_running = True
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._update, name=f"CameraThread-{self.source}", daemon=True)
            self._thread.start()
            logger.info(f"Camera thread started for source: {self.source}.")
        except Exception as e:
            logger.error(f"Failed to start camera stream for source {self.source}: {e}", exc_info=True)
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.cap = None
        return self # Return self for chaining

    def _update(self):
        logger.info("Camera update loop running...")
        while not self._stop_event.is_set() and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                logger.warning(f"Camera source {self.source}: Failed to grab frame or stream ended.")
                # Optional: Attempt to reopen stream?
                time.sleep(0.5) # Wait before retrying or exiting
                # For now, just stop if we can't read.
                # self.stop() # Let the main loop handle stopping based on is_running
                break # Exit loop if frame read fails

            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                # If queue is full, discard oldest frame and add newest
                try:
                    self.frame_queue.get_nowait() # Discard oldest
                    self.frame_queue.put(frame) # Add newest
                    # logger.warning("Camera frame queue was full. Discarded oldest frame.") # Can be noisy
                except queue.Empty:
                    pass # Should not happen if full, but handle race condition

            # Add a small sleep to prevent busy-waiting if frame rate is low
            # Adjust based on expected camera FPS
            time.sleep(0.01) # Approx 100 FPS max read rate

        # Cleanup when loop exits
        if self.cap:
            self.cap.release()
        self.is_running = False
        logger.info(f"Camera stream {self.source} released.")

    def read(self):
        """Reads the latest frame from the queue (non-blocking)."""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None # No new frame available

    def stop(self):
        logger.info(f"Stopping camera thread for source: {self.source}...")
        self._stop_event.set()
        # Wait for thread to finish
        if self._thread is not None and self._thread.is_alive():
             self._thread.join(timeout=2) # Wait up to 2 seconds
             if self._thread.is_alive():
                  logger.warning(f"Camera thread {self.source} did not stop gracefully.")
        logger.info(f"Camera thread {self.source} stopped.")
        # Ensure cap is released even if thread join times out
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info(f"Camera capture {self.source} explicitly released after stop.")
        self.is_running = False


# --- Frame Processing Thread ---
class ProcessingThread(threading.Thread):
    """
    Thread for processing video frames using EyeTracker and PDDetector.
    Receives frames from a camera queue, puts results onto a result queue.
    """
    def __init__(self, eye_tracker: EyeTracker, pd_detector: PDDetector, frame_queue: queue.Queue, result_queue: queue.Queue, genomic_trigger_queue: queue.Queue, debug_mode=True):
        super().__init__(name="ProcessingThread", daemon=True)
        self.eye_tracker = eye_tracker
        self.pd_detector = pd_detector
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.genomic_trigger_queue = genomic_trigger_queue
        self._stop_event = threading.Event()
        self.debug_mode = debug_mode
        self.session_active = False
        # self.raw_log = [] # Deprecated: Replaced by streaming to JSON
        self.current_genetic_score = 1.0  # Default baseline genetic risk score
        self.storage = StorageManager() # Get storage instance
        self.current_session_id = None # Will be set when session starts
        self.show_mesh = True  # Default to showing mesh
        self.mesh_lock = threading.Lock()  # For thread-safe access

    def toggle_mesh(self, show_mesh):
        """Toggle whether to show the face mesh overlay on processed frames"""
        with self.mesh_lock:
            self.show_mesh = show_mesh
        logger.info(f"Mesh display {'enabled' if show_mesh else 'disabled'}")

    def run(self):
        logger.info("Processing thread running...")
        frame_count = 0
        last_genomic_trigger_time = 0
        GENOMIC_TRIGGER_INTERVAL = 5 # seconds

        while not self._stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.5) # Wait up to 0.5s for a frame
                if frame is None: # Check for sentinel value if used
                    continue

                timestamp = time.time()

                # --- Eye Tracking ---
                # Get current mesh display state
                with self.mesh_lock:
                    show_mesh = self.show_mesh

                processed_frame, eye_metrics, _ = self.eye_tracker.process_frame(frame, draw_mesh=show_mesh)

                if eye_metrics:
                    try:
                        # --- PD Risk Detection ---
                        risk_tuple = self.pd_detector.assess_risk(eye_metrics, self.current_genetic_score)
                        
                        # Convert to dict for safe handling
                        risk_data = {
                            'risk_level': risk_tuple[0],
                            'reason': risk_tuple[1],
                            'ocular_score': risk_tuple[2],
                            'factors': risk_tuple[3]
                        }

                        # --- Combine Results ---
                        combined_metrics = {
                            "timestamp": timestamp,
                            "eye_metrics": eye_metrics,
                            "pd_risk": risk_data
                        }

                        # Stream metrics to JSONL file if session is active
                        if self.session_active and self.current_session_id:
                            self.storage.save_metric_stream(self.current_session_id, combined_metrics)

                        # Put processed frame and metrics onto result queue
                        try:
                            self.result_queue.put((
                                "processed_frame",
                                (processed_frame, combined_metrics)
                            ), block=False)
                        except queue.Full:
                            logger.warning("Result queue is full. Dropping processed frame/metrics.")

                        # --- Trigger Genomic Analysis Periodically ---
                        if self.session_active and (timestamp - last_genomic_trigger_time > GENOMIC_TRIGGER_INTERVAL):
                            if not self.genomic_trigger_queue.full():
                                trigger_data = {
                                    'eye_risk_level': risk_data['risk_level'].value,
                                    'raw_metrics': eye_metrics
                                }
                                self.genomic_trigger_queue.put(trigger_data, block=False)
                                last_genomic_trigger_time = timestamp
                                logger.debug("Triggered genomic analysis.")
                            else:
                                logger.warning("Genomic trigger queue full. Skipping trigger.")

                    except Exception as e:
                        logger.error(f"Error processing frame: {e}", exc_info=True)
                        # Put error frame on queue for UI to handle
                        self.result_queue.put(("error_frame", (frame, str(e))), block=False)

                else:
                     # If no eye metrics, still put the original/processed frame for display
                     try:
                          self.result_queue.put(("processed_frame", (processed_frame, {})), block=False) # Send frame with empty metrics
                     except queue.Full:
                          logger.warning("Result queue is full. Dropping frame.")


                self.frame_queue.task_done() # Mark frame as processed

            except queue.Empty:
                # No frame received within timeout, simply continue loop
                continue
            except Exception as e:
                logger.error(f"Error in processing thread: {e}", exc_info=True)
                # Put error onto result queue for UI to handle
                try:
                    self.result_queue.put(("error", (self.name, str(e))), block=False)
                except queue.Full:
                     logger.error("Result queue full while trying to report processing error.")
                time.sleep(0.1) # Avoid tight loop on continuous errors

        logger.info("Processing thread stopped.")

    def start_session(self, session_id):
        self.current_session_id = session_id # Store the active session ID
        self.session_active = True
        # self.raw_log = [] # Deprecated
        logger.info(f"Processing thread session started for ID: {session_id}")

    def stop_session(self):
        self.session_active = False
        logger.info("Processing thread session stopped.")
        # Log might be retrieved by main thread after stopping
    # def get_raw_log(self): # Deprecated: Metrics are streamed directly to JSONL
    #     return self.raw_log

    def set_debug_mode(self, is_debug):
        self.debug_mode = is_debug
        logger.info(f"Processing thread debug mode set to {is_debug}")

    def stop(self):
        logger.info("Stopping processing thread...")
        self._stop_event.set()


# --- Genomic Analysis Thread ---
class GenomicAnalysisThread(threading.Thread):
    """
    Thread for running simulated genomic analysis using BioNeMoClient.
    Triggered by items in genomic_trigger_queue, puts results onto result_queue.
    """
    def __init__(self, bionemo_client: BioNeMoRiskAssessor, trigger_queue: queue.Queue, result_queue: queue.Queue): # Removed ethnicity from init
        super().__init__(name="GenomicAnalysisThread", daemon=True)
        self.bionemo_client = bionemo_client
        self.trigger_queue = trigger_queue
        self.result_queue = result_queue # Still needed for errors/status
        # self.ethnicity = ethnicity # Ethnicity will be passed with trigger data
        self._stop_event = threading.Event()
        self.storage = StorageManager() # Get storage instance
        self.current_session_id = None # Will be set when session starts
        self.current_ethnicity = None # Will be set when session starts
        self.session_active = False

    def run(self):
        logger.info("Genomic analysis thread running...")
        while not self._stop_event.is_set():
            try:
                # Wait for a trigger event (can be just a signal or contain data)
                trigger_data = self.trigger_queue.get(timeout=1.0)
                if trigger_data is None: continue # Sentinel

                if not self.session_active or not self.current_session_id:
                     logger.debug("Genomic thread received trigger but session not active or ID missing. Ignoring.")
                     self.trigger_queue.task_done()
                     continue

                # Extract necessary info (e.g., eye metrics if needed by actual BioNeMo)
                # For simulation, we might just need ethnicity stored during start_session
                logger.info(f"Genomic thread triggered for session {self.current_session_id}, Ethnicity: {self.current_ethnicity}")

                # --- Perform Genomic Analysis ---
                # In a real scenario, pass relevant data (e.g., eye_metrics from trigger_data)
                # For the simulation, it might only need ethnicity
                # Let's assume the bionemo_client.analyze needs eye_metrics and ethnicity
                # We need to get the latest eye_metrics. This design is tricky.
                # Option 1: Pass latest metrics in trigger_data (might be stale)
                # Option 2: Genomic thread reads latest from metrics.jsonl (adds complexity)
                # Option 3: Simplify simulation to not require eye_metrics directly here.
                # Let's go with Option 3 for now, assuming analyze can work with just ethnicity
                # or potentially placeholder metrics if absolutely needed by the API structure.

                # Placeholder eye metrics if needed by the analyze method signature
                placeholder_eye_metrics = {'saccade_velocity_deg_s': 400, 'fixation_stability_deg': 0.5}

                genomic_results = self.bionemo_client.analyze(
                    eye_metrics=placeholder_eye_metrics, # Use placeholder or get from trigger_data if available
                    ethnicity=self.current_ethnicity
                )

                if genomic_results:
                    # --- Save results directly to JSON ---
                    self.storage.save_genomic_results(self.current_session_id, genomic_results)
                    # Optionally, put a simple notification on result_queue if UI needs to know it's done
                    try:
                         self.result_queue.put(("status_update", (self.name, f"Genomic analysis saved for {self.current_session_id}")), block=False)
                    except queue.Full:
                         logger.warning("Result queue full when sending genomic status update.")
                else:
                     logger.error(f"Genomic analysis failed for session {self.current_session_id}")
                     # Report error back to UI via result queue
                     try:
                          self.result_queue.put(("error", (self.name, f"Genomic analysis failed for session {self.current_session_id}")), block=False)
                     except queue.Full:
                          logger.error("Result queue full while trying to report genomic analysis error.")


                self.trigger_queue.task_done()

            except queue.Empty:
                # Timeout waiting for trigger, normal operation
                continue
            except Exception as e:
                logger.error(f"Error in genomic analysis thread: {e}", exc_info=True)
                try:
                    self.result_queue.put(("error", (self.name, str(e))), block=False)
                except queue.Full:
                     logger.error("Result queue full while trying to report genomic analysis error.")
                time.sleep(1) # Pause after error

        logger.info("Genomic analysis thread stopped.")

    def start_session(self, session_id, ethnicity):
        self.current_session_id = session_id
        self.current_ethnicity = ethnicity
        self.session_active = True
        # Clear the trigger queue at the start of a session
        while not self.trigger_queue.empty():
            try: self.trigger_queue.get_nowait()
            except queue.Empty: break
        logger.info(f"Genomic thread session started for ID: {session_id}, Ethnicity: {ethnicity}")


    def stop_session(self):
        self.session_active = False
        logger.info("Genomic thread session stopped.")

    # def update_ethnicity(self, ethnicity): # Deprecated: Ethnicity passed via start_session
    #      self.ethnicity = ethnicity
    #      logger.info(f"Genomic thread ethnicity updated to: {ethnicity}")

    def stop(self):
        logger.info("Stopping genomic analysis thread...")
        self._stop_event.set()
        # Optionally put a sentinel value to unblock the queue.get()
        # try:
        #     self.trigger_queue.put(None, block=False)
        # except queue.Full:
        #     pass


# --- LLM Analysis Thread ---
class LLMAnalysisThread(threading.Thread):
    """
    Thread for handling potentially long-running LLM API calls.
    Takes requests from llm_queue, puts results onto result_queue.
    """
    def __init__(self, llm_client: OpenRouterClient, request_queue: queue.Queue, result_queue: queue.Queue):
        super().__init__(name="LLMAnalysisThread", daemon=True)
        self.llm_client = llm_client
        self.request_queue = request_queue
        self.result_queue = result_queue
        self._stop_event = threading.Event()

    def run(self):
        logger.info("LLM analysis thread running...")
        while not self._stop_event.is_set():
            try:
                # Wait for an analysis request
                request_data = self.request_queue.get(timeout=1.0)
                if request_data is None: # Sentinel check
                    continue

                # Handle both tuple and dictionary formats
                if isinstance(request_data, tuple) and len(request_data) == 2:
                    # If it's a tuple, unpack it directly
                    prompt_type, data_payload = request_data
                elif isinstance(request_data, dict):
                    # If it's a dictionary, get the values using keys
                    prompt_type = request_data.get("prompt_type")
                    data_payload = request_data.get("data")
                else:
                    logger.warning(f"Invalid LLM request format: {type(request_data)}")
                    self.request_queue.task_done()
                    continue

                if not prompt_type or data_payload is None:
                    logger.warning("Invalid LLM request received (missing type or data).")
                    self.request_queue.task_done()
                    continue

                logger.info(f"LLM thread received request for: {prompt_type}")

                # Call the LLM client
                analysis_result = self.llm_client.generate_summary(prompt_type, data_payload)

                # Put the result or error message onto the main result queue
                try:
                    # Tag the result with the type of analysis performed
                    self.result_queue.put(("llm_result", (prompt_type, analysis_result)), block=False)
                    logger.info(f"LLM analysis for '{prompt_type}' complete, results queued.")
                except queue.Full:
                    logger.warning(f"Result queue full. Discarding LLM analysis result for '{prompt_type}'.")

                self.request_queue.task_done()

            except queue.Empty:
                # Timeout waiting for request, normal operation
                continue
            except Exception as e:
                logger.error(f"Error in LLM analysis thread: {e}", exc_info=True)
                try:
                    # Report error back to UI
                    self.result_queue.put(("error", (self.name, f"LLM analysis failed: {e}")), block=False)
                except queue.Full:
                     logger.error("Result queue full while trying to report LLM analysis error.")
                time.sleep(1) # Pause after error

        logger.info("LLM analysis thread stopped.")

    def stop(self):
        logger.info("Stopping LLM analysis thread...")
        self._stop_event.set()
        # Optionally put a sentinel value to unblock the queue.get()
        # try:
        #     self.request_queue.put(None, block=False)
        # except queue.Full:
        #     pass
