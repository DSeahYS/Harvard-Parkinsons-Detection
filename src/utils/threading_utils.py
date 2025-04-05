import cv2
import threading
import time
import queue
import logging

from ..models.eye_tracker import EyeTracker
from ..models.pd_detector import PDDetector
# Import other necessary components if needed, e.g., BioNeMoClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

class RTSPCameraStream:
    """
    Dedicated thread for capturing frames from a camera source (like cv2.VideoCapture)
    to prevent blocking the main processing or GUI thread. Uses a lock for safe
    frame access.
    """
    def __init__(self, src=0, name="CameraThread"):
        """
        Initializes the camera stream.

        Args:
            src (int or str): The source index or path for cv2.VideoCapture.
            name (str): Name for the thread.
        """
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            logging.error(f"Failed to open camera source: {src}")
            raise ValueError(f"Could not open video source: {src}")

        self.stopped = False
        self.frame = None
        self.frame_lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, name=name, daemon=True)
        logging.info(f"RTSPCameraStream initialized for source: {src}")

    def start(self):
        """Starts the frame reading thread."""
        self.stopped = False
        self.thread.start()
        logging.info("Camera thread started.")
        return self

    def update(self):
        """Continuously reads frames from the stream."""
        logging.info("Camera update loop running...")
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                logging.warning("Camera stream returned False (end of stream or error). Stopping.")
                self.stop()
                break

            with self.frame_lock:
                self.frame = frame
        # Release stream resource when thread stops
        self.stream.release()
        logging.info("Camera stream released.")

    def read(self):
        """
        Returns the latest frame captured by the thread.

        Returns:
            np.ndarray or None: The latest frame (copied for safety), or None if no frame
                                is available or the stream hasn't started/is stopped.
        """
        with self.frame_lock:
            # Return a copy to prevent race conditions if the frame is modified elsewhere
            frame_copy = self.frame.copy() if self.frame is not None else None
        return frame_copy

    def stop(self):
        """Signals the thread to stop."""
        logging.info("Stopping camera thread...")
        self.stopped = True
        # Wait briefly for the thread to potentially finish its current loop iteration
        if self.thread.is_alive():
             self.thread.join(timeout=1.0) # Wait max 1 sec for thread to stop gracefully
             if self.thread.is_alive():
                 logging.warning("Camera thread did not stop gracefully within timeout.")


class ProcessingThread(threading.Thread):
    """
    Dedicated thread for running the eye tracking and PD detection pipeline.
    Reads frames from a camera stream and puts results into a queue.
    """
    def __init__(self, camera_stream, result_queue, eye_tracker: EyeTracker, pd_detector: PDDetector, name="ProcessingThread"):
        """
        Initializes the processing thread.

        Args:
            camera_stream (RTSPCameraStream): The camera stream instance to read frames from.
            result_queue (queue.Queue): Queue to put the processing results into
                                        (e.g., (processed_frame, metrics, risk_results)).
            eye_tracker (EyeTracker): Instance of the eye tracker model.
            pd_detector (PDDetector): Instance of the Parkinson's detector model.
            name (str): Name for the thread.
        """
        super().__init__(name=name, daemon=True)
        self.camera_stream = camera_stream
        self.result_queue = result_queue
        self.eye_tracker = eye_tracker
        self.pd_detector = pd_detector
        self.running = False
        self._stop_event = threading.Event()
        logging.info("ProcessingThread initialized.")

    def run(self):
        """The main loop for the processing thread."""
        logging.info("Processing thread running...")
        self.running = True
        while not self._stop_event.is_set():
            frame = self.camera_stream.read()
            if frame is None:
                # If camera stopped or hasn't provided a frame yet, wait briefly
                time.sleep(0.01)
                continue

            try:
                # Process frame using EyeTracker
                # Unpack all 4 return values from process_frame
                output_frame, original_frame_rgb, metrics, face_landmarks = self.eye_tracker.process_frame(frame)

                risk_results = None
                if metrics:
                    # Get risk assessment from PDDetector
                    # Assuming ethnicity needs to be passed; get it from patient profile? Placeholder for now.
                    current_ethnicity = "default" # TODO: Get actual ethnicity
                    risk_level, factors = self.pd_detector.predict(metrics, ethnicity=current_ethnicity)
                    risk_results = (risk_level, factors)

                # Put results into the queue for the main (GUI) thread
                # Include raw metrics and face landmarks if needed by other components (e.g., BioNeMo, visualization)
                result_package = {
                    'processed_frame': output_frame, # Use the frame potentially with drawings
                    'raw_metrics': metrics,
                    'risk_results': risk_results,
                    'face_landmarks': face_landmarks # Pass raw landmarks if needed downstream
                }
                self.result_queue.put(result_package)

            except Exception as e:
                logging.error(f"Error during frame processing: {e}", exc_info=True)
                # Optionally put an error message in the queue or handle differently
                time.sleep(0.1) # Avoid spamming errors

        self.running = False
        try:
            self.eye_tracker.close() # Release MediaPipe resources
        except Exception as e:
            logging.warning(f"Error closing eye tracker: {e}")
        logging.info("Processing thread stopped.")
        
    def reset(self):
        """Resets the processor with fresh resources."""
        self.stop()
        self.eye_tracker.reset()
        self._stop_event.clear()

    def stop(self):
        """Signals the processing thread to stop."""
        logging.info("Stopping processing thread...")
        self._stop_event.set()


class GenomicAnalysisThread(threading.Thread):
    """
    Dedicated thread for running potentially long-running genomic analysis.
    Receives necessary data via one queue and puts results into another.
    """
    def __init__(self, input_queue, output_queue, bionemo_client, name="GenomicThread"):
        """
        Initializes the genomic analysis thread.

        Args:
            input_queue (queue.Queue): Queue to receive data needed for analysis
                                       (e.g., {'eye_risk_level': float, 'ethnicity': str}).
            output_queue (queue.Queue): Queue to put the genomic analysis results into.
            bionemo_client: Instance of the BioNeMo client (or similar).
            name (str): Name for the thread.
        """
        super().__init__(name=name, daemon=True)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.bionemo_client = bionemo_client
        self._stop_event = threading.Event()
        logging.info("GenomicAnalysisThread initialized.")

    def run(self):
        """The main loop for the genomic analysis thread."""
        logging.info("Genomic analysis thread running...")
        while not self._stop_event.is_set():
            try:
                # Wait for input data (blocking)
                analysis_input = self.input_queue.get(timeout=1.0) # Timeout to allow checking stop event

                if analysis_input is None: # Sentinel value to stop
                    logging.info("Received None, stopping genomic analysis thread.")
                    break

                eye_risk = analysis_input.get('eye_risk_level')
                ethnicity = analysis_input.get('ethnicity')

                if eye_risk is not None and ethnicity is not None:
                    logging.info(f"Performing genomic analysis for ethnicity: {ethnicity}, eye risk: {eye_risk:.3f}")
                    # Perform the potentially long-running analysis
                    genomic_results = self.bionemo_client.analyze_genomics(
                        eye_risk_level=eye_risk,
                        ethnicity=ethnicity
                    )
                    # Put results in the output queue
                    self.output_queue.put(genomic_results)
                else:
                    logging.warning("Genomic analysis input missing required data (eye_risk_level or ethnicity).")

                self.input_queue.task_done() # Signal that the item is processed

            except queue.Empty:
                # Timeout occurred, just loop again to check stop event
                continue
            except Exception as e:
                logging.error(f"Error during genomic analysis: {e}", exc_info=True)
                # Optionally put an error message in the output queue
                self.output_queue.put({'error': str(e)})
                # Ensure task_done is called even if there's an error processing the item
                if 'analysis_input' in locals() and analysis_input is not None:
                     self.input_queue.task_done()


        logging.info("Genomic analysis thread stopped.")

    def stop(self):
        """Signals the genomic analysis thread to stop."""
        logging.info("Stopping genomic analysis thread...")
        self._stop_event.set()
        # Put a sentinel value in the queue to unblock the get() call if it's waiting
        self.input_queue.put(None)

# Example of how to use (usually instantiated in the main application/dashboard)
if __name__ == '__main__':
    print("Testing threading utilities...")

    # Dummy components for testing
    class MockEyeTracker:
        def process_frame(self, frame):
            print("MockEyeTracker processing frame...")
            time.sleep(0.05) # Simulate work
            return frame, {'saccade_velocity': 100, 'fixation_stability': 0.5, 'blink_rate': 15}, None
        def close(self): print("MockEyeTracker closed.")

    class MockPDDetector:
        def predict(self, metrics, ethnicity):
            print(f"MockPDDetector predicting for {ethnicity}...")
            time.sleep(0.02) # Simulate work
            return 0.45, {'factor1': 0.2, 'factor2': 0.25}

    class MockBioNeMo:
         def analyze_genomics(self, eye_risk_level, ethnicity):
             print(f"MockBioNeMo analyzing for {ethnicity}, risk {eye_risk_level:.2f}...")
             time.sleep(0.5) # Simulate longer work
             return {'simulated_risk': eye_risk_level * 0.5, 'ethnicity': ethnicity}


    results_q = queue.Queue()
    genomic_in_q = queue.Queue()
    genomic_out_q = queue.Queue()

    try:
        cam = RTSPCameraStream(src=0).start() # Use default webcam
        time.sleep(2) # Allow camera to warm up

        proc_thread = ProcessingThread(cam, results_q, MockEyeTracker(), MockPDDetector())
        gen_thread = GenomicAnalysisThread(genomic_in_q, genomic_out_q, MockBioNeMo())

        proc_thread.start()
        gen_thread.start()

        # Simulate main loop getting results
        for i in range(10):
            print(f"\nMain loop iteration {i+1}")
            try:
                result = results_q.get(timeout=1.0)
                print(f"  Got processing result: Risk={result['risk_results'][0] if result['risk_results'] else 'N/A'}")

                # Occasionally trigger genomic analysis
                if i % 4 == 0 and result.get('risk_results'):
                    print("  Triggering genomic analysis...")
                    genomic_in_q.put({'eye_risk_level': result['risk_results'][0], 'ethnicity': 'test_ethnicity'})

            except queue.Empty:
                print("  Processing queue empty.")

            # Check for genomic results
            try:
                gen_result = genomic_out_q.get_nowait()
                print(f"  Got genomic result: {gen_result}")
            except queue.Empty:
                pass # No genomic result yet

            time.sleep(0.1)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("\nStopping threads...")
        if 'cam' in locals(): cam.stop()
        if 'proc_thread' in locals(): proc_thread.stop()
        if 'gen_thread' in locals(): gen_thread.stop()

        # Wait for threads to finish
        if 'proc_thread' in locals() and proc_thread.is_alive(): proc_thread.join(timeout=2)
        if 'gen_thread' in locals() and gen_thread.is_alive(): gen_thread.join(timeout=2)

        print("Threads stopped.")
