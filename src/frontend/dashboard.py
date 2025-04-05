import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, Listbox, Scrollbar, Frame, Label, Button, Canvas, Text
from PIL import Image, ImageTk
import cv2
import queue
import threading
import logging
import time
import numpy as np
import os
import json # Added for LLM result parsing example

# --- Project Imports ---
# Use absolute imports from the project root assuming 'src' is in PYTHONPATH
# or adjust relative paths carefully based on execution context.
# If running dashboard.py directly, relative paths might need adjustment or
# run using `python -m src.frontend.dashboard` from the GenomeGuard directory.
try:
    from ..models.eye_tracker import EyeTracker
    from ..models.pd_detector import PDDetector
    from ..genomic.bionemo_client import BioNeMoClient
    from ..llm.openrouter_client import OpenRouterClient
    from ..data.storage import StorageManager
    from ..utils.threading_utils import RTSPCameraStream, ProcessingThread, GenomicAnalysisThread
    from ..utils import visualization # Import the visualization module
except ImportError:
    # Fallback for running script directly (adjust as needed)
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..')) # Add GenomeGuard root to path
    from src.models.eye_tracker import EyeTracker
    from src.models.pd_detector import PDDetector
    from src.genomic.bionemo_client import BioNeMoClient
    from src.llm.openrouter_client import OpenRouterClient
    from src.data.storage import StorageManager
    from src.utils.threading_utils import RTSPCameraStream, ProcessingThread, GenomicAnalysisThread
    from src.utils import visualization


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

class Dashboard:
    """
    Main Tkinter-based GUI for GenomeGuard application.
    Manages data processing threads and displays results.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("GenomeGuard - Parkinson's Risk Assessment")
        # Set a default size
        self.root.geometry("1200x800")

        # --- Initialize Core Components ---
        logging.info("Initializing core components...")
        self.storage = StorageManager() # Singleton instance
        self.eye_tracker = EyeTracker(refine_landmarks=True) # Use refined landmarks
        self.pd_detector = PDDetector()
        self.bionemo_client = BioNeMoClient() # Placeholder/real client
        self.openrouter_client = OpenRouterClient() # Placeholder/real client
        logging.info("Core components initialized.")

        # --- Initialize Threading Components ---
        logging.info("Initializing threading components...")
        self.processing_result_queue = queue.Queue(maxsize=10) # Queue for results from ProcessingThread
        self.genomic_input_queue = queue.Queue()   # Queue to send data TO GenomicAnalysisThread
        self.genomic_output_queue = queue.Queue() # Queue for results FROM GenomicAnalysisThread

        self.camera_stream = None # Will be initialized in start_processing
        self.processing_thread = None
        self.genomic_thread = None
        logging.info("Threading components initialized.")

        # --- State Variables ---
        self.is_processing = False
        self.current_patient_id = None
        self.current_patient_info = {}
        self.current_session_id = None
        self.session_data_log = [] # Store detailed metrics for saving later
        self.last_risk_level = None # Store last known risk level
        self.last_eye_metrics_raw = None # Store last known raw metrics
        self.last_eye_metrics_summary = None # Store last known summary
        self.last_genomic_result = None # Store last known genomic result
        self.selected_history_session_id = None # Track selected past session ID
        self.loaded_history_session_data = None # Store loaded data for selected past session

        # --- Setup UI ---
        self._setup_ui()

        # --- Handle Window Closing ---
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_ui(self):
        """Creates the main UI layout."""
        logging.info("Setting up UI...")
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Left Panel (Controls, Patient Info) ---
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False) # Prevent resizing based on content

        # Control Buttons
        control_frame = ttk.LabelFrame(left_panel, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))

        self.start_button = ttk.Button(control_frame, text="Start Processing", command=self.start_processing)
        self.start_button.pack(fill=tk.X, pady=5)
        self.stop_button = ttk.Button(control_frame, text="Stop Processing", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=5)

# Removed duplicate block


        # Patient Selection / Management
        patient_frame = ttk.LabelFrame(left_panel, text="Patient Management", padding="10")
        patient_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(patient_frame, text="Select Patient:").pack(anchor=tk.W)
        self.patient_listbox = Listbox(patient_frame, height=6, exportselection=False)
        self.patient_listbox.pack(fill=tk.X, expand=True, pady=(0, 5))
        self.patient_listbox.bind('<<ListboxSelect>>', self._on_patient_select)
        self._load_patients_into_listbox() # Populate listbox

        patient_button_frame = ttk.Frame(patient_frame)
        patient_button_frame.pack(fill=tk.X)
        ttk.Button(patient_button_frame, text="New", command=self._new_patient).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,2))
        # Button removed, history is now loaded automatically in the dedicated tab
        # Add Edit/Delete later if needed

        # Current Patient Info Display
        info_frame = ttk.LabelFrame(left_panel, text="Current Patient", padding="10")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        self.patient_name_label = ttk.Label(info_frame, text="Name: N/A")
        self.patient_name_label.pack(anchor=tk.W)
        self.patient_ethnicity_label = ttk.Label(info_frame, text="Ethnicity: N/A")
        self.patient_ethnicity_label.pack(anchor=tk.W)
        # Add more labels (DOB, History) as needed

        # --- Right Panel (Video Feed, Analysis Tabs) ---
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Video Feed Area
        video_frame = ttk.LabelFrame(right_panel, text="Live Feed", padding="5")
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10)) # Allow expansion

        # Use Canvas for potential drawing over video later
        self.video_canvas = Canvas(video_frame, bg="black")
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        # Placeholder text until feed starts
        self.video_placeholder = self.video_canvas.create_text(320, 240, text="Webcam Feed Disconnected", fill="white", font=("Arial", 16))
        self.video_label_ref = None # To hold the PhotoImage reference

        # Analysis Tabs
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Eye Metrics
        eye_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(eye_tab, text="Eye Analysis")
        # Risk Meter (Canvas)
        self.risk_meter_canvas = Canvas(eye_tab, width=200, height=50, bg='lightgrey') # Made smaller
        self.risk_meter_canvas.pack(pady=10)
        self.risk_meter_ref = None # Reference for risk meter image
        # Metrics Display (Labels)
        metrics_frame = ttk.Frame(eye_tab)
        metrics_frame.pack(fill=tk.X, pady=5)
        self.metrics_labels = {}
        metric_keys = ['saccade_velocity', 'fixation_stability', 'blink_rate', 'vertical_saccade_velocity', 'eye_aspect_ratio_left'] # Example keys
        for key in metric_keys:
            frame = ttk.Frame(metrics_frame)
            frame.pack(fill=tk.X)
            ttk.Label(frame, text=f"{key.replace('_', ' ').title()}:").pack(side=tk.LEFT, padx=5)
            self.metrics_labels[key] = ttk.Label(frame, text="N/A", width=15, anchor=tk.W)
            self.metrics_labels[key].pack(side=tk.LEFT)
        # Add contributing factors display
        self.factors_label = ttk.Label(eye_tab, text="Contributing Factors: N/A", wraplength=300, justify=tk.LEFT)
        self.factors_label.pack(pady=5, anchor=tk.W)


        # Tab 2: Genomic Analysis
        genomic_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(genomic_tab, text="Genomic Simulation")
        self.genomic_trigger_button = ttk.Button(genomic_tab, text="Run Genomic Simulation", command=self._trigger_genomic_analysis, state=tk.DISABLED)
        self.genomic_trigger_button.pack(pady=10)
        self.genomic_results_text = Text(genomic_tab, height=10, width=60, wrap=tk.WORD, state=tk.DISABLED)
        self.genomic_results_text.pack(fill=tk.BOTH, expand=True)

        # Tab 3: LLM Analysis
        llm_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(llm_tab, text="LLM Summary")
        self.llm_trigger_button = ttk.Button(llm_tab, text="Get LLM Analysis", command=self._trigger_llm_analysis, state=tk.DISABLED)
        self.llm_trigger_button.pack(pady=10)
        self.llm_results_text = Text(llm_tab, height=10, width=60, wrap=tk.WORD, state=tk.DISABLED)
        self.llm_results_text.pack(fill=tk.BOTH, expand=True)

        # Tab 4: Patient History
        history_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(history_tab, text="Patient History")

        history_list_frame = ttk.Frame(history_tab)
        history_list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        ttk.Label(history_list_frame, text="Past Sessions:").pack(anchor=tk.W)
        self.history_listbox = Listbox(history_list_frame, height=15, exportselection=False)
        history_scrollbar = Scrollbar(history_list_frame, orient=tk.VERTICAL, command=self.history_listbox.yview)
        self.history_listbox.config(yscrollcommand=history_scrollbar.set)
        self.history_listbox.pack(side=tk.LEFT, fill=tk.Y)
        history_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_listbox.bind('<<ListboxSelect>>', self._on_session_history_select)

        history_detail_frame = ttk.Frame(history_tab)
        history_detail_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(history_detail_frame, text="Session Details:").pack(anchor=tk.W)
        self.history_detail_text = Text(history_detail_frame, height=15, width=60, wrap=tk.WORD, state=tk.DISABLED)
        detail_scrollbar = Scrollbar(history_detail_frame, orient=tk.VERTICAL, command=self.history_detail_text.yview)
        self.history_detail_text.config(yscrollcommand=detail_scrollbar.set)
        self.history_detail_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        detail_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        # Add button to trigger LLM for selected history item
        self.llm_history_button = ttk.Button(history_detail_frame, text="Analyze Selected Session (LLM)", command=lambda: self._trigger_llm_analysis(use_history=True), state=tk.DISABLED)
        self.llm_history_button.pack(pady=5)

        logging.info("UI setup complete.")

    # --- Patient Management Methods ---
    def _load_patients_into_listbox(self):
        """Loads patient names and IDs from storage into the listbox."""
        self.patient_listbox.delete(0, tk.END)
        patients = self.storage.list_patients()
        for patient in patients:
            self.patient_listbox.insert(tk.END, f"{patient['name']} (ID: {patient['id']})")
            # Store patient ID along with the display text if needed, e.g., using a dictionary
            # self.patient_map[f"{patient['name']} (ID: {patient['id']})"] = patient['id']

    def _on_patient_select(self, event=None):
        """Handles selection of a patient from the listbox."""
        selected_indices = self.patient_listbox.curselection()
        if not selected_indices:
            return
        selected_text = self.patient_listbox.get(selected_indices[0])
        # Extract ID (assuming format "Name (ID: id)")
        try:
            self.current_patient_id = int(selected_text.split("(ID: ")[1].replace(")", ""))
            self.current_patient_info = self.storage.get_patient(self.current_patient_id)
            if self.current_patient_info:
                self.patient_name_label.config(text=f"Name: {self.current_patient_info.get('name', 'N/A')}")
                self.patient_ethnicity_label.config(text=f"Ethnicity: {self.current_patient_info.get('ethnicity', 'N/A')}")
                logging.info(f"Selected patient ID: {self.current_patient_id}")
                # Enable analysis buttons if a patient is selected and processing is active
                self.genomic_trigger_button.config(state=tk.NORMAL if self.is_processing else tk.DISABLED)
                self.llm_trigger_button.config(state=tk.NORMAL if self.is_processing else tk.DISABLED) # LLM for live data
                # Load and display patient's session history
                self._display_patient_sessions()

            else:
                self._clear_patient_selection()
        except (IndexError, ValueError) as e:
            logging.error(f"Error parsing patient ID from listbox: {selected_text} - {e}")
            self._clear_patient_selection()

    def _clear_patient_selection(self):
        """Resets current patient info."""
        self.current_patient_id = None
        self.current_patient_info = {}
        self.patient_name_label.config(text="Name: N/A")
        self.patient_ethnicity_label.config(text="Ethnicity: N/A")
        self.genomic_trigger_button.config(state=tk.DISABLED)
        self.llm_trigger_button.config(state=tk.DISABLED)
        self.history_listbox.delete(0, tk.END) # Clear history list
        self.history_detail_text.config(state=tk.NORMAL)
        self.history_detail_text.delete('1.0', tk.END)
        self.history_detail_text.config(state=tk.DISABLED)
        self.llm_history_button.config(state=tk.DISABLED)
        self.selected_history_session_id = None
        self.loaded_history_session_data = None
        logging.info("Cleared patient selection and history.")


    def _new_patient(self):
        """Opens dialog to add a new patient."""
        name = simpledialog.askstring("New Patient", "Enter Patient Name:")
        if not name: return
        ethnicity = simpledialog.askstring("New Patient", "Enter Ethnicity (e.g., chinese, malay, indian):")
        # Add more fields (DOB, medical history) as needed
        new_id = self.storage.add_patient(name, ethnicity=ethnicity)
        if new_id:
            messagebox.showinfo("Success", f"Patient '{name}' added with ID {new_id}.")
            self._load_patients_into_listbox()
            # Optionally auto-select the new patient
        else:
            messagebox.showerror("Error", "Failed to add patient.")

    def _display_patient_sessions(self):
        """Fetches and displays past sessions for the current patient."""
        self.history_listbox.delete(0, tk.END)
        self.history_detail_text.config(state=tk.NORMAL)
        self.history_detail_text.delete('1.0', tk.END)
        self.history_detail_text.insert(tk.END, "Select a session from the list.")
        self.history_detail_text.config(state=tk.DISABLED)
        self.llm_history_button.config(state=tk.DISABLED)
        self.selected_history_session_id = None
        self.loaded_history_session_data = None


        if not self.current_patient_id:
            return

        sessions = self.storage.get_patient_sessions(self.current_patient_id)
        if not sessions:
            self.history_listbox.insert(tk.END, "No past sessions found.")
            return

        for session in sessions:
            start_time = session.get('start_time', 'N/A')
            session_id = session.get('id')
            risk = session.get('avg_risk_level')
            risk_str = f"{risk:.3f}" if risk is not None else "N/A"
            display_text = f"ID: {session_id} | Start: {start_time} | Avg Risk: {risk_str}"
            self.history_listbox.insert(tk.END, display_text)
            # Could store session ID mapping if needed, similar to patient list

    def _on_session_history_select(self, event=None):
        """Handles selection of a session from the history listbox."""
        selected_indices = self.history_listbox.curselection()
        if not selected_indices:
            return
        selected_text = self.history_listbox.get(selected_indices[0])

        # Extract session ID
        try:
            session_id_str = selected_text.split("ID: ")[1].split(" |")[0]
            self.selected_history_session_id = int(session_id_str)
            logging.info(f"Selected history session ID: {self.selected_history_session_id}")

            # Load session details from JSON
            self.loaded_history_session_data = self.storage.get_session_details(self.selected_history_session_id)

            self.history_detail_text.config(state=tk.NORMAL)
            self.history_detail_text.delete('1.0', tk.END)

            if self.loaded_history_session_data:
                # Display formatted JSON or summary
                # Ensure loaded_history_session_data is treated as a list of log entries
                if isinstance(self.loaded_history_session_data, list):
                     display_str = f"Session Log ({len(self.loaded_history_session_data)} entries):\n"
                     # Display first few entries as example, or summary stats
                     for i, entry in enumerate(self.loaded_history_session_data[:5]): # Show first 5
                         ts = time.strftime('%H:%M:%S', time.localtime(entry.get('timestamp', 0)))
                         risk = entry.get('risk_results', [None])[0]
                         risk_str = f"{risk:.3f}" if risk is not None else "N/A"
                         display_str += f"- {ts}: Risk={risk_str}\n"
                     if len(self.loaded_history_session_data) > 5:
                         display_str += f"... ({len(self.loaded_history_session_data) - 5} more entries)"
                else:
                    # Fallback if data is not a list (e.g., older format?)
                    display_str = json.dumps(self.loaded_history_session_data, indent=2, cls=self.storage.NumpyEncoder)

                self.history_detail_text.insert(tk.END, display_str)
                self.llm_history_button.config(state=tk.NORMAL) # Enable LLM analysis for this session
            else:
                self.history_detail_text.insert(tk.END, f"Could not load details for session {self.selected_history_session_id}.")
                self.llm_history_button.config(state=tk.DISABLED)

            self.history_detail_text.config(state=tk.DISABLED)

        except (IndexError, ValueError, TypeError) as e:
            logging.error(f"Error parsing session ID or loading details: {selected_text} - {e}", exc_info=True)
            self.history_detail_text.config(state=tk.NORMAL)
            self.history_detail_text.delete('1.0', tk.END)
            self.history_detail_text.insert(tk.END, "Error loading session details.")
            self.history_detail_text.config(state=tk.DISABLED)
            self.llm_history_button.config(state=tk.DISABLED)
            self.selected_history_session_id = None
            self.loaded_history_session_data = None

    # --- Mode Control Callbacks ---
    def _toggle_debug_mode(self):
        """Callback for the debug mode checkbutton."""
        is_debug = self.debug_mode_var.get()
        self.eye_tracker.set_debug_mode(is_debug)
        logging.info(f"Debug mode set to: {is_debug}")

    def _set_eye_processing_mode(self):
        """Callback for the eye processing mode radio buttons."""
        mode = self.eye_mode_var.get()
        self.eye_tracker.set_eye_processing_mode(mode)
        logging.info(f"Eye processing mode set to: {mode}")
        # Optionally clear metrics display when mode changes?
        for label in self.metrics_labels.values():
            label.config(text="N/A")
        self.factors_label.config(text="Contributing Factors: N/A")
        if self.risk_meter_canvas.winfo_exists():
             self.risk_meter_canvas.delete("all")
             self.risk_meter_canvas.create_text(100, 25, text="Risk: N/A", fill="black")
    # --- Processing Control ---
    def start_processing(self):
        """Starts the camera capture and processing threads."""
        if self.is_processing:
            logging.warning("Processing already running.")
            return

        logging.info("Starting processing...")
        try:
            # Initialize camera stream here to ensure it's fresh
            self.camera_stream = RTSPCameraStream(src=0).start()
            time.sleep(1.0) # Give camera time to initialize

            # Start a new session in the database
            self.current_session_id = self.storage.start_session(self.current_patient_id)
            self.session_data_log = [] # Reset log for new session
            self.last_risk_level = None # Reset state
            self.last_eye_metrics_raw = None
            self.last_eye_metrics_summary = None
            self.last_genomic_result = None


            # Initialize and start processing thread
            self.processing_thread = ProcessingThread(
                self.camera_stream,
                self.processing_result_queue,
                self.eye_tracker,
                self.pd_detector
            )
            self.processing_thread.start()

            # Initialize and start genomic thread (it will wait for input)
            self.genomic_thread = GenomicAnalysisThread(
                self.genomic_input_queue,
                self.genomic_output_queue,
                self.bionemo_client
            )
            self.genomic_thread.start()


            self.is_processing = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            if self.current_patient_id: # Enable analysis buttons only if patient selected
                 self.genomic_trigger_button.config(state=tk.NORMAL)
                 self.llm_trigger_button.config(state=tk.NORMAL)

            # Start checking the queue for results
            self._check_queues()
            logging.info("Processing started.")

        except Exception as e:
            logging.error(f"Error starting processing: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to start processing:\n{e}")
            # Clean up if partial start occurred
            if self.camera_stream: self.camera_stream.stop()
            if self.processing_thread and self.processing_thread.is_alive(): self.processing_thread.stop()
            if self.genomic_thread and self.genomic_thread.is_alive(): self.genomic_thread.stop()
            self.is_processing = False


    def stop_processing(self):
        """Stops the processing threads and resets for clean restart."""
        if not self.is_processing:
            logging.warning("Processing not running.")
            return

        logging.info("Stopping processing...")
        self.is_processing = False # Signal queue checker to stop requesting frames

        # Clear any pending frames in queues
        self._clear_queues()

        # Stop threads gracefully
        if self.genomic_thread and self.genomic_thread.is_alive():
            self.genomic_thread.stop()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.reset() # Reset with fresh resources

        # Reset camera stream if exists
        if hasattr(self, 'camera_stream'):
            self.camera_stream.stop()
        if self.camera_stream:
            self.camera_stream.stop()

        # Wait for threads to finish (optional, but good practice)
        # Add timeouts
        if self.processing_thread and self.processing_thread.is_alive(): self.processing_thread.join(timeout=2)
        if self.genomic_thread and self.genomic_thread.is_alive(): self.genomic_thread.join(timeout=2)


        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.genomic_trigger_button.config(state=tk.DISABLED)
        self.llm_trigger_button.config(state=tk.DISABLED)

        # End the session in the database and save detailed log
        if self.current_session_id:
            avg_risk = np.mean([item['risk_results'][0] for item in self.session_data_log if item.get('risk_results')]) if self.session_data_log else None
            json_filename = self.storage.save_session_details(self.current_session_id, self.session_data_log)
            self.storage.end_session(self.current_session_id, avg_risk_level=avg_risk, json_log_filename=json_filename)
            logging.info(f"Session {self.current_session_id} ended and saved.")
            self.current_session_id = None

    def _clear_queues(self):
        """Clears all processing queues to prepare for restart."""
        try:
            while not self.processing_result_queue.empty():
                self.processing_result_queue.get_nowait()
            while not self.genomic_input_queue.empty():
                self.genomic_input_queue.get_nowait()
            while not self.genomic_output_queue.empty():
                self.genomic_output_queue.get_nowait()
            logging.info("Cleared all processing queues")
        except queue.Empty:
            pass
        except Exception as e:
            logging.warning(f"Error clearing queues: {e}")

        # Clear video display
        if self.video_label_ref:
             self.video_canvas.delete("all") # Clear previous image/text
             # Check if canvas exists before creating text
             if self.video_canvas.winfo_exists():
                 canvas_w = self.video_canvas.winfo_width()
                 canvas_h = self.video_canvas.winfo_height()
                 self.video_placeholder = self.video_canvas.create_text(canvas_w//2 if canvas_w > 0 else 320,
                                                                        canvas_h//2 if canvas_h > 0 else 240,
                                                                        text="Webcam Feed Disconnected", fill="white", font=("Arial", 16))
             self.video_label_ref = None


        logging.info("Processing stopped.")


    # --- Queue Handling and UI Updates ---
    def _check_queues(self):
        """Periodically checks result queues and schedules UI updates."""
        # Check processing queue
        try:
            while True: # Process all available items
                result = self.processing_result_queue.get_nowait()
                self._update_display(result)
                # Log data for session saving
                if self.current_session_id:
                    # Add timestamp for logging
                    result['timestamp'] = time.time()
                    # Store only necessary parts if result is large
                    log_entry = {
                        'timestamp': result['timestamp'],
                        'raw_metrics': result.get('raw_metrics'),
                        'risk_results': result.get('risk_results')
                        # Avoid storing full frames in the log
                    }
                    self.session_data_log.append(log_entry)

        except queue.Empty:
            pass # No new processing results

        # Check genomic analysis output queue
        try:
            while True:
                genomic_result = self.genomic_output_queue.get_nowait()
                self.last_genomic_result = genomic_result # Store latest result
                self._update_genomic_display(genomic_result)
        except queue.Empty:
            pass # No new genomic results

        # Reschedule check if processing is still active
        if self.is_processing:
            self.root.after(50, self._check_queues) # Check again in 50ms

    def _convert_frame_to_tk(self, frame):
        """Converts an OpenCV frame (BGR) to a Tkinter PhotoImage."""
        if frame is None:
            return None
        try:
            # Ensure frame is in RGB format for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize frame to fit canvas if necessary (optional)
            # target_w, target_h = 640, 480 # Example target size
            # frame_rgb = cv2.resize(frame_rgb, (target_w, target_h))

            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            return imgtk
        except Exception as e:
            logging.error(f"Error converting frame to Tkinter format: {e}")
            return None

    def _update_display(self, result_data):
        """Updates the UI with data from the processing thread. Runs in MainThread."""
        processed_frame = result_data.get('processed_frame')
        metrics = result_data.get('raw_metrics')
        risk_results = result_data.get('risk_results') # Tuple (risk_level, factors)

        # Store latest results for potential use by analysis triggers
        self.last_eye_metrics_raw = metrics
        self.last_eye_metrics_summary = {'risk_level': risk_results[0] if risk_results else None,
                                         'contributing_factors': risk_results[1] if risk_results else {}}
        if risk_results:
            self.last_risk_level = risk_results[0]


        # --- Update Video Feed ---
        imgtk = self._convert_frame_to_tk(processed_frame)
        if imgtk and self.video_canvas.winfo_exists():
            self.video_label_ref = imgtk # Keep reference!
            self.video_canvas.delete("all") # Clear previous image/text
            # Calculate position to center image if smaller than canvas
            canvas_w = self.video_canvas.winfo_width()
            canvas_h = self.video_canvas.winfo_height()
            img_x = (canvas_w - imgtk.width()) // 2 if canvas_w > 0 else 0
            img_y = (canvas_h - imgtk.height()) // 2 if canvas_h > 0 else 0
            self.video_canvas.create_image(img_x, img_y, anchor=tk.NW, image=imgtk)
        elif self.video_placeholder is None and self.video_canvas.winfo_exists(): # Only recreate placeholder if it doesn't exist
             self.video_canvas.delete("all")
             canvas_w = self.video_canvas.winfo_width()
             canvas_h = self.video_canvas.winfo_height()
             self.video_placeholder = self.video_canvas.create_text(canvas_w//2 if canvas_w > 0 else 320,
                                                                    canvas_h//2 if canvas_h > 0 else 240,
                                                                    text="Error displaying frame", fill="red", font=("Arial", 16))


        # --- Update Metrics ---
        if metrics:
            for key, label in self.metrics_labels.items():
                value = metrics.get(key)
                if value is not None:
                    label.config(text=f"{value:.3f}" if isinstance(value, float) else str(value))
                else:
                    label.config(text="N/A")

        # --- Update Risk Meter & Factors ---
        if risk_results:
            risk_level, factors = risk_results
            # Update meter visualization
            risk_meter_img = visualization.create_risk_meter(risk_level, width=200, height=50) # Smaller meter
            risk_meter_tk = self._convert_frame_to_tk(risk_meter_img)
            if risk_meter_tk and self.risk_meter_canvas.winfo_exists():
                self.risk_meter_ref = risk_meter_tk # Keep reference
                self.risk_meter_canvas.delete("all")
                self.risk_meter_canvas.config(width=risk_meter_tk.width(), height=risk_meter_tk.height())
                self.risk_meter_canvas.create_image(0, 0, anchor=tk.NW, image=risk_meter_tk)

            # Update factors text
            factors_text = "Contributing Factors:\n" + "\n".join([f"- {k}: {v:.3f}" for k, v in factors.items()])
            self.factors_label.config(text=factors_text)
        elif self.risk_meter_canvas.winfo_exists():
             # Clear risk meter if no results
             self.risk_meter_canvas.delete("all")
             self.risk_meter_canvas.create_text(100, 25, text="Risk: N/A", fill="black")
             self.factors_label.config(text="Contributing Factors: N/A")


    def _update_genomic_display(self, genomic_result):
        """Updates the Genomic Analysis tab with results."""
        self.genomic_results_text.config(state=tk.NORMAL)
        self.genomic_results_text.delete('1.0', tk.END)
        if isinstance(genomic_result, dict) and 'error' in genomic_result:
             self.genomic_results_text.insert(tk.END, f"Error: {genomic_result['error']}")
        elif isinstance(genomic_result, dict):
             # Pretty print the dictionary result
             formatted_result = json.dumps(genomic_result, indent=4)
             self.genomic_results_text.insert(tk.END, formatted_result)
        else:
             self.genomic_results_text.insert(tk.END, str(genomic_result)) # Fallback
        self.genomic_results_text.config(state=tk.DISABLED)
        logging.info("Updated genomic display.")
        # Re-enable genomic button after completion (or handle errors)
        if self.is_processing and self.current_patient_id:
             self.genomic_trigger_button.config(state=tk.NORMAL)


    def _update_llm_display(self, llm_result, target_widget, target_button):
        """
        Updates the specified Text widget with LLM results and re-enables the
        corresponding button based on current state.
        """
        target_widget.config(state=tk.NORMAL)
        target_widget.delete('1.0', tk.END)
        target_widget.insert(tk.END, llm_result)
        target_widget.config(state=tk.DISABLED)
        logging.info("Updated LLM display.")

        # Re-enable the specific button that triggered the analysis, checking state
        if target_button:
            should_enable = False
            if target_button == self.llm_trigger_button:
                # Enable live button only if processing and patient selected
                if self.is_processing and self.current_patient_id:
                    should_enable = True
            elif target_button == self.llm_history_button:
                 # Enable history button only if a valid history session is selected
                 if self.selected_history_session_id:
                     should_enable = True

            if should_enable:
                target_button.config(state=tk.NORMAL)
            else:
                 # Ensure button remains disabled if conditions aren't met
                 # (e.g., processing stopped while LLM was running for live analysis)
                 target_button.config(state=tk.DISABLED)


    # --- Analysis Triggers ---
    def _trigger_genomic_analysis(self):
        """Sends data to the genomic analysis thread."""
        if not self.is_processing:
            messagebox.showwarning("Not Processing", "Start processing before running genomic analysis.")
            return
        if not self.current_patient_id:
             messagebox.showwarning("No Patient", "Select a patient before running genomic analysis.")
             return

        # Use the stored last risk level
        if self.last_risk_level is None:
             messagebox.showerror("Error", "No eye risk score calculated yet.")
             return

        ethnicity = self.current_patient_info.get('ethnicity', 'default')
        if not ethnicity: # Handle case where ethnicity might be None or empty string
            ethnicity = 'default'
            logging.warning("Patient ethnicity not set, using 'default' for genomic analysis.")


        logging.info(f"Queueing genomic analysis for patient {self.current_patient_id}, ethnicity {ethnicity}, risk {self.last_risk_level:.3f}")
        self.genomic_input_queue.put({'eye_risk_level': self.last_risk_level, 'ethnicity': ethnicity})
        self.genomic_trigger_button.config(state=tk.DISABLED) # Disable button while running
        # Clear previous results
        self.genomic_results_text.config(state=tk.NORMAL)
        self.genomic_results_text.delete('1.0', tk.END)
        self.genomic_results_text.insert(tk.END, "Running genomic simulation...")
        self.genomic_results_text.config(state=tk.DISABLED)


    def _trigger_llm_analysis(self, use_history=False):
        """
        Triggers analysis by the OpenRouter LLM for either the live data
        or a selected historical session.

        Args:
            use_history (bool): If True, use data from the selected historical session.
                                Otherwise, use the latest live data.
        """
        target_button = self.llm_history_button if use_history else self.llm_trigger_button
        target_text_widget = self.llm_results_text # Could have separate text widgets if desired

        if use_history:
            if not self.selected_history_session_id or not self.loaded_history_session_data:
                 messagebox.showwarning("No History Selected", "Select a session from the history list first.")
                 return
            if not self.current_patient_id: # Should be set if history is loaded, but double-check
                 messagebox.showwarning("No Patient", "Select the corresponding patient.")
                 return
        else: # Live analysis
            if not self.is_processing:
                messagebox.showwarning("Not Processing", "Start processing before running live LLM analysis.")
                return
            if not self.current_patient_id:
                 messagebox.showwarning("No Patient", "Select a patient before running live LLM analysis.")
                 return

        if not self.openrouter_client.api_key:
             messagebox.showerror("API Key Missing", "OpenRouter API key not configured.")
             return

        # --- Gather necessary data based on mode ---
        patient_data = self.current_patient_info
        eye_summary_data = None
        eye_raw_data = None
        # Use last known live genomic result for both live and history analysis for now
        # A more advanced implementation might store genomic snapshots with sessions.
        genomic_data = self.last_genomic_result if self.last_genomic_result else {'error': 'Genomic analysis not run or failed'}

        if use_history:
            logging.info(f"Gathering data for LLM analysis from historical session ID: {self.selected_history_session_id}")
            if not isinstance(self.loaded_history_session_data, list) or not self.loaded_history_session_data:
                 messagebox.showerror("Error", "Loaded historical session data is invalid or empty.")
                 target_button.config(state=tk.NORMAL) # Re-enable button
                 return

            # --- Calculate representative data from history log ---
            valid_entries = [e for e in self.loaded_history_session_data if e.get('raw_metrics') and e.get('risk_results')]
            if not valid_entries:
                 messagebox.showwarning("Missing Data", "No valid metric entries found in the selected historical session log.")
                 target_button.config(state=tk.NORMAL) # Re-enable button
                 return

            # Example: Calculate averages
            risk_values = [e['risk_results'][0] for e in valid_entries if e.get('risk_results') and e['risk_results'][0] is not None]
            avg_risk = np.mean(risk_values) if risk_values else None

            # Average raw metrics (handle potential None values)
            # Use keys from the first valid entry's raw_metrics if available
            avg_raw_metrics = {}
            if valid_entries[0].get('raw_metrics'):
                raw_metric_keys = valid_entries[0]['raw_metrics'].keys()
                for key in raw_metric_keys:
                    values = [e['raw_metrics'].get(key) for e in valid_entries if e.get('raw_metrics') and e['raw_metrics'].get(key) is not None]
                    if values and isinstance(values[0], (int, float)): # Only average numeric types
                        avg_raw_metrics[key] = np.mean(values)
                    elif values: # Keep last known non-numeric value (like 'active_eye_mode')
                        avg_raw_metrics[key] = values[-1]
                    else:
                        avg_raw_metrics[key] = None # Or 'N/A'

            # Factors are harder to average meaningfully, maybe take factors from highest risk entry?
            # For simplicity, just note that factors are not averaged.
            eye_summary_data = {'risk_level': avg_risk if avg_risk is not None and not np.isnan(avg_risk) else None,
                                'contributing_factors': {'Note': 'Factors not averaged from history'}}
            eye_raw_data = avg_raw_metrics

        else: # Use live data
            logging.info("Gathering latest live data for LLM analysis.")
            if not self.last_eye_metrics_summary or not self.last_eye_metrics_raw:
                messagebox.showwarning("Missing Data", "Live eye tracking data not available yet for LLM analysis.")
                # Don't re-enable button here, let the calling context handle it if needed
                return
            eye_summary_data = self.last_eye_metrics_summary
            eye_raw_data = self.last_eye_metrics_raw

        # Check if essential eye data was gathered successfully
        if eye_summary_data is None or eye_raw_data is None:
             messagebox.showerror("Error", "Could not gather necessary eye data for LLM analysis.")
             target_button.config(state=tk.NORMAL) # Re-enable button as the process failed early
             return

        # --- Trigger LLM ---
        logging.info(f"Triggering LLM analysis for patient {self.current_patient_id} (History: {use_history})")
        target_button.config(state=tk.DISABLED) # Disable the correct button
        target_text_widget.config(state=tk.NORMAL)
        target_text_widget.delete('1.0', tk.END)
        target_text_widget.insert(tk.END, "Requesting LLM analysis...")
        target_text_widget.config(state=tk.DISABLED)

        # Run LLM call in a separate thread to avoid blocking GUI
        def llm_task():
            analysis = self.openrouter_client.analyze_combined_data(
                eye_metrics_summary=eye_summary_data,
                eye_metrics_raw=eye_raw_data,
                genomic_results=genomic_data, # Use the potentially updated genomic_data
                patient_info=patient_data
            )
            # Schedule UI update back in the main thread
            # Pass the target widget and button to re-enable
            self.root.after(0, self._update_llm_display, analysis, target_text_widget, target_button)

        llm_thread = threading.Thread(target=llm_task, daemon=True)
        llm_thread.start()


    # --- Window Closing ---
    def _on_close(self):
        """Handles window closing event."""
        logging.info("Close button clicked.")
        if self.is_processing:
            if messagebox.askokcancel("Quit", "Processing is active. Stop processing and quit?"):
                self.stop_processing()
                # Wait a moment for threads to potentially signal stop
                time.sleep(0.5)
                self.root.destroy()
            else:
                return # Don't close if user cancels
        else:
             # Ensure DB connection for main thread is closed if open
             self.storage.close_connection()
             self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = Dashboard(root)
    root.mainloop()
