import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox # Add messagebox
from PIL import Image, ImageTk
import time
import json
from src.utils.visualization import create_metrics_visualization, create_risk_meter
from src.data.storage import DataStorage, NumpyEncoder # Import DataStorage & NumpyEncoder

class Dashboard:
    def __init__(self, eye_tracker, pd_detector, metrics_history, window_title="GenomicGuard: Early Parkinson's Detection"):
        # Store components
        self.eye_tracker = eye_tracker
        self.pd_detector = pd_detector
        self.metrics_history = metrics_history
        self.storage = DataStorage() # Initialize storage

        # Create main window FIRST
        self.window = tk.Tk()
        self.window.title(window_title)
        self.window.minsize(1200, 700)

        # --- Initialize UI Variables (AFTER creating self.window) ---
        # Tracking Variables
        self.running = False # Non-Tkinter var, can be earlier
        self.cap = None      # Non-Tkinter var
        self.after_id = None # Non-Tkinter var
        self.cycle_start_time = None
        self.cycle_metrics = []
        self.debug_mode = False # Initialize debug mode flag here

        # Patient Profile Variables (Moved earlier)
        self.patient_name_var = tk.StringVar()
        self.patient_age_var = tk.StringVar()
        self.patient_gender_var = tk.StringVar(value='Male')
        self.patient_var = tk.StringVar() # For the patient ID combobox
        self.ethnicity_var = tk.StringVar(value='chinese') # For ethnicity combobox

        # Eye Tracking Control Variables
        self.mode_var = tk.StringVar(value='face') # For face/eye mode radio buttons
        self.cycle_var = tk.BooleanVar(value=False) # For 15s cycle checkbox

        # --- Create and Setup Tabs ---
        self.tab_control = ttk.Notebook(self.window)

        # Eye tracking tab
        self.eye_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.eye_tab, text="Eye Tracking")
        self.setup_eye_tracking_tab(self.eye_tab) # Uses mode_var, cycle_var

        # Genomic analysis tab
        self.genomic_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.genomic_tab, text="Genomic Analysis")
        self.setup_genomic_tab(self.genomic_tab) # No specific vars needed before call

        # Patient analysis tab
        self.patient_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.patient_tab, text="Patient Analysis")
        # Uses patient_*, ethnicity_var - they are initialized now
        self.setup_patient_tab(self.patient_tab)

        # Pack the tab control after setting up all tabs
        self.tab_control.pack(expand=1, fill="both")

        # Initial population of patient list happens inside setup_patient_tab

    def setup_eye_tracking_tab(self, parent):
        # Create frames
        self.left_frame = ttk.Frame(parent, padding=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_frame = ttk.Frame(parent, padding=10)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create canvas for webcam feed
        self.canvas = tk.Canvas(self.left_frame, width=640, height=480)
        self.canvas.pack(padx=10, pady=10)

        # Create eye tracking visualization area
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        
        fig = Figure(figsize=(5, 3), dpi=100)
        self.eye_plot = fig.add_subplot(111)
        self.eye_plot.set_title("Eye Movement Tracking")
        self.eye_plot.set_xlim(-1, 1)  # Normalized coordinates
        self.eye_plot.set_ylim(-1, 1)
        self.eye_canvas = FigureCanvasTkAgg(fig, master=self.left_frame)
        self.eye_canvas.get_tk_widget().pack(padx=10, pady=10)

        # Create controls
        self.controls_frame = ttk.LabelFrame(self.left_frame, text="Controls", padding=10)
        self.controls_frame.pack(fill=tk.X, padx=10, pady=10)

        # Add mode switch
        self.mode_var = tk.StringVar(value='face')
        ttk.Radiobutton(self.controls_frame, text="Face Mode", variable=self.mode_var, 
                       value='face', command=self.update_tracker_mode).pack(side=tk.LEFT)
        ttk.Radiobutton(self.controls_frame, text="Eye Mode", variable=self.mode_var,
                       value='eye', command=self.update_tracker_mode).pack(side=tk.LEFT)

        self.btn_start = ttk.Button(self.controls_frame, text="Start", command=self.on_start)
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_stop = ttk.Button(self.controls_frame, text="Stop", command=self.on_stop)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        self.btn_debug = ttk.Button(self.controls_frame, text="Debug View", command=self.toggle_debug)
        self.btn_debug.pack(side=tk.LEFT, padx=5)
        self.debug_mode = False

        # Add 15-second cycle checkbox
        self.cycle_var = tk.BooleanVar(value=False)
        self.cycle_check = ttk.Checkbutton(self.controls_frame, text="15s Analysis Cycles",
                                         variable=self.cycle_var)
        self.cycle_check.pack(side=tk.LEFT, padx=5)

        # Create metrics frame
        self.metrics_frame = ttk.LabelFrame(self.right_frame, text="Eye Metrics", padding=10)
        self.metrics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Text widget for metrics
        self.metrics_text = tk.Text(self.metrics_frame, width=40, height=10)
        self.metrics_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create risk assessment frame
        self.risk_frame = ttk.LabelFrame(self.right_frame, text="Risk Assessment", padding=10)
        self.risk_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Risk meter label
        self.risk_label = ttk.Label(self.risk_frame, text="Risk Level: Collecting data...")
        self.risk_label.pack(padx=5, pady=5)

        # Canvas for risk meter visualization
        self.risk_canvas = tk.Canvas(self.risk_frame, width=400, height=100)
        self.risk_canvas.pack(padx=5, pady=5)

        # Frame for risk factors
        self.factors_frame = ttk.LabelFrame(self.risk_frame, text="Risk Factors", padding=10)
        self.factors_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Text widget for risk factors
        self.factors_text = tk.Text(self.factors_frame, width=40, height=10)
        self.factors_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Canvas for metrics visualization
        self.vis_canvas = tk.Canvas(self.right_frame, width=400, height=300)
        self.vis_canvas.pack(padx=10, pady=10)

    def update_tracker_mode(self):
        """Update the eye tracker mode based on UI selection"""
        self.eye_tracker.current_mode = self.mode_var.get()
        self.eye_tracker._init_face_mesh()
        self.metrics_text.insert(tk.END, f"Switched to {self.mode_var.get()} mode\n")

    def on_start(self):
        """Start the eye tracking process"""
        if not self.running:
            self.cycle_start_time = None # Reset cycle timer on start
            self.cycle_metrics = []      # Reset cycle metrics on start
            try:
                self.running = True
                self.cap = cv2.VideoCapture(0)  # Open default camera
                if not self.cap.isOpened():
                    raise RuntimeError("Could not open video device")
                
                self.metrics_text.insert(tk.END, "Started eye tracking...\n")
                self.btn_start.config(state=tk.DISABLED)
                self.btn_stop.config(state=tk.NORMAL)
                self.update_frame()
                
            except Exception as e:
                self.running = False
                if self.cap:
                    self.cap.release()
                    self.cap = None
                self.metrics_text.insert(tk.END, f"Error: {str(e)}\n")
                self.btn_start.config(state=tk.NORMAL)

    def on_stop(self):
        """Stop the eye tracking process"""
        if self.running:
            self.running = False
            if self.after_id:
                self.window.after_cancel(self.after_id)
                self.after_id = None
            if self.cap:
                self.cap.release()
                self.cap = None
            self.metrics_text.insert(tk.END, "Stopped eye tracking\n")
            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)

    def update_frame(self):
        """Update the video frame and process eye tracking"""
        if self.running and self.cap:
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    raise ValueError("Failed to read frame from camera")
                
                # Process frame with eye tracker (returns processed frame + metrics)
                processed_frame, metrics = self.eye_tracker.process_frame(
                    frame,
                    debug_mode=self.debug_mode
                )
                
                # Update frame with potentially drawn debug info
                frame = processed_frame
                
                # Always update metrics (may contain error messages)
                self.update_metrics(metrics)
                
                # Handle 15-second analysis cycle
                if self.cycle_var.get():
                    if self.cycle_start_time is None:
                        self.cycle_start_time = time.time()
                    
                    if isinstance(metrics, dict) and metrics.get('error') is None:
                        self.cycle_metrics.append(metrics)
                    
                    if time.time() - self.cycle_start_time >= 15.0 and self.cycle_metrics:
                        print(f"Processing {len(self.cycle_metrics)} metrics from the last 15 seconds...")
                        self.process_cycle_metrics(self.cycle_metrics)
                        self.cycle_metrics = []
                        self.cycle_start_time = None
                else:
                    # If cycle mode is off, reset
                    self.cycle_start_time = None
                    self.cycle_metrics = []

            except Exception as e:
                error_msg = f"Frame processing error: {str(e)}"
                print(error_msg)
                self.metrics_text.insert(tk.END, error_msg + "\n")
                # Continue running despite the error
                
                # Convert frame to RGB and display with robust error handling
                try:
                    if frame is None or frame.size == 0:
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    img = Image.fromarray(frame)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.canvas.imgtk = imgtk
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                except Exception as e:
                    print(f"Error displaying frame: {str(e)}")
                    # Fallback to blank frame
                    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    img = Image.fromarray(blank_frame)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.canvas.imgtk = imgtk
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                
                # Schedule next frame update
                self.after_id = self.window.after(10, self.update_frame)

    def process_cycle_metrics(self, metrics_list):
        """Process metrics collected over a 15-second cycle."""
        if not metrics_list:
            return
            
        try:
            # Ensure we're working with a list of dictionaries
            valid_metrics = [m for m in metrics_list if isinstance(m, dict)]
            
            if not valid_metrics:
                print("No valid metrics found in cycle")
                return
                
            print(f"Processing cycle with {len(valid_metrics)} valid data points")
            
            # Calculate aggregated metrics
            metrics = {
                'avg_saccade_velocity': np.mean([
                    m.get('avg_saccade_velocity', 0)
                    for m in valid_metrics
                    if 'avg_saccade_velocity' in m
                ]).item() if len(valid_metrics) > 0 else 0,
                
                'avg_fixation_stability': np.mean([
                    m.get('avg_fixation_stability', 0)
                    for m in valid_metrics
                    if 'avg_fixation_stability' in m
                ]).item() if len(valid_metrics) > 0 else 0,
                
                'mode': valid_metrics[-1].get('mode', 'face')  # Use most recent mode
            }
            
            ethnicity = self.ethnicity_var.get() if hasattr(self, 'ethnicity_var') else 'chinese'
            self.update_risk_assessment(metrics, ethnicity)
            
        except Exception as e:
            print(f"Error processing cycle metrics: {str(e)}")
        self.update_risk_assessment(metrics_list, ethnicity)

    def update_risk_assessment(self, metrics_data, ethnicity='chinese'):
        """Update the risk assessment with robust error handling"""
        if not metrics_data:
            return
            
        try:
            # Ensure input is in correct format
            if isinstance(metrics_data, np.ndarray):
                metrics_data = metrics_data.item() if metrics_data.size == 1 else metrics_data.tolist()
                
            if isinstance(metrics_data, list):
                # Use last valid metric if available
                prediction_input = next((m for m in reversed(metrics_data) if isinstance(m, dict)), {})
            elif isinstance(metrics_data, dict):
                prediction_input = metrics_data
            else:
                raise ValueError("Invalid metrics data format")

            risk_level, factors = self.pd_detector.predict(prediction_input, ethnicity)
            # If it's a list (from cycle), maybe average or use sequence
            if isinstance(metrics_data, list):
                 # Placeholder: Use the last metric or average specific features
                 if metrics_data:
                     prediction_input = metrics_data[-1] # Example: use last metric
                 else: return
            else: # Single metric dict
                 prediction_input = metrics_data

            # TODO: Ensure prediction_input format matches pd_detector expectations
            # Pass ethnicity to the detector's predict method
            risk_level, factors = self.pd_detector.predict(prediction_input, ethnicity=ethnicity)

            # Update UI
            self.risk_label.config(text=f"Risk Level: {risk_level}")
            
            # Update risk meter visualization
            img = create_risk_meter(risk_level) # Assuming this returns a PIL Image
            self.risk_meter_imgtk = ImageTk.PhotoImage(image=img)
            self.risk_canvas.delete("all") # Clear previous drawing
            self.risk_canvas.create_image(0, 0, anchor=tk.NW, image=self.risk_meter_imgtk)

            # Update factors text
            self.factors_text.delete(1.0, tk.END)
            self.factors_text.insert(tk.END, "Contributing Factors:\n")
            if factors:
                 for factor, value in factors.items():
                      self.factors_text.insert(tk.END, f"- {factor}: {value}\n")
            else:
                 self.factors_text.insert(tk.END, "N/A\n")

        except Exception as e:
            print(f"Error during risk assessment: {e}")
            self.risk_label.config(text="Risk Level: Error")
            self.factors_text.delete(1.0, tk.END)
            self.factors_text.insert(tk.END, f"Error: {e}\n")


    def run(self):
        """Run the main application loop"""
        try:
            # Bind close event
            self.window.protocol("WM_DELETE_WINDOW", self.close)
            self.window.mainloop()
        except Exception as e:
            print(f"Error running application: {e}")
        finally:
            # Ensure cleanup happens even if mainloop errors
            self.close()

    def setup_genomic_tab(self, parent):
        """Setup the genomic analysis tab"""
        # Create main frame
        main_frame = ttk.Frame(parent, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Analysis controls frame
        controls_frame = ttk.LabelFrame(main_frame, text="Genomic Analysis Controls", padding=10)
        controls_frame.pack(fill=tk.X, padx=10, pady=10)

        # Add analysis button
        self.btn_run_genomic = ttk.Button(controls_frame, text="Run Analysis",
                                        command=self.run_bionemo_analysis)
        self.btn_run_genomic.pack(side=tk.LEFT, padx=5)

        # Add LLM analysis button
        # LLM button removed from genomic tab

        # Results display area
        results_frame = ttk.LabelFrame(main_frame, text="Genomic Analysis Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Text widget for results
        self.genomic_text = scrolledtext.ScrolledText(results_frame, width=80, height=20)
        self.genomic_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def run_bionemo_analysis(self):
        """Run BioNemo genomic analysis"""
        self.genomic_text.delete(1.0, tk.END)
        self.genomic_text.insert(tk.END, "Running BioNemo genomic analysis...\n")
        # TODO: Implement actual BioNemo analysis
        self.genomic_text.insert(tk.END, "Analysis complete. No significant markers found.\n")

    def run_llm_analysis(self):
        """Run LLM-based genomic analysis (Placeholder - likely deprecated by patient tab version)"""
        # This method might be removed if LLM analysis is only done per patient
        self.genomic_text.delete(1.0, tk.END)
        self.genomic_text.insert(tk.END, "Running LLM genomic analysis (Genomic Tab - Placeholder)...\n")
        # TODO: Implement actual LLM analysis or remove this if superseded
        self.genomic_text.insert(tk.END, "Analysis complete. No significant findings.\n")

    def run_llm_analysis_patient(self):
        """Run LLM analysis on the currently loaded patient data."""
        patient_id = self.patient_combo.get().strip()
        if not patient_id:
             # Use metrics_text for general UI feedback if possible
             self.metrics_text.insert(tk.END, "Error: Please load or select a patient first for LLM analysis.\n")
             return
        
        if not self.metrics_history:
             self.metrics_text.insert(tk.END, f"Error: No metrics data loaded for patient {patient_id} to analyze.\n")
             return

        # Placeholder for LLM analysis based on self.metrics_history
        print(f"Running LLM analysis for patient {patient_id} with {len(self.metrics_history)} data points...")
        # Provide feedback in the patient analysis results area or a dedicated LLM output area if added
        # For now, using metrics_text as a general feedback area
        self.metrics_text.insert(tk.END, f"Running LLM analysis for patient {patient_id}...\n")
        # TODO: Implement actual LLM call using self.metrics_history (e.g., summarize trends, risks)
        # Example: llm_client.analyze(self.metrics_history)
        time.sleep(1) # Simulate analysis time
        self.metrics_text.insert(tk.END, f"LLM analysis complete for {patient_id}. Insights: [Placeholder - Check console/log]\n")
        # TODO: Display LLM results properly in the UI, perhaps in a new text widget on the patient tab
    def setup_patient_tab(self, parent):
        """Setup the patient analysis tab"""
        # Create main frame for the entire tab content
        tab_main_frame = ttk.Frame(parent, padding=10)
        tab_main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Top Frame for Patient Selection/Profile ---
        top_frame = ttk.Frame(tab_main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        # Patient Selection/Management Frame (Left side of top_frame)
        mgmt_frame = ttk.LabelFrame(top_frame, text="Patient Management", padding=10)
        mgmt_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        # Patient Profile Frame (Right side of top_frame)
        profile_frame = ttk.LabelFrame(top_frame, text="Patient Profile", padding=10)
        profile_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        # --- Populate Management Frame ---

        # Patient ID Selection
        ttk.Label(mgmt_frame, text="Patient ID:").grid(row=0, column=0, padx=(0, 5), pady=5, sticky=tk.W)
        self.patient_var = tk.StringVar()
        self.patient_combo = ttk.Combobox(mgmt_frame, textvariable=self.patient_var, width=25) # Wider
        self.patient_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        self._update_patient_list() # Populate dropdown
        # Bind selection change to load patient? Optional, use Load button for now.
        # self.patient_combo.bind('<<ComboboxSelected>>', lambda e: self.load_patient())

        # Ethnicity Selection
        ttk.Label(mgmt_frame, text="Ethnicity:").grid(row=1, column=0, padx=(0, 5), pady=5, sticky=tk.W)
        self.ethnicity_var = tk.StringVar(value='chinese') # Default value
        self.ethnicity_combo = ttk.Combobox(mgmt_frame, textvariable=self.ethnicity_var,
                                            values=['chinese', 'malay', 'indian', 'other'], width=10, state='readonly')
        self.ethnicity_combo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        # Buttons Frame within Management Frame
        button_frame_mgmt = ttk.Frame(mgmt_frame)
        button_frame_mgmt.grid(row=2, column=0, columnspan=2, pady=(10,0))

        self.btn_load = ttk.Button(button_frame_mgmt, text="Load Patient", command=self.load_patient)
        self.btn_load.pack(side=tk.LEFT, padx=5)
        
        # Save button now saves PROFILE data via storage class
        self.btn_save_profile = ttk.Button(button_frame_mgmt, text="Save Profile", command=self.save_patient_profile) # Renamed command
        self.btn_save_profile.pack(side=tk.LEFT, padx=5)

        # LLM Analysis Button
        self.btn_run_llm_patient = ttk.Button(button_frame_mgmt, text="Run LLM Analysis",
                                            command=self.run_llm_analysis_patient)
        self.btn_run_llm_patient.pack(side=tk.LEFT, padx=5)

        # --- Populate Profile Frame ---
        ttk.Label(profile_frame, text="Name:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.patient_name_entry = ttk.Entry(profile_frame, textvariable=self.patient_name_var, width=30) # Correct variable
        self.patient_name_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)

        ttk.Label(profile_frame, text="Age:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.patient_age_entry = ttk.Entry(profile_frame, textvariable=self.patient_age_var, width=5)
        self.patient_age_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)

        ttk.Label(profile_frame, text="Gender:").grid(row=1, column=2, padx=5, pady=2, sticky=tk.W)
        self.patient_gender_combo = ttk.Combobox(profile_frame, textvariable=self.patient_gender_var,
                                                 values=['Male', 'Female', 'Other'], width=8, state='readonly')
        self.patient_gender_combo.grid(row=1, column=3, padx=5, pady=2, sticky=tk.W)

        ttk.Label(profile_frame, text="Medical History:").grid(row=2, column=0, padx=5, pady=2, sticky=tk.NW)
        self.medical_history_text = scrolledtext.ScrolledText(profile_frame, width=40, height=3, wrap=tk.WORD)
        self.medical_history_text.grid(row=2, column=1, columnspan=3, padx=5, pady=2, sticky=tk.EW)

        # --- Bottom Frame for Analysis Results ---
        # Results display area (takes remaining space)
        results_frame = ttk.LabelFrame(tab_main_frame, text="Patient Analysis Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0) # No extra padding needed

        # Notebook for different views within results
        self.patient_notebook = ttk.Notebook(results_frame)
        self.patient_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5) # Padding inside the frame

        # Metrics tab
        self.metrics_tab = ttk.Frame(self.patient_notebook)
        self.patient_notebook.add(self.metrics_tab, text="Metrics History")
        self.setup_metrics_tab(self.metrics_tab)

        # Risk assessment tab
        self.risk_tab = ttk.Frame(self.patient_notebook)
        self.patient_notebook.add(self.risk_tab, text="Risk Assessment")
        self.setup_risk_tab(self.risk_tab)

    def setup_metrics_tab(self, parent):
        """Setup metrics history visualization for the patient tab"""
        self.patient_metrics_canvas = tk.Canvas(parent) # Renamed for clarity
        self.patient_metrics_canvas.pack(fill=tk.BOTH, expand=True)

    def setup_risk_tab(self, parent):
        """Setup risk assessment visualization for the patient tab"""
        self.patient_risk_canvas = tk.Canvas(parent) # Renamed for clarity
        self.patient_risk_canvas.pack(fill=tk.BOTH, expand=True)

    def load_patient(self):
        """Load patient profile from DB and metrics history from JSON file."""
        patient_id = self.patient_combo.get().strip()
        if not patient_id:
            self.metrics_text.insert(tk.END, "Error: Please select or enter a Patient ID to load.\n")
            return

        # --- Load Patient Profile from DB ---
        try:
            # Use the new get_patient method
            patient_profile = self.storage.get_patient(patient_id)
            if patient_profile:
                 # TODO: Update UI elements with profile data (name, age, etc.) when they are added
                 print(f"Loaded profile for {patient_id}: {patient_profile}")
                 self.metrics_text.insert(tk.END, f"Loaded profile for Patient ID: {patient_id}\n")
                 # Example: self.patient_name_var.set(patient_profile.get('name', ''))
            else:
                 self.metrics_text.insert(tk.END, f"No profile found in DB for Patient ID: {patient_id}\n")
                 # Decide if we should proceed to load session data even without profile
                 # For now, we'll proceed

        except Exception as e:
             self.metrics_text.insert(tk.END, f"Error loading profile for {patient_id} from DB: {e}\n")
             # Decide if we should stop or try loading session data

        # --- Load Metrics History from JSON (as before) ---
        # Note: This assumes metrics_history is saved separately per patient ID,
        # matching the save_patient logic. If sessions are saved instead, this needs adjustment.
        # Construct path relative to the project structure
        # Construct path relative to the project structure (assuming CWD is PycharmProjects)
        # Use new path within src/data
        data_dir = os.path.join("GenomeGuard", "src", "data", "patient")
        filename = os.path.join(data_dir, f"{patient_id}_data.json")

        if not os.path.exists(filename):
            self.metrics_text.insert(tk.END, f"No metrics history file found for Patient ID: {patient_id} at {filename}\n")
            self.metrics_history = [] # Clear history if file not found
            self.update_patient_views() # Update UI to show no data
            return

        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Load metrics history from the JSON file
            self.metrics_history = data.get('metrics_history', [])
            self.metrics_text.insert(tk.END, f"Loaded metrics history for Patient ID: {patient_id} ({len(self.metrics_history)} records)\n")
            self.update_patient_views() # Update UI with loaded metrics data

        except Exception as e:
            self.metrics_text.insert(tk.END, f"Error loading metrics history for {patient_id} from {filename}: {e}\n")
            self.metrics_history = [] # Clear history on error
            self.update_patient_views() # Update UI

    def save_patient_profile(self): # Renamed method
        """Saves the patient profile data entered in the UI to the database."""
        name = self.patient_name_var.get().strip()
        age_str = self.patient_age_var.get().strip()
        
        if not name:
            messagebox.showwarning("Invalid Data", "Patient name is required.")
            return
            
        try:
            age = int(age_str) if age_str.isdigit() else 0
        except ValueError:
            messagebox.showwarning("Invalid Data", "Age must be a valid number.")
            return

        # Extract patient ID from combobox (might be "ID - Name" or just ID/Name)
        # We prioritize saving under the ID part if it exists, otherwise save potentially new patient
        patient_selection = self.patient_combo.get().strip()
        # Handle cases where only name is entered or "ID - Name" format is used
        patient_id_part = None
        if ' - ' in patient_selection:
            patient_id_part = patient_selection.split(' - ')[0]
            # Check if the name part matches the current name field
            name_part = patient_selection.split(' - ', 1)[1]
            if name_part != name:
                # Name changed or new patient based on existing ID format, treat as new/update
                 patient_id_part = None # Let storage handle ID generation/replacement logic based on name
        elif patient_selection == name: # Only name entered, likely new patient
             patient_id_part = None
        else: # Something else entered, could be just an ID or a new name
             # If it looks like an existing ID format (e.g., PT12345), use it
             if patient_selection.startswith("PT") and patient_selection[2:].isdigit():
                  patient_id_part = patient_selection
             else: # Treat as potentially new patient name, let storage handle ID
                  patient_id_part = None


        patient_data = {
            'id': patient_id_part, # Pass potential existing ID or None
            'name': name,
            'age': age,
            'gender': self.patient_gender_var.get(),
            'medical_history': self.medical_history_text.get(1.0, tk.END).strip() # Get from Text widget
        }
        
        try:
            # Use storage class to save profile to DB (handles INSERT OR REPLACE)
            saved_id = self.storage.save_patient(patient_data)
            
            # Update UI to reflect saved state
            new_selection_text = f"{saved_id} - {name}"
            self.patient_var.set(new_selection_text) # Update combobox display variable
            self._update_patient_list() # Refresh dropdown list with potentially new/updated entry
            
            # Ensure the newly saved/updated patient is selected in the combobox
            # This might require finding the index if _update_patient_list doesn't preserve selection perfectly
            if new_selection_text in self.patient_combo['values']:
                 self.patient_combo.set(new_selection_text)
            
            messagebox.showinfo("Patient Saved", f"Patient profile for '{name}' saved successfully with ID: {saved_id}")
            self.metrics_text.insert(tk.END, f"Saved profile for Patient ID: {saved_id}\n")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save patient profile: {e}")
            self.metrics_text.insert(tk.END, f"Error saving profile: {e}\n")

    # Note: Saving metrics_history to JSON is now separate.
    # Consider adding a "Save Session" button if needed.

        # --- Also Save Metrics History to JSON ---
        # This saves the current in-memory metrics history when the profile is saved.
        if self.metrics_history: # Only save if there's history
            # Construct path relative to project root
            metrics_data_dir = os.path.join("GenomeGuard", "patient_data")
            os.makedirs(metrics_data_dir, exist_ok=True)
            metrics_filename = os.path.join(metrics_data_dir, f"{saved_id}_data.json")

            metrics_to_save = {
                'patient_id': saved_id,
                'save_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'metrics_history': self.metrics_history
            }

            try:
                with open(metrics_filename, 'w') as f:
                    # Use NumpyEncoder from storage.py if metrics contain numpy types
                    json.dump(metrics_to_save, f, cls=NumpyEncoder, indent=4)
                self.metrics_text.insert(tk.END, f"Saved current metrics history for {saved_id} to {metrics_filename}\n")
            except Exception as e:
                self.metrics_text.insert(tk.END, f"Error saving metrics history for {saved_id}: {e}\n")
        else:
             self.metrics_text.insert(tk.END, f"No current metrics history to save for {saved_id}.\n")

    # Note: Saving metrics_history to JSON is currently tied to saving the profile.
    def update_patient_views(self):
        """Update patient tab visualizations based on loaded metrics_history."""
        self.patient_metrics_canvas.delete("all")
        self.patient_risk_canvas.delete("all") # Use the renamed canvas

        if self.metrics_history:
            try:
                # Visualize metrics history
                # Assuming create_metrics_visualization returns a PIL Image
                metrics_img = create_metrics_visualization(self.metrics_history)
                # Keep reference to avoid garbage collection
                self.patient_metrics_vis_imgtk = ImageTk.PhotoImage(image=metrics_img)
                self.patient_metrics_canvas.create_image(0, 0, anchor=tk.NW, image=self.patient_metrics_vis_imgtk)
                
                # TODO: Update risk assessment visualization on patient_risk_canvas based on historical data
                # Example: Draw a historical risk meter or trend
                # risk_history = [m.get('risk_level') for m in self.metrics_history if m and 'risk_level' in m]
                # if risk_history:
                #    # Use create_risk_meter or another function for historical view
                #    pass

            except Exception as e:
                print(f"Error updating patient visualizations: {e}")
                self.patient_metrics_canvas.create_text(10, 10, anchor=tk.NW, text=f"Error: {e}")
        else:
             self.patient_metrics_canvas.create_text(10, 10, anchor=tk.NW, text="No patient data loaded.")
             self.patient_risk_canvas.create_text(10, 10, anchor=tk.NW, text="No patient data loaded.")

    def toggle_debug(self):
        """Toggle debug visualization mode"""
        self.debug_mode = not self.debug_mode
        self.eye_tracker.debug = self.debug_mode
        debug_status = "ON" if self.debug_mode else "OFF"
        self.metrics_text.insert(tk.END, f"Debug mode {debug_status}\n")

    def update_metrics(self, metrics):
        """Update metrics display and visualizations"""
        if not isinstance(metrics, dict):
            return
        
        try:
            # Update text metrics display
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(tk.END, "Current Metrics:\n")
            
            # Filter out non-serializable/numeric values for display
            displayable_metrics = {k:v for k,v in metrics.items()
                                 if isinstance(v, (int, float, str))}
            for k, v in displayable_metrics.items():
                self.metrics_text.insert(tk.END, f"{k}: {v}\n")

            # Update eye movement visualization if available
            if hasattr(self, 'eye_plot'):
                # Handle missing/None values safely
                x = metrics.get('gaze_x', 0) or 0
                y = metrics.get('gaze_y', 0) or 0
                
                if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                    return
                
                self.eye_plot.clear()
                self.eye_plot.scatter(x, y, color='red', s=100, alpha=0.5)
                self.eye_plot.set_title("Eye Position")
                self.eye_plot.set_xlim(0, 1)  # Relative coordinates
                self.eye_plot.set_ylim(0, 1)
                self.eye_plot.grid(True)
                
                # Draw movement path history
                if hasattr(self, 'metrics_history'):
                    xs = [m.get('gaze_x', 0) for m in self.metrics_history[-20:]
                         if isinstance(m, dict) and isinstance(m.get('gaze_x'), (int, float))]
                    ys = [m.get('gaze_y', 0) for m in self.metrics_history[-20:]
                         if isinstance(m, dict) and isinstance(m.get('gaze_y'), (int, float))]
                    
                    if xs and ys:
                        self.eye_plot.plot(xs, ys, '-o', color='blue', alpha=0.3, linewidth=1)
                
                self.eye_canvas.draw()

        except Exception as e:
            print(f"Error updating metrics display: {str(e)}")
            self.metrics_text.insert(tk.END, f"\nDisplay error: {str(e)}\n")
        """Update metrics display and visualizations"""
        if not isinstance(metrics, dict):
            print(f"Warning: Unexpected metrics type received: {type(metrics)}. Expected dict.")
            return

        # Update text metrics display
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, "Current Metrics:\n")
        for k, v in metrics.items():
            self.metrics_text.insert(tk.END, f"{k}: {v}\n")

        # Update eye movement visualization if available
        if hasattr(self, 'eye_plot'):
            x = metrics.get('gaze_x', 0)
            y = metrics.get('gaze_y', 0)
            
            self.eye_plot.clear()
            self.eye_plot.scatter(x, y, color='red', s=100, alpha=0.5)
            self.eye_plot.set_title("Eye Movement Tracking")
            self.eye_plot.set_xlim(-1, 1)  # Normalized coordinates
            self.eye_plot.set_ylim(-1, 1)
            self.eye_plot.grid(True)
            
            # Draw movement path from history
            if hasattr(self, 'metrics_history'):
                xs = [m.get('gaze_x', 0) for m in self.metrics_history[-100:] if isinstance(m, dict)]
                ys = [m.get('gaze_y', 0) for m in self.metrics_history[-100:] if isinstance(m, dict)]
                if xs and ys:
                    self.eye_plot.plot(xs, ys, '-o', color='blue', alpha=0.3, linewidth=1)
            
            self.eye_canvas.draw()
        """Update metrics display with mode-specific information"""
        self.metrics_text.delete(1.0, tk.END)

        # Safeguard: Check if metrics is actually a dictionary
        if not isinstance(metrics, dict):
            error_msg = f"Error: update_metrics received type {type(metrics)}, expected dict. Value: {metrics}"
            print(error_msg)
            self.metrics_text.insert(tk.END, error_msg + "\n")
            # Attempt to use default values or return early
            metrics = {} # Use empty dict to prevent further errors in this method
            # return # Or simply return if preferred
        
        if metrics:
            # Add mode information
            self.metrics_text.insert(tk.END, f"Tracking Mode: {metrics.get('mode', 'unknown').upper()}\n\n")
            
            # Add iris offset if in eye mode
            if metrics.get('mode') == 'eye' and 'iris_offset_x' in metrics and 'iris_offset_y' in metrics:
                self.metrics_text.insert(tk.END, 
                    f"Iris Offset - X: {metrics['iris_offset_x']:.2f}, Y: {metrics['iris_offset_y']:.2f}\n\n")
            
            # Add standard metrics
            for key, value in metrics.items():
                if key not in ['mode', 'iris_offset_x', 'iris_offset_y']:
                    if isinstance(value, (int, float)):
                        self.metrics_text.insert(tk.END, f"{key}: {value:.4f}\n")
                    else:
                        self.metrics_text.insert(tk.END, f"{key}: {value}\n")
            
            # Update risk assessment in real-time if not in cycle mode
            if not self.cycle_var.get():
                 ethnicity = self.ethnicity_var.get() if hasattr(self, 'ethnicity_var') else 'chinese' # Get selected ethnicity
                 self.update_risk_assessment(metrics, ethnicity)

    def close(self):
        """Cleanup resources and close the application."""
        print("Closing application...")
        if self.running:
            self.on_stop() # Stop camera feed and processing
        
        # Explicitly release camera if on_stop didn't (e.g., if error occurred before stop)
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("Camera released.")
            
        self.window.destroy()
        print("Window destroyed.")

    def _update_patient_list(self):
        """Scans the patient_data directory and updates the patient combobox."""
        data_dir = "patient_data"
        try:
            if not os.path.exists(data_dir):
                self.patient_combo['values'] = []
                return

            patient_files = [f for f in os.listdir(data_dir) if f.endswith("_data.json")]
            patient_ids = sorted([f.replace("_data.json", "") for f in patient_files])
            
            current_value = self.patient_var.get() # Preserve selection if possible
            self.patient_combo['values'] = patient_ids
            if current_value in patient_ids:
                 self.patient_var.set(current_value)
            elif patient_ids:
                 self.patient_var.set(patient_ids[0]) # Select first if current is invalid
            else:
                 self.patient_var.set("") # Clear if no patients

        except Exception as e:
            print(f"Error updating patient list: {e}")
            self.patient_combo['values'] = []
