import tkinter as tk
from tkinter import messagebox, simpledialog, ttk, Canvas, Text
import cv2
from PIL import Image, ImageTk
import time
import queue
import numpy as np
import os
import json
import logging
from pathlib import Path

# Configure logging at module level first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Project Imports ---
# Use absolute imports now instead of relative
try:
    from src.utils.config import Config
    from src.data.storage import StorageManager
    from src.models.eye_tracker import EyeTracker
    from src.models.pd_detector import PDDetector, RiskLevel
    from src.genomic.bionemo_client import BioNeMoRiskAssessor
    from src.llm.openrouter_client import OpenRouterClient
    from src.utils.threading_utils import RTSPCameraStream, ProcessingThread, GenomicAnalysisThread, LLMAnalysisThread
    from src.utils.visualization import draw_risk_meter
except ImportError as e:
    logger.error(f"Import Error: {e}. Ensure PYTHONPATH is set correctly or run from project root.", exc_info=True)
    raise ImportError(f"Failed to import required modules: {e}")

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

# --- Constants ---
DEFAULT_FONT = ("Arial", 10)
SECTION_PADY = 5
SECTION_PADDING = 5

# --- Main Dashboard Class ---
class Dashboard:
    MIN_WIDTH = 1440 # Wider for tabbed interface
    MIN_HEIGHT = 900
    COLOR_PRIMARY = '#2c7fb8'
    COLOR_ALERT = '#ffa500'
    COLOR_SAFE = '#4caf50'
    COLOR_WARNING = '#ffeb3b'
    COLOR_DANGER = '#f44336'
    FONT_FAMILY = 'Roboto'
    FONT_SIZES = {
        'body': 16,
        'subheader': 18,
        'header': 20
    }

    def __init__(self, root):
        self.root = root
        logger.info("Initializing Dashboard...")

        # --- Core Components ---
        try:
            self.config = Config()
            self.storage = StorageManager()
            self.eye_tracker = EyeTracker()
            self.pd_detector = PDDetector()
            self.bionemo_assessor = BioNeMoRiskAssessor()
            self.llm_client = OpenRouterClient() if self.config.get_openrouter_api_key() else None
            logger.info("Core components initialized.")
        except Exception as e:
            logger.critical(f"Failed to initialize core components: {e}", exc_info=True)
            messagebox.showerror("Initialization Error", f"Failed to initialize core components:\n{e}")
            self.root.destroy()
            return

        # --- Threading Components ---
        self.result_queue = queue.Queue(maxsize=50)
        self.genomic_queue = queue.Queue(maxsize=10)
        self.llm_queue = queue.Queue(maxsize=5)
        self.video_thread = None
        self.processing_thread = None
        self.genomic_thread = None
        self.llm_thread = None
        logger.info("Threading components initialized.")

        # --- Session State ---
        self.session_active = False
        self.current_patient_id = None
        self.current_patient_info = None
        self.current_session_id = None
        self.session_start_time = None
        self.last_metrics_update = {} # Store latest full metrics dict
        self.last_genomic_update = {} # Store latest genomic results
        self.last_risk_assessment = (RiskLevel.LOW, "N/A", 0.0, {}) # level, reason, ocular_score, factors

        # --- UI Setup ---
        logger.info("Setting up UI...")
        self.setup_ui()
        logger.info("UI setup complete.")

        # --- Start Queue Checking & Handle Close ---
        self.check_queues()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def setup_ui(self):
        self.root.title("WorkingGenomeGuard - Clinical Dashboard")
        self.root.geometry(f"{self.MIN_WIDTH}x{self.MIN_HEIGHT}")
        self.root.minsize(self.MIN_WIDTH, self.MIN_HEIGHT)

        # --- Main Layout ---
        # Use PanedWindow for resizable sections if desired, or simple packing
        top_frame = ttk.Frame(self.root, padding=SECTION_PADDING)
        top_frame.pack(fill=tk.X)

        # --- Top Bar (Patient Management & Session Control) ---
        patient_frame = ttk.LabelFrame(top_frame, text="Patient Management", padding=SECTION_PADDING)
        patient_frame.pack(side=tk.LEFT, padx=5, pady=SECTION_PADY, fill=tk.X, expand=True)
        ttk.Button(patient_frame, text="Load Patient", command=self.load_patient).pack(side=tk.LEFT, padx=5)
        # TODO: Add New Patient functionality (similar to TrueGenomeGuard's PatientDialog)
        self.patient_info_label = ttk.Label(patient_frame, text="No patient selected.", relief=tk.SUNKEN, padding=2, width=50)
        self.patient_info_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        session_frame = ttk.LabelFrame(top_frame, text="Session Control", padding=SECTION_PADDING)
        session_frame.pack(side=tk.LEFT, padx=5, pady=SECTION_PADY)
        self.start_btn = ttk.Button(session_frame, text="Start Session", command=self.start_session, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(session_frame, text="Stop Session", command=self.stop_session, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.session_status_label = ttk.Label(session_frame, text="Status: Idle", relief=tk.SUNKEN, padding=2, width=20)
        self.session_status_label.pack(side=tk.LEFT, padx=5)

        # --- Main Content Area (Divided Vertically) ---
        content_pane = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        content_pane.pack(fill=tk.BOTH, expand=True, padx=SECTION_PADDING, pady=SECTION_PADY)

        # --- Top Row in Content Area ---
        top_row_pane = ttk.PanedWindow(content_pane, orient=tk.HORIZONTAL)
        content_pane.add(top_row_pane, weight=1)

        # --- Section 1: Patient Information Panel ---
        patient_details_frame = ttk.LabelFrame(top_row_pane, text="1. Patient Information", padding=SECTION_PADDING)
        top_row_pane.add(patient_details_frame, weight=1)
        
        # Photo display
        photo_frame = ttk.Frame(patient_details_frame)
        photo_frame.pack(anchor=tk.W, pady=5)
        self.patient_photo = Canvas(photo_frame, width=80, height=80, bg='lightgray')
        self.patient_photo.pack(side=tk.LEFT, padx=5)
        self.patient_photo.create_text(40, 40, text="Patient\nPhoto", fill="black")
        
        # Patient info fields
        info_frame = ttk.Frame(patient_details_frame)
        info_frame.pack(anchor=tk.W, fill=tk.X, expand=True)
        
        # Name and dropdowns
        name_frame = ttk.Frame(info_frame)
        name_frame.pack(anchor=tk.W, fill=tk.X)
        ttk.Label(name_frame, text="Name:", font=DEFAULT_FONT).pack(side=tk.LEFT)
        self.patient_name_var = tk.StringVar()
        self.patient_name_entry = ttk.Entry(name_frame, textvariable=self.patient_name_var, width=20)
        self.patient_name_entry.pack(side=tk.LEFT, padx=5)
        
        # Birthday dropdown
        dob_frame = ttk.Frame(info_frame)
        dob_frame.pack(anchor=tk.W, fill=tk.X)
        ttk.Label(dob_frame, text="Birthday:", font=DEFAULT_FONT).pack(side=tk.LEFT)
        self.dob_day = ttk.Combobox(dob_frame, values=list(range(1,32)), width=3, state='readonly')
        self.dob_day.pack(side=tk.LEFT, padx=2)
        self.dob_month = ttk.Combobox(dob_frame, values=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], width=4, state='readonly')
        self.dob_month.pack(side=tk.LEFT, padx=2)
        self.dob_year = ttk.Combobox(dob_frame, values=list(range(1900, 2026)), width=5, state='readonly')
        self.dob_year.pack(side=tk.LEFT, padx=2)
        
        # Symptoms year
        symptoms_frame = ttk.Frame(info_frame)
        symptoms_frame.pack(anchor=tk.W, fill=tk.X)
        ttk.Label(symptoms_frame, text="Symptoms Started:", font=DEFAULT_FONT).pack(side=tk.LEFT)
        self.symptoms_year = ttk.Combobox(symptoms_frame, values=list(range(1900, 2026)), width=5, state='readonly')
        self.symptoms_year.pack(side=tk.LEFT, padx=5)
        
        # Contact info
        contact_frame = ttk.Frame(info_frame)
        contact_frame.pack(anchor=tk.W, fill=tk.X)
        ttk.Label(contact_frame, text="Contact:", font=DEFAULT_FONT).pack(side=tk.LEFT)
        self.contact_entry = ttk.Entry(contact_frame, width=25)
        self.contact_entry.pack(side=tk.LEFT, padx=5)
        
        # Ethnicity dropdown
        ethnicity_frame = ttk.Frame(info_frame)
        ethnicity_frame.pack(anchor=tk.W, fill=tk.X)
        ttk.Label(ethnicity_frame, text="Ethnicity:", font=DEFAULT_FONT).pack(side=tk.LEFT)
        self.ethnicity_var = tk.StringVar()
        self.ethnicity_dropdown = ttk.Combobox(ethnicity_frame,
            values=['Caucasian', 'African', 'Asian', 'Hispanic', 'Other'],
            textvariable=self.ethnicity_var,
            state='readonly',
            width=15)
        self.ethnicity_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Session info
        session_frame = ttk.Frame(info_frame)
        session_frame.pack(anchor=tk.W, fill=tk.X)
        self.pat_info_session_dur = ttk.Label(session_frame, text="Session Duration: 00:00", font=DEFAULT_FONT)
        self.pat_info_session_dur.pack(side=tk.LEFT, padx=5)
        self.pat_info_total_sessions = ttk.Label(session_frame, text="Total Sessions: -", font=DEFAULT_FONT)
        self.pat_info_total_sessions.pack(side=tk.LEFT, padx=5)

        # --- Section 2: Primary Ocular Biomarker Visualization ---
        ocular_frame = ttk.LabelFrame(top_row_pane, text="2. Ocular Biomarkers", padding=SECTION_PADDING)
        top_row_pane.add(ocular_frame, weight=2) # Give more space
        
        # Video Feed and Controls
        video_control_frame = ttk.Frame(ocular_frame)
        video_control_frame.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=5)
        
        # Video display
        self.video_label = ttk.Label(video_control_frame, background='black')
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Webcam toggle button
        self.webcam_toggle = ttk.Button(video_control_frame, text="Toggle Mesh View",
                                      command=self.toggle_mesh_view)
        self.webcam_toggle.pack(pady=5)
        self.show_mesh = True  # Default to showing mesh
        
        # Metrics Display Area
        ocular_metrics_frame = ttk.Frame(ocular_frame)
        ocular_metrics_frame.pack(fill=tk.Y, side=tk.LEFT, padx=5)
        
        # Saccade velocity with threshold indicator
        saccade_frame = ttk.Frame(ocular_metrics_frame)
        saccade_frame.pack(anchor=tk.W, pady=2)
        ttk.Label(saccade_frame, text="Saccade Vel (°/s):", font=DEFAULT_FONT).pack(side=tk.LEFT)
        self.oc_saccade_vel = ttk.Label(saccade_frame, text="-", font=DEFAULT_FONT, foreground='black')
        self.oc_saccade_vel.pack(side=tk.LEFT)
        self.saccade_thresh_ind = Canvas(saccade_frame, width=20, height=20, bg='white', highlightthickness=0)
        self.saccade_thresh_ind.pack(side=tk.LEFT, padx=5)
        self.saccade_thresh_ind.create_oval(2, 2, 18, 18, fill='grey', outline='')
        
        # Fixation stability
        self.oc_fixation_stab = ttk.Label(ocular_metrics_frame, text="Fixation Stab (°): -", font=DEFAULT_FONT)
        self.oc_fixation_stab.pack(anchor=tk.W, pady=2)
        
        # Blink rate with normal range indicator
        blink_frame = ttk.Frame(ocular_metrics_frame)
        blink_frame.pack(anchor=tk.W, pady=2)
        ttk.Label(blink_frame, text="Blink Rate (bpm):", font=DEFAULT_FONT).pack(side=tk.LEFT)
        self.oc_blink_rate = ttk.Label(blink_frame, text="-", font=DEFAULT_FONT)
        self.oc_blink_rate.pack(side=tk.LEFT)
        self.blink_range_ind = ttk.Label(blink_frame, text="(Normal: 10-20)", font=("Arial", 8), foreground='grey')
        self.blink_range_ind.pack(side=tk.LEFT, padx=5)
        
        # Anti-saccade performance
        self.oc_anti_saccade = ttk.Label(ocular_metrics_frame, text="Anti-Saccade Err: -", font=DEFAULT_FONT)
        self.oc_anti_saccade.pack(anchor=tk.W, pady=2)
        
        # Fixation heatmap
        heatmap_frame = ttk.LabelFrame(ocular_metrics_frame, text="Fixation Heatmap", padding=2)
        heatmap_frame.pack(pady=5)
        self.fixation_heatmap_canvas = Canvas(heatmap_frame, bg='black', width=120, height=120)
        self.fixation_heatmap_canvas.pack()
        
        # --- Section 7: Technical Performance Indicators --- (Placing near video)
        tech_frame = ttk.LabelFrame(ocular_frame, text="7. Technical Indicators", padding=SECTION_PADDING)
        tech_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        self.tech_tracking_qual = ttk.Label(tech_frame, text="Tracking Quality: -", font=DEFAULT_FONT)
        self.tech_tracking_qual.pack(anchor=tk.W)
        self.tech_distance = ttk.Label(tech_frame, text="Distance (cm): -", font=DEFAULT_FONT)
        self.tech_distance.pack(anchor=tk.W)
        self.tech_calibration = ttk.Label(tech_frame, text="Calibration: N/A", font=DEFAULT_FONT)
        self.tech_calibration.pack(anchor=tk.W)

        # --- Bottom Row in Content Area ---
        bottom_row_pane = ttk.PanedWindow(content_pane, orient=tk.HORIZONTAL)
        content_pane.add(bottom_row_pane, weight=1)

        # --- Section 3: Genomic Analysis Section ---
        genomic_frame = ttk.LabelFrame(bottom_row_pane, text="3. Genomic Analysis", padding=SECTION_PADDING)
        bottom_row_pane.add(genomic_frame, weight=1)
        self.gen_summary = ttk.Label(genomic_frame, text="BioNeMo Summary: -", font=DEFAULT_FONT)
        self.gen_summary.pack(anchor=tk.W)
        self.gen_variants_text = Text(genomic_frame, height=4, width=30, wrap=tk.WORD, state=tk.DISABLED)
        self.gen_variants_text.pack(fill=tk.X, pady=2)
        self.gen_pathway_label = ttk.Label(genomic_frame, text="Affected Pathways: -", font=DEFAULT_FONT)
        self.gen_pathway_label.pack(anchor=tk.W)
        # TODO: Add Pathway Visualization (Canvas Placeholder)
        self.pathway_canvas = Canvas(genomic_frame, bg='grey', width=150, height=80)
        self.pathway_canvas.pack(pady=5)
        self.pathway_canvas.create_text(75, 40, text="Pathway Viz\n(TODO)", fill="white")

        # --- Section 4: Combined Risk Assessment ---
        risk_frame = ttk.LabelFrame(bottom_row_pane, text="4. Combined Risk", padding=SECTION_PADDING)
        bottom_row_pane.add(risk_frame, weight=1)
        self.risk_meter_canvas = Canvas(risk_frame, bg='lightgrey', height=50, width=200)
        self.risk_meter_canvas.pack(pady=5)
        self.risk_classification_label = ttk.Label(risk_frame, text="Risk Class: -", font=("Arial", 12, "bold"))
        self.risk_classification_label.pack()
        self.risk_reason_label = ttk.Label(risk_frame, text="Reason: -", font=DEFAULT_FONT, wraplength=180)
        self.risk_reason_label.pack()
        self.risk_confidence_label = ttk.Label(risk_frame, text="Confidence: -", font=DEFAULT_FONT)
        self.risk_confidence_label.pack()

        # --- Section 5: Longitudinal Tracking ---
        longitudinal_frame = ttk.LabelFrame(bottom_row_pane, text="5. Longitudinal Tracking", padding=SECTION_PADDING)
        bottom_row_pane.add(longitudinal_frame, weight=2) # Give more space
        # TODO: Add Trend Visualization (Canvas Placeholder for Matplotlib/Plotly)
        self.trend_canvas = Canvas(longitudinal_frame, bg='grey', width=300, height=150)
        self.trend_canvas.pack(pady=5, fill=tk.BOTH, expand=True)
        self.trend_canvas.create_text(150, 75, text="Trend Graphs (TODO)", fill="white")
        self.long_baseline_comp = ttk.Label(longitudinal_frame, text="Baseline Comparison: -", font=DEFAULT_FONT)
        self.long_baseline_comp.pack(anchor=tk.W)
        self.long_population_comp = ttk.Label(longitudinal_frame, text="Population Comparison: -", font=DEFAULT_FONT)
        self.long_population_comp.pack(anchor=tk.W)

        # --- Section 6: Clinical Decision Support ---
        cds_frame = ttk.LabelFrame(bottom_row_pane, text="6. Clinical Decision Support", padding=SECTION_PADDING)
        bottom_row_pane.add(cds_frame, weight=1)
        self.cds_findings_label = ttk.Label(cds_frame, text="Automated Findings:", font=DEFAULT_FONT)
        self.cds_findings_label.pack(anchor=tk.W)
        self.cds_findings_text = Text(cds_frame, height=4, width=30, wrap=tk.WORD, state=tk.DISABLED, bg='lightyellow')
        self.cds_findings_text.pack(fill=tk.X, pady=2)
        self.cds_actions_label = ttk.Label(cds_frame, text="Suggested Actions: -", font=DEFAULT_FONT)
        self.cds_actions_label.pack(anchor=tk.W)
        self.cds_medication_label = ttk.Label(cds_frame, text="Medication Tracking: -", font=DEFAULT_FONT)
        self.cds_medication_label.pack(anchor=tk.W)

        # --- Section 8: Export and Sharing Options ---
        export_frame = ttk.LabelFrame(self.root, text="8. Export & Sharing", padding=SECTION_PADDING)
        export_frame.pack(fill=tk.X, padx=SECTION_PADDING, pady=SECTION_PADY)
        self.report_button = ttk.Button(export_frame, text="Generate Clinical Report (LLM)", command=self.generate_report, state=tk.DISABLED)
        self.report_button.pack(side=tk.LEFT, padx=5)
        # TODO: Add Data Export Button
        # TODO: Add Sharing Controls Button/Options

    def load_patient(self):
        # Simplified patient loading - replace with a proper selection dialog
        try:
            # List available patients (IDs and names)
            patients = self.storage.list_patients()
            if not patients:
                messagebox.showinfo("No Patients", "No patients found in the database.", parent=self.root)
                return

            # Simple dialog to choose ID (replace with a better list selection)
            patient_id_str = simpledialog.askstring("Load Patient",
                                                    "Enter Patient Internal ID to load:",
                                                    parent=self.root)
            if not patient_id_str: return

            patient_db_id = int(patient_id_str)
            patient_info = self.storage.get_patient_info(patient_db_id)

            if patient_info:
                self.current_patient_id = patient_db_id
                self.current_patient_info = patient_info
                self.update_patient_display()
                self.start_btn.config(state=tk.NORMAL)
                logger.info(f"Loaded patient ID: {patient_db_id} ({patient_info.get('patient_id_str', 'N/A')})")
                # TODO: Load patient history for longitudinal view
            else:
                messagebox.showerror("Error", f"Patient with internal ID {patient_db_id} not found.", parent=self.root)
                self._clear_patient_data()

        except ValueError:
             messagebox.showerror("Error", "Invalid Patient ID entered.", parent=self.root)
        except Exception as e:
             logger.error(f"Error loading patient: {e}", exc_info=True)
             messagebox.showerror("Error", f"An unexpected error occurred: {e}", parent=self.root)

    def update_patient_display(self):
        """Updates the UI elements with current patient data."""
        if not self.current_patient_info:
            self._clear_patient_data()
            return

        p_info = self.current_patient_info
        id_str = p_info.get('patient_id_str', 'N/A')
        ethnicity = p_info.get('ethnicity', 'N/A')
        dob = p_info.get('dob', 'N/A')
        history = p_info.get('medical_history', 'N/A')
        symptom_year = p_info.get('symptom_year', 'N/A')
        contact = p_info.get('contact', 'N/A')
        total_sessions = p_info.get('total_sessions', 'N/A')

        # Top bar info
        self.patient_info_label.config(text=f"Loaded: {id_str} ({ethnicity})")
        
        # Section 1 info
        self.patient_name_var.set(id_str)
        
        # Parse and set DOB if in format "DD-MMM-YYYY"
        if dob and isinstance(dob, str) and len(dob.split('-')) == 3:
            day, month, year = dob.split('-')
            self.dob_day.set(day)
            self.dob_month.set(month)
            self.dob_year.set(year)
        else:
            self.dob_day.set('')
            self.dob_month.set('')
            self.dob_year.set('')
            
        self.symptoms_year.set(symptom_year)
        self.contact_entry.delete(0, tk.END)
        self.contact_entry.insert(0, contact)
        self.ethnicity_var.set(ethnicity)
        self.pat_info_total_sessions.config(text=f"Total Sessions: {total_sessions}")

    def _clear_patient_data(self):
        """Clears patient-specific UI elements."""
        self.current_patient_id = None
        self.current_patient_info = None
        # Top bar
        self.patient_info_label.config(text="No patient selected.")
        self.start_btn.config(state=tk.DISABLED)
        # Section 1
        self.patient_name_var.set("")
        self.dob_day.set('')
        self.dob_month.set('')
        self.dob_year.set('')
        self.symptoms_year.set('')
        self.contact_entry.delete(0, tk.END)
        self.ethnicity_var.set('')
        self.pat_info_total_sessions.config(text="Total Sessions: -")
        # Clear photo
        self.patient_photo.delete("all")
        self.patient_photo.create_text(40, 40, text="Patient\nPhoto", fill="black")
        # TODO: Clear history/longitudinal views

    def start_session(self):
        if self.session_active:
            messagebox.showwarning("Session Active", "A session is already running.", parent=self.root)
            return
        if not self.current_patient_id:
            messagebox.showwarning("No Patient", "Please load a patient first.", parent=self.root)
            return

        logger.info("Starting session...")
        self.session_active = True
        self.session_start_time = time.time()
        self.last_metrics_update = {}
        self.last_genomic_update = {}
        self.last_risk_assessment = (RiskLevel.LOW, "N/A", 0.0, {})

        # Start DB session logging
        try:
            self.current_session_id = self.storage.start_session(self.current_patient_id)
            logger.info(f"Started session ID: {self.current_session_id} for patient ID: {self.current_patient_id}")
        except Exception as e:
            logger.error(f"Failed to start database session: {e}", exc_info=True)
            messagebox.showerror("Database Error", f"Failed to start session logging: {e}", parent=self.root)
            self.session_active = False
            return

        # Start camera thread
        self.video_thread = RTSPCameraStream(source=0).start() # Use webcam 0
        time.sleep(1) # Give camera time to initialize

        if not self.video_thread or not self.video_thread.is_running():
             messagebox.showerror("Camera Error", "Failed to start camera stream.", parent=self.root)
             logger.error("Failed to start camera stream.")
             self.storage.end_session(self.current_session_id, error="Camera failed") # End session with error
             self.session_active = False
             self.current_session_id = None
             return

        # Start processing threads
        self.processing_thread = ProcessingThread(
            self.eye_tracker,
            self.pd_detector,
            self.video_thread.frame_queue,
            self.result_queue,
            self.genomic_queue, # Pass queue for triggering genomic analysis
            debug_mode=False # Start in non-debug mode for clinical view
        ).start()

        self.genomic_thread = GenomicAnalysisThread(
            self.bionemo_assessor, # Use the new assessor class
            self.genomic_queue,
            self.result_queue,
            self.current_patient_info['ethnicity'] if self.current_patient_info else 'Other'
        ).start()

        if self.llm_client:
            self.llm_thread = LLMAnalysisThread(
                self.llm_client,
                self.llm_queue,
                self.result_queue
            ).start()
            logger.info("LLM thread started.")
        else:
             logger.info("LLM client not configured, LLM thread not started.")

        # Update UI
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.report_button.config(state=tk.DISABLED) # Disable report gen during session
        self.session_status_label.config(text=f"Status: Running (Sess: {self.current_session_id})")
        self._clear_displays() # Clear previous results
        logger.info("Session started successfully.")

    def stop_session(self):
        if not self.session_active:
            return

        logger.info("Stopping session...")
        self.session_active = False

        # Stop threads gracefully
        self.stop_session_threads()

        # End DB session logging
        if self.current_session_id:
            try:
                # TODO: Calculate average risk or other summary stats for the session
                session_log_path = self.storage.end_session(
                    self.current_session_id,
                    raw_log_data=self.processing_thread.get_raw_log() if self.processing_thread else []
                )
                logger.info(f"Ended session ID: {self.current_session_id}")
                if session_log_path:
                     messagebox.showinfo("Session Saved", f"Session {self.current_session_id} ended.\nLog saved to:\n{session_log_path}", parent=self.root)
                else:
                     messagebox.showinfo("Session Ended", f"Session {self.current_session_id} ended.", parent=self.root)
            except Exception as e:
                logger.error(f"Failed to end database session {self.current_session_id}: {e}", exc_info=True)
                messagebox.showerror("Database Error", f"Failed to properly end session logging: {e}", parent=self.root)

        # Reset state
        self.current_session_id = None
        self.session_start_time = None

        # Update UI
        self.start_btn.config(state=tk.NORMAL if self.current_patient_id else tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.report_button.config(state=tk.NORMAL if self.llm_client and self.current_patient_id else tk.DISABLED) # Enable report gen after session
        self.session_status_label.config(text="Status: Idle")
        logger.info("Session stopped.")

    def stop_session_threads(self):
        """Safely stops all running threads."""
        # Stop in reverse order of dependency/creation
        if self.llm_thread and self.llm_thread.is_alive():
            logger.debug("Stopping LLM thread...")
            self.llm_thread.stop()
            self.llm_thread.join(timeout=2)
        self.llm_thread = None

        if self.genomic_thread and self.genomic_thread.is_alive():
            logger.debug("Stopping Genomic thread...")
            self.genomic_thread.stop()
            self.genomic_thread.join(timeout=2)
        self.genomic_thread = None

        if self.processing_thread and self.processing_thread.is_alive():
            logger.debug("Stopping Processing thread...")
            self.processing_thread.stop()
            self.processing_thread.join(timeout=3)
        self.processing_thread = None

        if self.video_thread and self.video_thread.is_alive():
            logger.debug("Stopping Video thread...")
            self.video_thread.stop()
            self.video_thread.join(timeout=2)
        self.video_thread = None
        logger.debug("All threads stopped.")

    def check_queues(self):
        """Periodically check queues for results from threads."""
        try:
            while not self.result_queue.empty():
                result_type, data = self.result_queue.get_nowait()

                if result_type == "processed_frame":
                    frame, metrics, risk_assessment_results = data
                    self.update_video_display(frame)
                    self.update_ocular_metrics_display(metrics)
                    self.update_risk_display(risk_assessment_results)
                    self.update_technical_display(metrics)
                    self.last_metrics_update = metrics # Store latest full metrics
                    self.last_risk_assessment = risk_assessment_results

                elif result_type == "genomic_result":
                    self.last_genomic_update = data
                    self.update_genomic_display(data)
                    # Potentially trigger LLM summary after genomic analysis?
                    # self.trigger_llm_analysis()

                elif result_type == "llm_result":
                    analysis_type, analysis_text = data
                    self.update_cds_display(analysis_text, analysis_type) # Update CDS section

                elif result_type == "status_update":
                    thread_name, status_msg = data
                    logger.info(f"Status from {thread_name}: {status_msg}")
                    # Update a general status bar if needed

                elif result_type == "error":
                     thread_name, error_msg = data
                     logger.error(f"Error from {thread_name}: {error_msg}")
                     messagebox.showerror(f"Thread Error ({thread_name})", error_msg, parent=self.root)
                     # self.stop_session() # Consider stopping on critical errors

                self.result_queue.task_done()

        except queue.Empty:
            pass # Normal
        except Exception as e:
            logger.error(f"Error processing result queue: {e}", exc_info=True)

        # Update session duration if active
        if self.session_active and self.session_start_time:
            elapsed = time.time() - self.session_start_time
            elapsed_str = time.strftime('%M:%S', time.gmtime(elapsed))
            self.pat_info_session_dur.config(text=f"Session Duration: {elapsed_str}")

        # Reschedule the check
        self.root.after(50, self.check_queues) # Check every 50ms

    def update_video_display(self, frame):
        """Updates the video label with a new frame."""
        if frame is None: return
        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)

            # Resize to fit label while maintaining aspect ratio
            lbl_w = self.video_label.winfo_width()
            lbl_h = self.video_label.winfo_height()
            if lbl_w <= 1 or lbl_h <= 1: return # Avoid division by zero if label not rendered yet

            img_w, img_h = img_pil.size
            ratio = min(lbl_w / img_w, lbl_h / img_h)
            new_size = (int(img_w * ratio), int(img_h * ratio))

            if new_size[0] > 0 and new_size[1] > 0:
                img_resized = img_pil.resize(new_size, Image.Resampling.LANCZOS)
                self.imgtk = ImageTk.PhotoImage(image=img_resized)
                self.video_label.config(image=self.imgtk)
            else:
                 self.video_label.config(image=None) # Clear if size is invalid

        except Exception as e:
            logger.error(f"Error updating video display: {e}", exc_info=True)
            self.video_label.config(image=None) # Clear image on error

    def update_ocular_metrics_display(self, metrics):
        """Updates labels in Section 2: Ocular Biomarkers."""
        if not metrics: return
        self.oc_saccade_vel.config(text=f"Saccade Vel (°/s): {metrics.get('saccade_velocity_deg_s', 'N/A'):.1f}")
        self.oc_fixation_stab.config(text=f"Fixation Stab (°): {metrics.get('fixation_stability_deg', 'N/A'):.2f}")
        self.oc_blink_rate.config(text=f"Blink Rate (bpm): {metrics.get('blink_rate_bpm', 'N/A'):.1f}")
        # TODO: Update Anti-saccade label when metric is available
        # self.oc_anti_saccade.config(text=f"Anti-Saccade Err: {metrics.get('anti_saccade_error_rate', 'N/A')}")
        # TODO: Update Fixation Stability Heatmap

    def update_risk_display(self, risk_assessment_results):
        """Updates Section 4: Combined Risk Assessment."""
        if not risk_assessment_results: return
        risk_level, reason, ocular_score, factors = risk_assessment_results

        # Update Risk Meter
        draw_risk_meter(self.risk_meter_canvas, ocular_score) # Use ocular score for the meter

        # Update Classification and Reason Labels
        level_str = risk_level.value if isinstance(risk_level, RiskLevel) else str(risk_level)
        self.risk_classification_label.config(text=f"Risk Class: {level_str}")
        self.risk_reason_label.config(text=f"Reason: {reason}")
        # TODO: Update Confidence Metric label

    def update_genomic_display(self, genomic_results):
        """Updates Section 3: Genomic Analysis."""
        if not genomic_results: return
        self.gen_summary.config(text=f"BioNeMo Score: {genomic_results.get('genetic_risk_score', 'N/A'):.2f}x")

        variants = genomic_results.get('variants_detected', [])
        variant_text = f"{len(variants)} variants detected:\n"
        variant_text += "\n".join([f"- {v['gene']}-{v['variant']}" for v in variants[:3]]) # Show top 3
        if len(variants) > 3: variant_text += "\n..."

        self.gen_variants_text.config(state=tk.NORMAL)
        self.gen_variants_text.delete('1.0', tk.END)
        self.gen_variants_text.insert('1.0', variant_text)
        self.gen_variants_text.config(state=tk.DISABLED)

        pathways = genomic_results.get('dominant_pathways', [])
        self.gen_pathway_label.config(text=f"Affected Pathways: {', '.join(pathways)}")
        # TODO: Update Pathway Visualization

    def update_technical_display(self, metrics):
        """Updates Section 7: Technical Performance Indicators."""
        if not metrics: return
        # TODO: Add Tracking Quality metric calculation/display
        self.tech_distance.config(text=f"Distance (cm): {metrics.get('estimated_distance_cm', 'N/A'):.1f}")
        # TODO: Add Calibration Status display

    def update_cds_display(self, text, analysis_type="LLM Analysis"):
        """Updates Section 6: Clinical Decision Support (using LLM results)."""
        self.cds_findings_label.config(text=f"Automated Findings ({analysis_type}):")
        self.cds_findings_text.config(state=tk.NORMAL)
        self.cds_findings_text.delete('1.0', tk.END)
        self.cds_findings_text.insert('1.0', text)
        self.cds_findings_text.config(state=tk.DISABLED)
        # TODO: Extract suggested actions or medication info if LLM provides structured output

    def _clear_displays(self):
        """Clears all dynamic data displays."""
        # Ocular
        self.oc_saccade_vel.config(text="Saccade Vel (°/s): -")
        self.oc_fixation_stab.config(text="Fixation Stab (°): -")
        self.oc_blink_rate.config(text="Blink Rate (bpm): -")
        self.oc_anti_saccade.config(text="Anti-Saccade Err: -")
        # TODO: Clear heatmap canvas
        # Genomic
        self.gen_summary.config(text="BioNeMo Summary: -")
        self.gen_variants_text.config(state=tk.NORMAL); self.gen_variants_text.delete('1.0', tk.END); self.gen_variants_text.config(state=tk.DISABLED)
        self.gen_pathway_label.config(text="Affected Pathways: -")
        # TODO: Clear pathway canvas
        # Risk
        draw_risk_meter(self.risk_meter_canvas, 0.0) # Reset meter
        self.risk_classification_label.config(text="Risk Class: -")
        self.risk_reason_label.config(text="Reason: -")
        self.risk_confidence_label.config(text="Confidence: -")
        # Longitudinal
        # TODO: Clear trend canvas
        self.long_baseline_comp.config(text="Baseline Comparison: -")
        self.long_population_comp.config(text="Population Comparison: -")
        # CDS
        self.cds_findings_label.config(text="Automated Findings:")
        self.cds_findings_text.config(state=tk.NORMAL); self.cds_findings_text.delete('1.0', tk.END); self.cds_findings_text.config(state=tk.DISABLED)
        self.cds_actions_label.config(text="Suggested Actions: -")
        self.cds_medication_label.config(text="Medication Tracking: -")
        # Technical
        self.tech_tracking_qual.config(text="Tracking Quality: -")
        self.tech_distance.config(text="Distance (cm): -")
        self.tech_calibration.config(text="Calibration: N/A")
        # Video
        self.video_label.config(image=None) # Clear video feed

    def generate_report(self):
        """Triggers LLM to generate a clinical report based on last session data."""
        if not self.llm_client:
            messagebox.showwarning("LLM Not Configured", "OpenRouter API key not found. Cannot generate report.", parent=self.root)
            return
        if not self.current_patient_id:
             messagebox.showwarning("No Patient", "Load a patient before generating a report.", parent=self.root)
             return

        # TODO: Load data from the *last completed* session for this patient
        # For now, use the data stored from the last active session if available
        if not self.last_metrics_update and not self.last_genomic_update:
             messagebox.showwarning("No Data", "No session data available to generate a report.", parent=self.root)
             return

        logger.info("Triggering LLM for Clinical Report Generation...")
        # Combine available data
        report_data = {
            "patient_info": self.current_patient_info,
            "ocular_metrics": self.last_metrics_update,
            "risk_assessment": self.last_risk_assessment,
            "genomic_analysis": self.last_genomic_update,
        }
        # Put data onto the LLM queue with a specific type
        self.llm_queue.put(("generate_report", report_data))
        self.update_cds_display("Generating clinical report...", "Status")


    def _on_close(self):
        """Handles window closing event."""
        logger.info("Close button clicked.")
        if self.session_active:
            if messagebox.askyesno("Session Active", "A session is currently running. Stop session and exit?", parent=self.root):
                self.stop_session()
                self.root.destroy()
            else:
                return # Don't close if user cancels
        else:
            # Stop threads just in case (should be stopped already)
            self.stop_session_threads()
            self.root.destroy()

    def toggle_mesh_view(self):
        """Toggles between normal webcam view and mesh overlay view."""
        self.show_mesh = not self.show_mesh
        if self.processing_thread:
            self.processing_thread.toggle_mesh(self.show_mesh)
        self.webcam_toggle.config(text="Toggle Normal View" if self.show_mesh else "Toggle Mesh View")

# --- Main Application Entry Point ---
if __name__ == '__main__':
    root = tk.Tk()
    app = Dashboard(root)
    root.mainloop()
    logger.info("Application exited.")
