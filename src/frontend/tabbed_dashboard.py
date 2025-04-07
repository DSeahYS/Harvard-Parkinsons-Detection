import tkinter as tk
from tkinter import messagebox, simpledialog, ttk, Canvas, Text, IntVar, StringVar
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
        self.root.title("GenomeGuard PD Detection System")
        self.root.geometry(f"{self.MIN_WIDTH}x{self.MIN_HEIGHT}")
        self.root.minsize(self.MIN_WIDTH, self.MIN_HEIGHT)

        # Configure styles
        self.style = ttk.Style()
        self.style.configure('.', font=(self.FONT_FAMILY, self.FONT_SIZES['body']))
        self.style.configure('TNotebook.Tab', font=(self.FONT_FAMILY, self.FONT_SIZES['subheader']))
        self.style.configure('Header.TLabel', font=(self.FONT_FAMILY, self.FONT_SIZES['header'], 'bold'))
        
        # --- Header Bar ---
        header_frame = ttk.Frame(self.root, padding=5)
        header_frame.pack(fill=tk.X)
        
        ttk.Label(header_frame, text="GenomeGuard", style='Header.TLabel', 
                 foreground=self.COLOR_PRIMARY).pack(side=tk.LEFT)
        
        self.user_label = ttk.Label(header_frame, text="User: Clinician")
        self.user_label.pack(side=tk.RIGHT, padx=10)
        
        # --- Main Content Area ---
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create Notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tab frames
        self.tab_frames = {
            'patient': ttk.Frame(self.notebook),
            'ocular': ttk.Frame(self.notebook),
            'genomic': ttk.Frame(self.notebook),
            'risk': ttk.Frame(self.notebook),
            'longitudinal': ttk.Frame(self.notebook),
            'clinical': ttk.Frame(self.notebook)
        }
        
        # Add tabs to notebook
        self.notebook.add(self.tab_frames['patient'], text="1. Patient Info")
        self.notebook.add(self.tab_frames['ocular'], text="2. Ocular & Technical")
        self.notebook.add(self.tab_frames['genomic'], text="3. Genomic Analysis")
        self.notebook.add(self.tab_frames['risk'], text="4. Risk Assessment")
        self.notebook.add(self.tab_frames['longitudinal'], text="5. Longitudinal")
        self.notebook.add(self.tab_frames['clinical'], text="6. Clinical & Export")
        
        # --- Session Control Panel (Fixed at bottom) ---
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X)
        
        # Patient management
        patient_ctrl = ttk.Frame(control_frame)
        patient_ctrl.pack(side=tk.LEFT, padx=5)
        ttk.Button(patient_ctrl, text="Load Patient", command=self.load_patient).pack(side=tk.LEFT)
        self.patient_info_label = ttk.Label(patient_ctrl, text="No patient selected")
        self.patient_info_label.pack(side=tk.LEFT, padx=5)
        
        # Session control
        session_ctrl = ttk.Frame(control_frame)
        session_ctrl.pack(side=tk.LEFT, padx=5)
        self.start_btn = ttk.Button(session_ctrl, text="Start Session", command=self.start_session, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(session_ctrl, text="Stop Session", command=self.stop_session, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)
        self.session_status_label = ttk.Label(session_ctrl, text="Status: Idle")
        self.session_status_label.pack(side=tk.LEFT, padx=5)
        
        # Setup each tab's content
        self.setup_patient_tab()
        self.setup_ocular_tab()  # Now includes technical indicators
        self.setup_genomic_tab()
        self.setup_risk_tab()
        self.setup_longitudinal_tab()
        self.setup_clinical_tab()  # Now includes export options

    def setup_patient_tab(self):
        """Setup Patient Information tab"""
        tab = self.tab_frames['patient']
        
        # Photo frame
        photo_frame = ttk.Frame(tab)
        photo_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.patient_photo = Canvas(photo_frame, width=120, height=120, bg='lightgray')
        self.patient_photo.pack()
        self.patient_photo.create_text(60, 60, text="Patient\nPhoto", fill="black")
        
        # Info frame
        info_frame = ttk.Frame(tab)
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add Save and Create buttons at the top
        save_frame = ttk.Frame(info_frame)
        save_frame.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=5)
        ttk.Button(save_frame, text="Save Patient Data", command=self.save_patient_data).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(save_frame, text="Create New Patient", command=self.create_new_patient).pack(side=tk.LEFT)
        
        # Name
        ttk.Label(info_frame, text="Full Name:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.patient_name_var = tk.StringVar()
        ttk.Entry(info_frame, textvariable=self.patient_name_var, width=30).grid(row=1, column=1, sticky=tk.W)
        
        # Birthday
        ttk.Label(info_frame, text="Birthday:").grid(row=2, column=0, sticky=tk.W, pady=5)
        dob_frame = ttk.Frame(info_frame)
        dob_frame.grid(row=2, column=1, sticky=tk.W)
        self.dob_day = ttk.Combobox(dob_frame, values=list(range(1,32)), width=3, state='readonly')
        self.dob_day.pack(side=tk.LEFT)
        self.dob_month = ttk.Combobox(dob_frame, values=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
                                    width=5, state='readonly')
        self.dob_month.pack(side=tk.LEFT)
        self.dob_year = ttk.Combobox(dob_frame, values=list(range(1900, 2026)), width=5, state='readonly')
        self.dob_year.pack(side=tk.LEFT)
        
        # Symptoms year
        ttk.Label(info_frame, text="Symptoms Started:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.symptoms_year = ttk.Combobox(info_frame, values=list(range(1900, 2026)), width=5, state='readonly')
        self.symptoms_year.grid(row=3, column=1, sticky=tk.W)
        
        # Contact
        ttk.Label(info_frame, text="Contact:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.contact_entry = ttk.Entry(info_frame, width=30)
        self.contact_entry.grid(row=4, column=1, sticky=tk.W)
        
        # Ethnicity
        ttk.Label(info_frame, text="Ethnicity:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.ethnicity_var = tk.StringVar()
        self.ethnicity_dropdown = ttk.Combobox(info_frame,
            values=['Chinese', 'Malay', 'Indian', 'Caucasian', 'Other Asian', 'Other'],
            textvariable=self.ethnicity_var,
            state='readonly',
            width=15)
        self.ethnicity_dropdown.grid(row=5, column=1, sticky=tk.W)
        
        # Session info
        ttk.Label(info_frame, text="Current Session:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.pat_info_session_dur = ttk.Label(info_frame, text="00:00")
        self.pat_info_session_dur.grid(row=6, column=1, sticky=tk.W)
        
        ttk.Label(info_frame, text="Total Sessions:").grid(row=7, column=0, sticky=tk.W, pady=5)
        self.pat_info_total_sessions = ttk.Label(info_frame, text="-")
        self.pat_info_total_sessions.grid(row=7, column=1, sticky=tk.W)

    def setup_ocular_tab(self):
        """Setup Ocular Biomarkers tab"""
        tab = self.tab_frames['ocular']
        
        # Video frame
        video_frame = ttk.Frame(tab)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.video_label = ttk.Label(video_frame, background='black')
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        self.webcam_toggle = ttk.Button(video_frame, text="Toggle Mesh View", 
                                      command=self.toggle_mesh_view)
        self.webcam_toggle.pack(pady=5)
        self.show_mesh = True
        
        # Metrics frame
        metrics_frame = ttk.Frame(tab)
        metrics_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        # Saccade velocity
        saccade_frame = ttk.Frame(metrics_frame)
        saccade_frame.pack(fill=tk.X, pady=5)
        ttk.Label(saccade_frame, text="Saccade Velocity:").pack(side=tk.LEFT)
        self.oc_saccade_vel = ttk.Label(saccade_frame, text="-", foreground='black')
        self.oc_saccade_vel.pack(side=tk.LEFT)
        ttk.Label(saccade_frame, text="°/s").pack(side=tk.LEFT)
        self.saccade_thresh_ind = Canvas(saccade_frame, width=20, height=20, bg='white', highlightthickness=0)
        self.saccade_thresh_ind.pack(side=tk.LEFT, padx=5)
        self.saccade_thresh_ind.create_oval(2, 2, 18, 18, fill='grey')
        ttk.Label(saccade_frame, text="(Ref: >400°/s)", font=("Arial", 8)).pack(side=tk.LEFT)
        
        # Fixation stability
        fixation_frame = ttk.Frame(metrics_frame)
        fixation_frame.pack(fill=tk.X, pady=5)
        ttk.Label(fixation_frame, text="Fixation Stability:").pack(side=tk.LEFT)
        self.oc_fixation_stab = ttk.Label(fixation_frame, text="-")
        self.oc_fixation_stab.pack(side=tk.LEFT)
        ttk.Label(fixation_frame, text="°").pack(side=tk.LEFT)
        
        # Blink rate
        blink_frame = ttk.Frame(metrics_frame)
        blink_frame.pack(fill=tk.X, pady=5)
        ttk.Label(blink_frame, text="Blink Rate:").pack(side=tk.LEFT)
        self.oc_blink_rate = ttk.Label(blink_frame, text="-")
        self.oc_blink_rate.pack(side=tk.LEFT)
        ttk.Label(blink_frame, text="bpm").pack(side=tk.LEFT)
        self.blink_range_ind = ttk.Label(blink_frame, text="(Normal: 10-20)", font=("Arial", 8), foreground='grey')
        self.blink_range_ind.pack(side=tk.LEFT, padx=5)
        
        # Anti-saccade
        anti_frame = ttk.Frame(metrics_frame)
        anti_frame.pack(fill=tk.X, pady=5)
        ttk.Label(anti_frame, text="Anti-Saccade Error:").pack(side=tk.LEFT)
        self.oc_anti_saccade = ttk.Label(anti_frame, text="-")
        self.oc_anti_saccade.pack(side=tk.LEFT)
        ttk.Label(anti_frame, text="%").pack(side=tk.LEFT)
        
        # Fixation heatmap
        heatmap_frame = ttk.LabelFrame(metrics_frame, text="Fixation Heatmap")
        heatmap_frame.pack(pady=10)
        self.fixation_heatmap_canvas = Canvas(heatmap_frame, bg='black', width=150, height=150)
        self.fixation_heatmap_canvas.pack()
        
        # Technical indicators section (moved from technical tab)
        tech_frame = ttk.LabelFrame(tab, text="Technical Indicators")
        tech_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        # Quality indicators
        quality_frame = ttk.Frame(tech_frame)
        quality_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Tracking quality
        track_frame = ttk.Frame(quality_frame)
        track_frame.pack(fill=tk.X, pady=5)
        ttk.Label(track_frame, text="Tracking Quality:").pack(side=tk.LEFT)
        self.tech_tracking_qual_var = tk.IntVar(value=0)
        self.tech_tracking_qual = ttk.Progressbar(track_frame, orient=tk.HORIZONTAL,
                                               length=200, mode='determinate',
                                               variable=self.tech_tracking_qual_var)
        self.tech_tracking_qual.pack(side=tk.LEFT, padx=5)
        self.tech_tracking_qual_label = ttk.Label(track_frame, text="0%")
        self.tech_tracking_qual_label.pack(side=tk.LEFT)
        
        # Distance and calibration in one row
        dist_calib_frame = ttk.Frame(quality_frame)
        dist_calib_frame.pack(fill=tk.X, pady=5)
        
        # Distance
        ttk.Label(dist_calib_frame, text="Distance:").pack(side=tk.LEFT)
        self.tech_distance = ttk.Label(dist_calib_frame, text="-")
        self.tech_distance.pack(side=tk.LEFT)
        ttk.Label(dist_calib_frame, text="cm").pack(side=tk.LEFT)
        self.tech_distance_status = ttk.Label(dist_calib_frame, text="(Optimal: 40-60cm)",
                                           foreground='grey')
        self.tech_distance_status.pack(side=tk.LEFT, padx=5)
        
        # Calibration
        ttk.Label(dist_calib_frame, text="  Calibration:").pack(side=tk.LEFT, padx=(20, 0))
        self.tech_calibration = ttk.Label(dist_calib_frame, text="Not Calibrated")
        self.tech_calibration.pack(side=tk.LEFT, padx=5)
        ttk.Button(dist_calib_frame, text="Calibrate", command=self.calibrate_camera).pack(side=tk.LEFT)
        
        # System status in one row
        status_frame = ttk.Frame(quality_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        # Camera
        ttk.Label(status_frame, text="Camera:").pack(side=tk.LEFT)
        self.camera_status = ttk.Label(status_frame, text="Disconnected", foreground=self.COLOR_DANGER)
        self.camera_status.pack(side=tk.LEFT, padx=5)
        
        # Processing
        ttk.Label(status_frame, text="Processing:").pack(side=tk.LEFT, padx=(20, 0))
        self.processing_status = ttk.Label(status_frame, text="Inactive", foreground='grey')
        self.processing_status.pack(side=tk.LEFT, padx=5)
        
        # Database
        ttk.Label(status_frame, text="Database:").pack(side=tk.LEFT, padx=(20, 0))
        self.db_status = ttk.Label(status_frame, text="Connected", foreground=self.COLOR_SAFE)
        self.db_status.pack(side=tk.LEFT, padx=5)

    def setup_genomic_tab(self):
        """Setup Genomic Analysis tab"""
        tab = self.tab_frames['genomic']
        
        # Session selection dropdown
        session_frame = ttk.Frame(tab)
        session_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(session_frame, text="Select Session:").pack(side=tk.LEFT, padx=5)
        self.genomic_session_var = tk.StringVar()
        self.genomic_session_dropdown = ttk.Combobox(session_frame, textvariable=self.genomic_session_var, state="readonly", width=30)
        self.genomic_session_dropdown.pack(side=tk.LEFT, padx=5)
        ttk.Button(session_frame, text="Load", command=self.load_genomic_session).pack(side=tk.LEFT, padx=5)
        
        # Summary frame
        summary_frame = ttk.Frame(tab)
        summary_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Risk score
        score_frame = ttk.Frame(summary_frame)
        score_frame.pack(fill=tk.X, pady=5)
        ttk.Label(score_frame, text="Combined Risk Score:",
                 font=(self.FONT_FAMILY, self.FONT_SIZES['subheader'])).pack(side=tk.LEFT)
        self.gen_risk_score = ttk.Label(score_frame, text="-",
                                      font=(self.FONT_FAMILY, self.FONT_SIZES['subheader'], 'bold'))
        self.gen_risk_score.pack(side=tk.LEFT, padx=5)
        
        # Variants
        variants_frame = ttk.LabelFrame(tab, text="Detected Variants")
        variants_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.gen_variants_text = Text(variants_frame, height=6, width=40, wrap=tk.WORD, state=tk.DISABLED)
        self.gen_variants_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Pathways
        pathways_frame = ttk.LabelFrame(tab, text="Affected Pathways")
        pathways_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Pathway visualization
        self.pathway_canvas = Canvas(pathways_frame, bg='white', height=200)
        self.pathway_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.pathway_canvas.create_text(150, 100, text="Pathway Visualization", fill="gray")
        
    def setup_risk_tab(self):
        """Setup Risk Assessment tab"""
        tab = self.tab_frames['risk']
        
        # Session selection dropdown
        session_frame = ttk.Frame(tab)
        session_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(session_frame, text="Select Session:").pack(side=tk.LEFT, padx=5)
        self.risk_session_var = tk.StringVar()
        self.risk_session_dropdown = ttk.Combobox(session_frame, textvariable=self.risk_session_var, state="readonly", width=30)
        self.risk_session_dropdown.pack(side=tk.LEFT, padx=5)
        ttk.Button(session_frame, text="Load", command=self.load_risk_session).pack(side=tk.LEFT, padx=5)
        
        # Risk meter
        meter_frame = ttk.Frame(tab)
        meter_frame.pack(fill=tk.X, padx=20, pady=20)
        self.risk_meter_canvas = Canvas(meter_frame, bg='white', height=120, width=300)
        self.risk_meter_canvas.pack(pady=10)
        
        # Risk classification
        class_frame = ttk.Frame(tab)
        class_frame.pack(fill=tk.X, padx=20, pady=10)
        ttk.Label(class_frame, text="Risk Classification:",
                 font=(self.FONT_FAMILY, self.FONT_SIZES['subheader'])).pack(anchor=tk.CENTER)
        self.risk_classification_label = ttk.Label(class_frame, text="LOW",
                                                 font=(self.FONT_FAMILY, 24, 'bold'),
                                                 foreground=self.COLOR_SAFE)
        self.risk_classification_label.pack(anchor=tk.CENTER)
        
        # Risk details
        details_frame = ttk.LabelFrame(tab, text="Risk Details")
        details_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Factors
        ttk.Label(details_frame, text="Primary Factors:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.risk_factors_text = Text(details_frame, height=5, width=40, wrap=tk.WORD, state=tk.DISABLED)
        self.risk_factors_text.grid(row=0, column=1, padx=10, pady=5)
        
        # Confidence
        ttk.Label(details_frame, text="Confidence:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.risk_confidence_frame = ttk.Frame(details_frame)
        self.risk_confidence_frame.grid(row=1, column=1, sticky=tk.W, padx=10, pady=5)
        self.risk_confidence_label = ttk.Label(self.risk_confidence_frame, text="85%")
        self.risk_confidence_label.pack(side=tk.LEFT)
        
        # Trend
        ttk.Label(details_frame, text="Trend:").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        self.risk_trend_label = ttk.Label(details_frame, text="Stable")
        self.risk_trend_label.grid(row=2, column=1, sticky=tk.W, padx=10, pady=5)
        
    def setup_longitudinal_tab(self):
        """Setup Longitudinal Tracking tab"""
        tab = self.tab_frames['longitudinal']
        
        # Session selection dropdown
        session_frame = ttk.Frame(tab)
        session_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(session_frame, text="Select Patient:").pack(side=tk.LEFT, padx=5)
        self.longitudinal_patient_var = tk.StringVar()
        self.longitudinal_patient_dropdown = ttk.Combobox(session_frame, textvariable=self.longitudinal_patient_var, state="readonly", width=30)
        self.longitudinal_patient_dropdown.pack(side=tk.LEFT, padx=5)
        ttk.Button(session_frame, text="Load", command=self.load_longitudinal_data).pack(side=tk.LEFT, padx=5)
        
        # Trend graph
        graph_frame = ttk.LabelFrame(tab, text="Trend Over Time")
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Graph canvas
        self.trend_canvas = Canvas(graph_frame, bg='white', height=300)
        self.trend_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.trend_canvas.create_text(150, 150, text="Trend Visualization", fill="gray")
        
        # Time period selector
        period_frame = ttk.Frame(graph_frame)
        period_frame.pack(fill=tk.X, pady=5)
        ttk.Label(period_frame, text="Time Period:").pack(side=tk.LEFT)
        self.period_var = tk.StringVar(value="30d")
        for period in ["7d", "30d", "90d", "1y"]:
            ttk.Radiobutton(period_frame, text=period, variable=self.period_var,
                           value=period).pack(side=tk.LEFT, padx=10)
        
        # Comparisons
        comp_frame = ttk.Frame(tab)
        comp_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Baseline comparison
        baseline_frame = ttk.LabelFrame(comp_frame, text="Baseline Comparison")
        baseline_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.long_baseline_comp = ttk.Label(baseline_frame, text="-",
                                           font=(self.FONT_FAMILY, self.FONT_SIZES['subheader']))
        self.long_baseline_comp.pack(padx=10, pady=10)
        
    def load_longitudinal_data(self):
        """Loads longitudinal data for the selected patient."""
        selected_patient = self.longitudinal_patient_var.get()
        if not selected_patient:
            messagebox.showwarning("No Patient Selected", "Please select a patient to load longitudinal data.", parent=self.root)
            return
            
        try:
            # Extract patient ID from dropdown text (format: "ID: X - Name")
            patient_id = int(selected_patient.split(':')[1].split('-')[0].strip())
            
            # Load all sessions for this patient
            sessions = self.storage.get_patient_sessions(patient_id)
            if not sessions or len(sessions) < 2:
                messagebox.showinfo("Insufficient Data", "At least two sessions are required for longitudinal analysis.", parent=self.root)
                return
                
            # Sort sessions by date
            sessions.sort(key=lambda s: s.get('start_time', ''))
            
            # Extract metrics from each session for trending
            trend_data = []
            for session in sessions:
                session_id = session.get('id')
                session_data = self.storage.get_session_details(session_id)
                if session_data:
                    # Extract key metrics
                    metrics = {
                        'date': session.get('start_time', 'Unknown'),
                        'saccade_velocity': session_data.get('avg_saccade_velocity', 0),
                        'fixation_stability': session_data.get('avg_fixation_stability', 0),
                        'risk_level': session_data.get('avg_risk_level', 0)
                    }
                    trend_data.append(metrics)
            
            # Update trend visualization
            self._update_trend_visualization(trend_data)
            
            # Update comparisons
            if trend_data:
                first_session = trend_data[0]
                last_session = trend_data[-1]
                
                # Calculate change in key metrics
                saccade_change = last_session['saccade_velocity'] - first_session['saccade_velocity']
                risk_change = last_session['risk_level'] - first_session['risk_level']
                
                # Update baseline comparison
                baseline_text = f"Baseline (First Session):\n"
                baseline_text += f"Saccade Velocity: {first_session['saccade_velocity']:.1f} deg/s\n"
                baseline_text += f"Current: {last_session['saccade_velocity']:.1f} deg/s\n"
                baseline_text += f"Change: {saccade_change:+.1f} deg/s"
                self.long_baseline_comp.config(text=baseline_text)
                
                # Update population comparison
                pop_text = f"Population Comparison:\n"
                pop_text += f"Current Risk: {last_session['risk_level']:.2f}\n"
                pop_text += f"Population Avg: 0.50\n"
                pop_text += f"Percentile: {int(last_session['risk_level'] * 100)}%"
                self.long_population_comp.config(text=pop_text)
            
            messagebox.showinfo("Data Loaded", f"Longitudinal data loaded for patient {patient_id}.", parent=self.root)
            
        except Exception as e:
            logger.error(f"Error loading longitudinal data: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to load longitudinal data: {e}", parent=self.root)
    
    def _update_trend_visualization(self, trend_data):
        """Updates the trend visualization with the provided data."""
        self.trend_canvas.delete("all")
        
        if not trend_data or len(trend_data) < 2:
            self.trend_canvas.create_text(150, 150, text="Insufficient data for trend visualization", fill="gray")
            return
            
        # Canvas dimensions
        canvas_width = self.trend_canvas.winfo_width() or 300
        canvas_height = self.trend_canvas.winfo_height() or 200
        
        # Margins
        margin_x = 40
        margin_y = 30
        plot_width = canvas_width - 2 * margin_x
        plot_height = canvas_height - 2 * margin_y
        
        # Draw axes
        self.trend_canvas.create_line(margin_x, canvas_height - margin_y,
                                     canvas_width - margin_x, canvas_height - margin_y,
                                     fill="black", width=2)  # X-axis
        self.trend_canvas.create_line(margin_x, margin_y,
                                     margin_x, canvas_height - margin_y,
                                     fill="black", width=2)  # Y-axis
        
        # Extract saccade velocity data for plotting
        dates = [d['date'] for d in trend_data]
        velocities = [d['saccade_velocity'] for d in trend_data]
        
        # Find min/max for scaling
        min_vel = min(velocities) if velocities else 0
        max_vel = max(velocities) if velocities else 400
        if min_vel == max_vel:  # Avoid division by zero
            min_vel = 0
            max_vel = max_vel * 1.1 if max_vel > 0 else 400
        
        # Plot points and connect with lines
        points = []
        for i, (date, velocity) in enumerate(zip(dates, velocities)):
            # Scale to canvas coordinates
            x = margin_x + (i / (len(dates) - 1 if len(dates) > 1 else 1)) * plot_width
            y = canvas_height - margin_y - ((velocity - min_vel) / (max_vel - min_vel)) * plot_height
            
            # Add point
            points.append((x, y))
            self.trend_canvas.create_oval(x-3, y-3, x+3, y+3, fill=self.COLOR_PRIMARY, outline="")
            
            # Add date label
            if isinstance(date, str) and len(date) > 10:
                date_label = date[:10]  # Just show the date part
            else:
                date_label = str(date)
            self.trend_canvas.create_text(x, canvas_height - margin_y + 15, text=date_label, fill="black", angle=45)
        
        # Connect points with lines
        if len(points) > 1:
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                self.trend_canvas.create_line(x1, y1, x2, y2, fill=self.COLOR_PRIMARY, width=2)
        
        # Add axis labels
        self.trend_canvas.create_text(canvas_width // 2, canvas_height - 10,
                                     text="Session Date", fill="black")
        self.trend_canvas.create_text(15, canvas_height // 2,
                                     text="Saccade Velocity (deg/s)", fill="black", angle=90)
        
        # Population comparison
        pop_frame = ttk.LabelFrame(comp_frame, text="Population Comparison")
        pop_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.long_population_comp = ttk.Label(pop_frame, text="-",
                                            font=(self.FONT_FAMILY, self.FONT_SIZES['subheader']))
        self.long_population_comp.pack(padx=10, pady=10)
        
    def setup_clinical_tab(self):
        """Setup Clinical Decision Support tab"""
        tab = self.tab_frames['clinical']
        
        # Session selection dropdown
        session_frame = ttk.Frame(tab)
        session_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(session_frame, text="Select Session:").pack(side=tk.LEFT, padx=5)
        self.clinical_session_var = tk.StringVar()
        self.clinical_session_dropdown = ttk.Combobox(session_frame, textvariable=self.clinical_session_var, state="readonly", width=30)
        self.clinical_session_dropdown.pack(side=tk.LEFT, padx=5)
        ttk.Button(session_frame, text="Load", command=self.load_clinical_session).pack(side=tk.LEFT, padx=5)
        
        # Findings
        findings_frame = ttk.LabelFrame(tab, text="Automated Findings")
        findings_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.cds_findings_text = Text(findings_frame, height=8, wrap=tk.WORD,
                                    state=tk.DISABLED, bg='lightyellow')
        self.cds_findings_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Actions
        actions_frame = ttk.LabelFrame(tab, text="Suggested Actions")
        actions_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.cds_actions_text = Text(actions_frame, height=6, wrap=tk.WORD,
                                   state=tk.DISABLED, bg='lightblue')
        self.cds_actions_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.cds_actions_label = ttk.Label(actions_frame, text="No actions suggested")
        self.cds_actions_label.pack(anchor=tk.W)
        
        # Medication tracking
        med_frame = ttk.LabelFrame(tab, text="Medication Tracking")
        med_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.cds_medication_canvas = Canvas(med_frame, bg='white', height=100)
        self.cds_medication_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.cds_medication_canvas.create_text(150, 50, text="Medication Timeline\n(No medications tracked)", fill="gray")
        self.cds_medication_label = ttk.Label(med_frame, text="No medication data")
        self.cds_medication_label.pack(anchor=tk.W)
        
        # Export section (moved from export tab)
        export_section = ttk.LabelFrame(tab, text="Export & Sharing")
        export_section.pack(fill=tk.X, padx=10, pady=10)
        
        # Report and export in one row
        report_export_frame = ttk.Frame(export_section)
        report_export_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Report generation
        report_frame = ttk.Frame(report_export_frame)
        report_frame.pack(side=tk.LEFT, padx=10, pady=5)
        self.report_button = ttk.Button(report_frame, text="Generate Clinical Report",
                                     command=self.generate_report, state=tk.DISABLED)
        self.report_button.pack(side=tk.LEFT)
        
        # Export options
        export_frame = ttk.Frame(report_export_frame)
        export_frame.pack(side=tk.LEFT, padx=10, pady=5)
        ttk.Label(export_frame, text="Format:").pack(side=tk.LEFT)
        self.export_format = ttk.Combobox(export_frame, values=["PDF", "CSV", "JSON"], state='readonly', width=5)
        self.export_format.set("PDF")
        self.export_format.pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="Export Data", command=self.export_data).pack(side=tk.LEFT, padx=5)
        
        # Sharing controls
        sharing_frame = ttk.Frame(export_section)
        sharing_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(sharing_frame, text="Email Report", command=self.email_report).pack(side=tk.LEFT, padx=10)
        ttk.Button(sharing_frame, text="EMR Integration", command=self.emr_integration).pack(side=tk.LEFT, padx=10)
        
    def setup_technical_tab(self):
        """Setup Technical Indicators tab"""
        tab = self.tab_frames['technical']
        
        # Quality indicators
        quality_frame = ttk.LabelFrame(tab, text="Signal Quality")
        quality_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Tracking quality
        track_frame = ttk.Frame(quality_frame)
        track_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(track_frame, text="Tracking Quality:").pack(side=tk.LEFT)
        self.tech_tracking_qual_var = tk.IntVar(value=0)
        self.tech_tracking_qual = ttk.Progressbar(track_frame, orient=tk.HORIZONTAL,
                                                length=200, mode='determinate',
                                                variable=self.tech_tracking_qual_var)
        self.tech_tracking_qual.pack(side=tk.LEFT, padx=5)
        self.tech_tracking_qual_label = ttk.Label(track_frame, text="0%")
        self.tech_tracking_qual_label.pack(side=tk.LEFT)
        
        # Distance
        distance_frame = ttk.Frame(quality_frame)
        distance_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(distance_frame, text="Distance:").pack(side=tk.LEFT)
        self.tech_distance = ttk.Label(distance_frame, text="-")
        self.tech_distance.pack(side=tk.LEFT)
        ttk.Label(distance_frame, text="cm").pack(side=tk.LEFT)
        self.tech_distance_status = ttk.Label(distance_frame, text="(Optimal: 40-60cm)",
                                            foreground='grey')
        self.tech_distance_status.pack(side=tk.LEFT, padx=5)
        
        # Calibration
        calib_frame = ttk.Frame(quality_frame)
        calib_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(calib_frame, text="Calibration:").pack(side=tk.LEFT)
        self.tech_calibration = ttk.Label(calib_frame, text="Not Calibrated")
        self.tech_calibration.pack(side=tk.LEFT, padx=5)
        ttk.Button(calib_frame, text="Calibrate", command=self.calibrate_camera).pack(side=tk.LEFT)
        
        # System status
        status_frame = ttk.LabelFrame(tab, text="System Status")
        status_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Camera
        camera_frame = ttk.Frame(status_frame)
        camera_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(camera_frame, text="Camera:").pack(side=tk.LEFT)
        self.camera_status = ttk.Label(camera_frame, text="Disconnected", foreground=self.COLOR_DANGER)
        self.camera_status.pack(side=tk.LEFT, padx=5)
        
        # Processing
        proc_frame = ttk.Frame(status_frame)
        proc_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(proc_frame, text="Processing:").pack(side=tk.LEFT)
        self.processing_status = ttk.Label(proc_frame, text="Inactive", foreground='grey')
        self.processing_status.pack(side=tk.LEFT, padx=5)
        
        # Database
        db_frame = ttk.Frame(status_frame)
        db_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(db_frame, text="Database:").pack(side=tk.LEFT)
        self.db_status = ttk.Label(db_frame, text="Connected", foreground=self.COLOR_SAFE)
        self.db_status.pack(side=tk.LEFT, padx=5)
        
    def setup_export_tab(self):
        """Setup Export & Sharing tab"""
        tab = self.tab_frames['export']
        
        # Report generation
        report_frame = ttk.LabelFrame(tab, text="Report Generation")
        report_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.report_button = ttk.Button(report_frame, text="Generate Clinical Report",
                                      command=self.generate_report, state=tk.DISABLED)
        self.report_button.pack(padx=10, pady=10)
        
        # Export options
        export_frame = ttk.LabelFrame(tab, text="Data Export")
        export_frame.pack(fill=tk.X, padx=10, pady=10)
        
        format_frame = ttk.Frame(export_frame)
        format_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(format_frame, text="Format:").pack(side=tk.LEFT)
        self.export_format = ttk.Combobox(format_frame, values=["PDF", "CSV", "JSON"], state='readonly')
        self.export_format.set("PDF")
        self.export_format.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(export_frame, text="Export Data", command=self.export_data).pack(padx=10, pady=5)
        
        # Sharing controls
        sharing_frame = ttk.LabelFrame(tab, text="Sharing Controls")
        sharing_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(sharing_frame, text="Email Report", command=self.email_report).pack(side=tk.LEFT, padx=10, pady=10)
        ttk.Button(sharing_frame, text="EMR Integration", command=self.emr_integration).pack(side=tk.LEFT, padx=10, pady=10)
    
    def load_patient(self):
        """Load a patient from the database"""
        try:
            # List available patients (IDs and names)
            patients = self.storage.list_patients()
            if not patients:
                messagebox.showinfo("No Patients", "No patients found in the database.", parent=self.root)
                return

            # Create a formatted list of patients for display
            patient_list = [f"{p['id']}: {p['name']} ({p.get('ethnicity', 'N/A')})" for p in patients]
            
            # Show a dialog with the list of patients
            patient_selection = simpledialog.askstring("Load Patient",
                                                     "Enter Patient ID to load:\n\nAvailable patients:\n" +
                                                     "\n".join(patient_list),
                                                     parent=self.root)
            if not patient_selection:
                return
                
            # Extract the ID from the selection
            try:
                patient_db_id = int(patient_selection.split(':')[0].strip()) if ':' in patient_selection else int(patient_selection)
            except ValueError:
                messagebox.showerror("Error", "Invalid Patient ID format. Please enter a number.", parent=self.root)
                return

            # Get patient information
            patient_info = self.storage.get_patient(patient_db_id)

            if patient_info:
                self.current_patient_id = patient_db_id
                self.current_patient_info = patient_info
                self.update_patient_display()
                self.start_btn.config(state=tk.NORMAL)
                logger.info(f"Loaded patient ID: {patient_db_id} ({patient_info.get('name', 'N/A')})")
                # Load patient history for longitudinal view
                self.load_patient_history()
            else:
                messagebox.showerror("Error", f"Patient with ID {patient_db_id} not found.", parent=self.root)
                self._clear_patient_data()

        except ValueError:
             messagebox.showerror("Error", "Invalid Patient ID entered.", parent=self.root)
        except Exception as e:
             logger.error(f"Error loading patient: {e}", exc_info=True)
             messagebox.showerror("Error", f"An unexpected error occurred: {e}", parent=self.root)
    
    def load_patient_history(self):
        """Load patient history for longitudinal tracking and populate session dropdowns"""
        if not self.current_patient_id:
            return
            
        logger.info(f"Loading history for patient {self.current_patient_id}")
        
        try:
            # Get all sessions for this patient
            sessions = self.storage.get_patient_sessions(self.current_patient_id)
            
            if not sessions:
                # Clear dropdowns and set default messages
                self.genomic_session_dropdown['values'] = []
                self.genomic_session_var.set("")
                
                # Only update these if they exist (they might not be created yet)
                if hasattr(self, 'long_baseline_comp'):
                    self.long_baseline_comp.config(text="No baseline data available")
                if hasattr(self, 'long_population_comp'):
                    self.long_population_comp.config(text="No population data available")
                return
                
            # Format session list for dropdowns
            session_list = []
            for session in sessions:
                session_id = session.get('id')
                start_time = session.get('start_time', 'Unknown')
                # Format: "Session ID: X - YYYY-MM-DD HH:MM"
                if isinstance(start_time, str) and len(start_time) > 16:
                    formatted_time = start_time[:16]  # Just take the date and time part
                else:
                    formatted_time = start_time
                session_list.append(f"Session ID: {session_id} - {formatted_time}")
                
            # Update all session dropdowns
            self.genomic_session_dropdown['values'] = session_list
            self.risk_session_dropdown['values'] = session_list
            self.clinical_session_dropdown['values'] = session_list
            
            # Update patient dropdown for longitudinal tab
            patient_list = []
            patients = self.storage.list_patients()
            for patient in patients:
                patient_id = patient.get('id')
                patient_name = patient.get('name', 'Unknown')
                patient_list.append(f"ID: {patient_id} - {patient_name}")
            
            self.longitudinal_patient_dropdown['values'] = patient_list
            
            # Set current patient as default for longitudinal
            if patient_list:
                for patient_item in patient_list:
                    if f"ID: {self.current_patient_id}" in patient_item:
                        self.longitudinal_patient_var.set(patient_item)
                        break
            
            # Set most recent session as default for other dropdowns
            if session_list:
                self.genomic_session_var.set(session_list[0])
                self.risk_session_var.set(session_list[0])
                self.clinical_session_var.set(session_list[0])
                
            # Update longitudinal tab with baseline data
            if len(sessions) > 1:
                if hasattr(self, 'long_baseline_comp'):
                    self.long_baseline_comp.config(text=f"First session: {sessions[0].get('start_time', 'Unknown')}")
                if hasattr(self, 'long_population_comp'):
                    self.long_population_comp.config(text="Population comparison available")
            else:
                if hasattr(self, 'long_baseline_comp'):
                    self.long_baseline_comp.config(text="No baseline data available (only one session)")
                if hasattr(self, 'long_population_comp'):
                    self.long_population_comp.config(text="No population data available")
                
        except Exception as e:
            logger.error(f"Error loading patient history: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to load patient history: {e}", parent=self.root)

    def update_patient_display(self):
        """Updates the UI elements with current patient data."""
        if not self.current_patient_info:
            self._clear_patient_data()
            return

        p_info = self.current_patient_info
        name = p_info.get('name', 'N/A')
        ethnicity = p_info.get('ethnicity', 'N/A')
        dob = p_info.get('dob', 'N/A')
        medical_history = p_info.get('medical_history', {})
        
        # Extract data from medical_history if it's a dictionary
        symptom_year = ''
        contact = ''
        if isinstance(medical_history, dict):
            symptom_year = medical_history.get('symptom_year', '')
            contact = medical_history.get('contact', '')
        
        # Get session count
        sessions = p_info.get('sessions', [])
        total_sessions = len(sessions) if isinstance(sessions, list) else 0

        # Top bar info
        self.patient_info_label.config(text=f"Loaded: {name} ({ethnicity})")
        
        # Section 1 info
        self.patient_name_var.set(name)
        
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
        self.pat_info_total_sessions.config(text=f"{total_sessions}")
        
        # Switch to patient tab to show the loaded data
        self.notebook.select(self.tab_frames['patient'])

    def _clear_patient_data(self):
        """Clears patient-specific UI elements."""
        self.current_patient_id = None
        self.current_patient_info = None
        # Top bar
        self.patient_info_label.config(text="No patient selected")
        self.start_btn.config(state=tk.DISABLED)
        # Section 1
        self.patient_name_var.set("")
        self.dob_day.set('')
        self.dob_month.set('')
        self.dob_year.set('')
        self.symptoms_year.set('')
        self.contact_entry.delete(0, tk.END)
        self.ethnicity_var.set('')
        self.pat_info_total_sessions.config(text="-")
        # Clear photo
        self.patient_photo.delete("all")
        self.patient_photo.create_text(60, 60, text="Patient\nPhoto", fill="black")
        # Clear history/longitudinal views
        self.long_baseline_comp.config(text="-")
        self.long_population_comp.config(text="-")

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

        if not self.video_thread or not self.video_thread.is_running:
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
        
        # Update status indicators
        self.camera_status.config(text="Connected", foreground=self.COLOR_SAFE)
        self.processing_status.config(text="Active", foreground=self.COLOR_SAFE)
        
        # Switch to ocular tab to show the video feed
        self.notebook.select(self.tab_frames['ocular'])
        
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
                # Calculate average risk or other summary stats for the session
                raw_log_data = self.processing_thread.get_raw_log() if self.processing_thread else []
                
                # Create a summary metrics dictionary from the raw log data
                summary_metrics = {}
                if raw_log_data:
                    # Calculate averages from the raw log data
                    saccade_velocities = [entry.get('eye_metrics', {}).get('saccade_velocity_deg_s', 0) for entry in raw_log_data if entry.get('eye_metrics')]
                    fixation_stabilities = [entry.get('eye_metrics', {}).get('fixation_stability_deg', 0) for entry in raw_log_data if entry.get('eye_metrics')]
                    blink_rates = [entry.get('eye_metrics', {}).get('blink_rate_bpm', 0) for entry in raw_log_data if entry.get('eye_metrics')]
                    
                    # Calculate risk levels from pd_risk entries
                    risk_levels = []
                    for entry in raw_log_data:
                        if entry.get('pd_risk') and isinstance(entry['pd_risk'], tuple) and len(entry['pd_risk']) > 0:
                            risk_level = entry['pd_risk'][0]
                            if hasattr(risk_level, 'value'):  # If it's an enum
                                risk_level_str = risk_level.value
                                if risk_level_str == "Low":
                                    risk_levels.append(0.0)
                                elif risk_level_str == "Moderate":
                                    risk_levels.append(0.5)
                                elif risk_level_str == "High":
                                    risk_levels.append(1.0)
                    
                    # Calculate averages
                    summary_metrics['avg_saccade_velocity'] = sum(saccade_velocities) / len(saccade_velocities) if saccade_velocities else 0
                    summary_metrics['avg_fixation_stability'] = sum(fixation_stabilities) / len(fixation_stabilities) if fixation_stabilities else 0
                    summary_metrics['avg_blink_rate'] = sum(blink_rates) / len(blink_rates) if blink_rates else 0
                    summary_metrics['avg_risk_level'] = sum(risk_levels) / len(risk_levels) if risk_levels else 0
                
                # Save raw log data to a JSON file
                json_log_filename = None
                if raw_log_data:
                    try:
                        # Create a filename based on session ID and timestamp
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"session_{self.current_session_id}_{timestamp}.json"
                        filepath = os.path.join(self.config.get('sessions_dir', './data/sessions'), filename)
                        
                        # Ensure directory exists
                        os.makedirs(os.path.dirname(filepath), exist_ok=True)
                        
                        # Convert raw log data to JSON-serializable format
                        serializable_data = []
                        for entry in raw_log_data:
                            serializable_entry = {}
                            for key, value in entry.items():
                                if key == 'pd_risk' and isinstance(value, tuple) and len(value) > 0:
                                    # Handle RiskLevel enum
                                    if hasattr(value[0], 'value'):
                                        serializable_entry[key] = [value[0].value] + list(value[1:])
                                    else:
                                        serializable_entry[key] = list(value)
                                elif isinstance(value, dict):
                                    serializable_entry[key] = value
                                else:
                                    serializable_entry[key] = str(value)
                            serializable_data.append(serializable_entry)
                        
                        # Write to file
                        with open(filepath, 'w') as f:
                            json.dump(serializable_data, f, indent=2)
                        
                        json_log_filename = filename
                    except Exception as e:
                        logger.error(f"Error saving raw log data: {e}")
                
                # End the session with the calculated summary metrics and JSON log filename
                session_log_path = self.storage.end_session(
                    self.current_session_id,
                    summary_metrics=summary_metrics,
                    json_log_filename=json_log_filename
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
        
        # Update status indicators
        self.camera_status.config(text="Disconnected", foreground=self.COLOR_DANGER)
        self.processing_status.config(text="Inactive", foreground='grey')
        
        logger.info("Session stopped.")

    def stop_session_threads(self):
        """Safely stops all running threads."""
        # Stop in reverse order of dependency/creation
        if self.llm_thread:
            logger.debug("Stopping LLM thread...")
            self.llm_thread.stop()
            # LLM thread is a standard threading.Thread, so we can use is_alive()
            if self.llm_thread.is_alive():
                self.llm_thread.join(timeout=2)
        self.llm_thread = None

        if self.genomic_thread:
            logger.debug("Stopping Genomic thread...")
            self.genomic_thread.stop()
            # GenomicAnalysisThread is a standard threading.Thread, so we can use is_alive()
            if self.genomic_thread.is_alive():
                self.genomic_thread.join(timeout=2)
        self.genomic_thread = None

        if self.processing_thread:
            logger.debug("Stopping Processing thread...")
            self.processing_thread.stop()
            # ProcessingThread is a standard threading.Thread, so we can use is_alive()
            if self.processing_thread.is_alive():
                self.processing_thread.join(timeout=3)
        self.processing_thread = None

        if self.video_thread:
            logger.debug("Stopping Video thread...")
            self.video_thread.stop()
            # RTSPCameraStream has its own thread, but doesn't expose is_alive()
            # It has a _thread attribute that we can check
            if hasattr(self.video_thread, '_thread') and self.video_thread._thread and self.video_thread._thread.is_alive():
                self.video_thread._thread.join(timeout=2)
        self.video_thread = None
        logger.debug("All threads stopped.")

    def check_queues(self):
        """Periodically check queues for results from threads."""
        try:
            while not self.result_queue.empty():
                result_type, data = self.result_queue.get_nowait()

                if result_type == "processed_frame":
                    frame, combined_metrics = data
                    self.update_video_display(frame)
                    
                    # Extract eye metrics from combined metrics
                    eye_metrics = combined_metrics.get('eye_metrics', {})
                    self.update_ocular_metrics_display(eye_metrics)
                    
                    # Extract risk assessment results from combined metrics
                    pd_risk_info = combined_metrics.get('pd_risk', None)
                    if pd_risk_info:
                        self.update_risk_display(pd_risk_info)
                        
                    self.update_technical_display(eye_metrics)
                    self.last_metrics_update = eye_metrics # Store latest full metrics
                    self.last_risk_assessment = pd_risk_info

                elif result_type == "genomic_result":
                    self.last_genomic_update = data
                    self.update_genomic_display(data)
                    # Switch to genomic tab to show results
                    self.notebook.select(self.tab_frames['genomic'])

                elif result_type == "llm_result":
                    analysis_type, analysis_text = data
                    self.update_cds_display(analysis_text, analysis_type) # Update CDS section
                    # Switch to clinical tab to show results
                    self.notebook.select(self.tab_frames['clinical'])

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
            self.pat_info_session_dur.config(text=f"{elapsed_str}")

        # Reschedule the check with a shorter interval for smoother video
        self.root.after(10, self.check_queues) # Check every 10ms for smoother video

    def update_video_display(self, frame):
        """Updates the video label with a new frame."""
        if frame is None: return
        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)

            # Use fixed dimensions for the video display
            # This prevents the camera box from growing over time
            fixed_width = 640  # Fixed width for the video display
            fixed_height = 480  # Fixed height for the video display
            
            # Calculate ratio to maintain aspect ratio
            img_w, img_h = img_pil.size
            ratio = min(fixed_width / img_w, fixed_height / img_h)
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
        """Updates the ocular metrics display with new data."""
        if not metrics: return
        
        # Update saccade velocity with threshold coloring
        # Use the correct key 'saccade_velocity_deg_s' from eye_tracker.py
        saccade_vel = metrics.get('saccade_velocity_deg_s', 0)
        # Format to 1 decimal place for better readability
        self.oc_saccade_vel.config(text=f"{saccade_vel:.1f}")
        if saccade_vel < 400:  # Critical PD threshold
            self.oc_saccade_vel.config(foreground=self.COLOR_DANGER)
            self.saccade_thresh_ind.itemconfig(1, fill=self.COLOR_DANGER)
        else:
            self.oc_saccade_vel.config(foreground=self.COLOR_SAFE)
            self.saccade_thresh_ind.itemconfig(1, fill=self.COLOR_SAFE)
            
        # Update fixation stability
        # Use the correct key 'fixation_stability_deg' from eye_tracker.py
        fixation_stability = metrics.get('fixation_stability_deg', 0)
        self.oc_fixation_stab.config(text=f"{fixation_stability:.2f}")
        
        # Update blink rate with normal range coloring
        # Use the correct key 'blink_rate_bpm' from eye_tracker.py
        blink_rate = metrics.get('blink_rate_bpm', 0)
        self.oc_blink_rate.config(text=f"{blink_rate:.1f}")
        if blink_rate < 10 or blink_rate > 20:  # Outside normal range
            self.oc_blink_rate.config(foreground=self.COLOR_ALERT)
        else:
            self.oc_blink_rate.config(foreground=self.COLOR_SAFE)
            
        # Update anti-saccade error rate
        self.oc_anti_saccade.config(text=f"{metrics.get('anti_saccade_error', '-')}")
        
        # Update fixation heatmap visualization
        self._update_fixation_heatmap(metrics)

    def load_risk_session(self):
        """Loads risk assessment data from the selected session."""
        selected_session = self.risk_session_var.get()
        if not selected_session:
            messagebox.showwarning("No Session Selected", "Please select a session to load.", parent=self.root)
            return
            
        try:
            # Extract session ID from dropdown text (format: "Session ID: X - Date")
            session_id = int(selected_session.split(':')[1].split('-')[0].strip())
            
            # Load session data from database
            session_data = self.storage.get_session_details(session_id)
            if not session_data:
                messagebox.showwarning("No Data", f"No data found for session {session_id}.", parent=self.root)
                return
                
            # Extract risk data from session
            risk_data = session_data.get('risk_data', {})
            if not risk_data:
                messagebox.showinfo("No Risk Data", f"No risk assessment data found for session {session_id}.", parent=self.root)
                return
                
            # Update the display with the loaded data
            self.update_risk_display(risk_data)
            messagebox.showinfo("Data Loaded", f"Risk assessment data loaded from session {session_id}.", parent=self.root)
            
        except Exception as e:
            logger.error(f"Error loading risk session data: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to load session data: {e}", parent=self.root)
    
    def update_risk_display(self, risk_assessment_results):
        """Updates the risk assessment display with new data."""
        if not risk_assessment_results: return
        risk_level, reason, ocular_score, factors = risk_assessment_results

        # Update Risk Meter
        draw_risk_meter(self.risk_meter_canvas, ocular_score) # Use ocular score for the meter

        # Update Classification and Reason Labels
        level_str = risk_level.value if isinstance(risk_level, RiskLevel) else str(risk_level)
        self.risk_classification_label.config(text=f"{level_str}")
        
        # Set color based on risk level
        if level_str == "HIGH":
            self.risk_classification_label.config(foreground=self.COLOR_DANGER)
        elif level_str == "MEDIUM":
            self.risk_classification_label.config(foreground=self.COLOR_ALERT)
        else:
            self.risk_classification_label.config(foreground=self.COLOR_SAFE)
            
        # Update risk factors text
        self.risk_factors_text.config(state=tk.NORMAL)
        self.risk_factors_text.delete('1.0', tk.END)
        self.risk_factors_text.insert('1.0', reason)
        self.risk_factors_text.config(state=tk.DISABLED)
        
        # Update confidence
        confidence = factors.get('confidence', 85)
        self.risk_confidence_label.config(text=f"{confidence}%")
        
        # Update trend
        trend = factors.get('trend', 'Stable')
        self.risk_trend_label.config(text=trend)

    def _update_fixation_heatmap(self, metrics):
        """
        Updates the fixation heatmap visualization based on eye metrics.
        
        Args:
            metrics (dict): Eye metrics dictionary containing pupil positions.
        """
        # Clear previous heatmap
        self.fixation_heatmap_canvas.delete("all")
        
        # Get pupil positions from metrics
        pupil_left = metrics.get('pupil_left')
        pupil_right = metrics.get('pupil_right')
        
        # Canvas dimensions
        canvas_width = 150
        canvas_height = 150
        
        # If no pupil data, just draw a grid
        if not pupil_left and not pupil_right:
            # Draw grid lines
            for i in range(0, canvas_width, 30):
                self.fixation_heatmap_canvas.create_line(i, 0, i, canvas_height, fill="gray", width=1)
            for i in range(0, canvas_height, 30):
                self.fixation_heatmap_canvas.create_line(0, i, canvas_width, i, fill="gray", width=1)
            return
        
        # Create a simulated heatmap based on pupil positions
        # In a real implementation, this would use actual gaze points over time
        
        # Draw background grid
        for i in range(0, canvas_width, 30):
            self.fixation_heatmap_canvas.create_line(i, 0, i, canvas_height, fill="gray", width=1)
        for i in range(0, canvas_height, 30):
            self.fixation_heatmap_canvas.create_line(0, i, canvas_width, i, fill="gray", width=1)
        
        # Function to map pupil coordinates to canvas coordinates
        def map_to_canvas(pupil_pos):
            if not pupil_pos:
                return None
            # Assuming pupil_pos is normalized [0,1] coordinates
            x, y = pupil_pos
            canvas_x = int(x * canvas_width)
            canvas_y = int(y * canvas_height)
            return canvas_x, canvas_y
        
        # Draw heatmap points
        if pupil_left:
            left_pos = map_to_canvas(pupil_left)
            if left_pos:
                x, y = left_pos
                # Draw a gradient circle for left eye (blue)
                for r in range(20, 0, -5):
                    # Use standard Tkinter color format without alpha
                    intensity = int(255 * (r/20))
                    color = f"#{intensity:02x}{intensity:02x}ff"
                    self.fixation_heatmap_canvas.create_oval(
                        x-r, y-r, x+r, y+r,
                        fill=color, outline="", tags="heatmap"
                    )
        
        if pupil_right:
            right_pos = map_to_canvas(pupil_right)
            if right_pos:
                x, y = right_pos
                # Draw a gradient circle for right eye (green)
                for r in range(20, 0, -5):
                    # Use standard Tkinter color format without alpha
                    intensity = int(255 * (r/20))
                    color = f"#{intensity:02x}ff{intensity:02x}"
                    self.fixation_heatmap_canvas.create_oval(
                        x-r, y-r, x+r, y+r,
                        fill=color, outline="", tags="heatmap"
                    )
        
        # Draw a crosshair at the center
        center_x = canvas_width // 2
        center_y = canvas_height // 2
        self.fixation_heatmap_canvas.create_line(center_x, 0, center_x, canvas_height, fill="white", width=1)
        self.fixation_heatmap_canvas.create_line(0, center_y, canvas_width, center_y, fill="white", width=1)

    def load_genomic_session(self):
        """Loads genomic data from the selected session."""
        selected_session = self.genomic_session_var.get()
        if not selected_session:
            messagebox.showwarning("No Session Selected", "Please select a session to load.", parent=self.root)
            return
            
        try:
            # Extract session ID from dropdown text (format: "Session ID: X - Date")
            session_id = int(selected_session.split(':')[1].split('-')[0].strip())
            
            # Load session data from database
            session_data = self.storage.get_session_details(session_id)
            if not session_data:
                messagebox.showwarning("No Data", f"No data found for session {session_id}.", parent=self.root)
                return
                
            # Extract genomic data from session
            genomic_data = session_data.get('genomic_data', {})
            if not genomic_data:
                messagebox.showinfo("No Genomic Data", f"No genomic analysis data found for session {session_id}.", parent=self.root)
                return
                
            # Update the display with the loaded data
            self.update_genomic_display(genomic_data)
            messagebox.showinfo("Data Loaded", f"Genomic data loaded from session {session_id}.", parent=self.root)
            
        except Exception as e:
            logger.error(f"Error loading genomic session data: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to load session data: {e}", parent=self.root)
    
    def update_genomic_display(self, genomic_results):
        """Updates the genomic analysis display with new data."""
        if not genomic_results: return
        
        # Update risk score
        risk_score = genomic_results.get('genetic_risk_score', 'N/A')
        if isinstance(risk_score, (int, float)):
            self.gen_risk_score.config(text=f"{risk_score:.2f}x")
        else:
            self.gen_risk_score.config(text=f"{risk_score}")

        # Update variants text
        variants = genomic_results.get('variants_detected', [])
        variant_text = f"{len(variants)} variants detected:\n"
        variant_text += "\n".join([f"- {v['gene']}-{v['variant']}" for v in variants[:3]]) # Show top 3
        if len(variants) > 3: variant_text += "\n..."

        self.gen_variants_text.config(state=tk.NORMAL)
        self.gen_variants_text.delete('1.0', tk.END)
        self.gen_variants_text.insert('1.0', variant_text)
        self.gen_variants_text.config(state=tk.DISABLED)

        # Update pathways visualization
        pathways = genomic_results.get('dominant_pathways', [])
        self.pathway_canvas.delete("all")
        if pathways:
            # Simple visualization of pathways
            y_pos = 30
            for pathway in pathways:
                self.pathway_canvas.create_text(150, y_pos, text=pathway, fill=self.COLOR_PRIMARY)
                y_pos += 30
        else:
            self.pathway_canvas.create_text(150, 100, text="No pathways identified", fill="gray")

    def update_technical_display(self, metrics):
        """Updates the technical indicators display with new data."""
        if not metrics: return
        
        # Update tracking quality
        quality = metrics.get('tracking_quality', 0)
        self.tech_tracking_qual_var.set(int(quality * 100))
        self.tech_tracking_qual_label.config(text=f"{int(quality * 100)}%")
        
        # Update distance with optimal range indicator
        distance = metrics.get('estimated_distance_cm', 0)
        self.tech_distance.config(text=f"{distance:.1f}")
        
        if 40 <= distance <= 60:
            self.tech_distance_status.config(text="(Optimal)", foreground=self.COLOR_SAFE)
        else:
            self.tech_distance_status.config(text="(Outside optimal range)", foreground=self.COLOR_ALERT)

    def load_clinical_session(self):
        """Loads clinical data from the selected session."""
        selected_session = self.clinical_session_var.get()
        if not selected_session:
            messagebox.showwarning("No Session Selected", "Please select a session to load.", parent=self.root)
            return
            
        try:
            # Extract session ID from dropdown text (format: "Session ID: X - Date")
            session_id = int(selected_session.split(':')[1].split('-')[0].strip())
            
            # Load session data from database
            session_data = self.storage.get_session_data(session_id)
            if not session_data:
                messagebox.showwarning("No Data", f"No data found for session {session_id}.", parent=self.root)
                return
                
            # Extract clinical data from session
            clinical_data = session_data.get('clinical_data', {})
            if not clinical_data:
                messagebox.showinfo("No Clinical Data", f"No clinical analysis data found for session {session_id}.", parent=self.root)
                return
                
            # Update the display with the loaded data
            self.update_cds_display(clinical_data.get('findings', 'No findings available.'),
                                   clinical_data.get('type', 'Historical Analysis'))
            
            # Update actions and medications if available
            actions = clinical_data.get('actions', 'No specific actions recommended')
            self.cds_actions_text.config(state=tk.NORMAL)
            self.cds_actions_text.delete('1.0', tk.END)
            self.cds_actions_text.insert('1.0', actions)
            self.cds_actions_text.config(state=tk.DISABLED)
            
            medications = clinical_data.get('medications', 'No medication data available')
            self.cds_medication_label.config(text=medications)
            
            messagebox.showinfo("Data Loaded", f"Clinical data loaded from session {session_id}.", parent=self.root)
            
        except Exception as e:
            logger.error(f"Error loading clinical session data: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to load session data: {e}", parent=self.root)
    
    def update_cds_display(self, text, analysis_type="LLM Analysis"):
        """Updates the clinical decision support display with new data."""
        # Update findings text
        self.cds_findings_text.config(state=tk.NORMAL)
        self.cds_findings_text.delete('1.0', tk.END)
        self.cds_findings_text.insert('1.0', text)
        self.cds_findings_text.config(state=tk.DISABLED)
        
        # Extract actions if available (in a real implementation, this would parse structured data)
        actions = "No specific actions recommended"
        self.cds_actions_text.config(state=tk.NORMAL)
        self.cds_actions_text.delete('1.0', tk.END)
        self.cds_actions_text.insert('1.0', actions)
        self.cds_actions_text.config(state=tk.DISABLED)
        
        # Update medication tracking (placeholder)
        self.cds_medication_label.config(text="No medication data available")

    def _clear_displays(self):
        """Clears all dynamic data displays."""
        # Ocular
        if hasattr(self, 'oc_saccade_vel'):
            self.oc_saccade_vel.config(text="-", foreground='black')
        if hasattr(self, 'saccade_thresh_ind'):
            self.saccade_thresh_ind.itemconfig(1, fill='grey')
        if hasattr(self, 'oc_fixation_stab'):
            self.oc_fixation_stab.config(text="-")
        if hasattr(self, 'oc_blink_rate'):
            self.oc_blink_rate.config(text="-", foreground='black')
        if hasattr(self, 'oc_anti_saccade'):
            self.oc_anti_saccade.config(text="-")
        # Clear heatmap canvas
        if hasattr(self, 'fixation_heatmap_canvas'):
            self.fixation_heatmap_canvas.delete("all")
        
        # Genomic
        if hasattr(self, 'gen_risk_score'):
            self.gen_risk_score.config(text="-")
        if hasattr(self, 'gen_variants_text'):
            self.gen_variants_text.config(state=tk.NORMAL)
            self.gen_variants_text.delete('1.0', tk.END)
            self.gen_variants_text.config(state=tk.DISABLED)
        if hasattr(self, 'pathway_canvas'):
            self.pathway_canvas.delete("all")
            self.pathway_canvas.create_text(150, 100, text="No genomic data", fill="gray")
        
        # Risk
        if hasattr(self, 'risk_meter_canvas'):
            draw_risk_meter(self.risk_meter_canvas, 0.0) # Reset meter
        if hasattr(self, 'risk_classification_label'):
            self.risk_classification_label.config(text="LOW", foreground=self.COLOR_SAFE)
        if hasattr(self, 'risk_factors_text'):
            self.risk_factors_text.config(state=tk.NORMAL)
            self.risk_factors_text.delete('1.0', tk.END)
            self.risk_factors_text.config(state=tk.DISABLED)
        if hasattr(self, 'risk_confidence_label'):
            self.risk_confidence_label.config(text="0%")
        if hasattr(self, 'risk_trend_label'):
            self.risk_trend_label.config(text="N/A")
        
        # Longitudinal
        if hasattr(self, 'trend_canvas'):
            self.trend_canvas.delete("all")
            self.trend_canvas.create_text(150, 150, text="No trend data available", fill="gray")
        if hasattr(self, 'long_baseline_comp'):
            self.long_baseline_comp.config(text="-")
        if hasattr(self, 'long_population_comp'):
            self.long_population_comp.config(text="-")
        
        # CDS
        if hasattr(self, 'cds_findings_text'):
            self.cds_findings_text.config(state=tk.NORMAL)
            self.cds_findings_text.delete('1.0', tk.END)
            self.cds_findings_text.config(state=tk.DISABLED)
        if hasattr(self, 'cds_actions_text'):
            self.cds_actions_text.config(state=tk.NORMAL)
            self.cds_actions_text.delete('1.0', tk.END)
            self.cds_actions_text.config(state=tk.DISABLED)
        if hasattr(self, 'cds_medication_label'):
            self.cds_medication_label.config(text="No medication data")
        
        # Technical
        if hasattr(self, 'tech_tracking_qual_var'):
            self.tech_tracking_qual_var.set(0)
        if hasattr(self, 'tech_tracking_qual_label'):
            self.tech_tracking_qual_label.config(text="0%")
        if hasattr(self, 'tech_distance'):
            self.tech_distance.config(text="-")
        if hasattr(self, 'tech_distance_status'):
            self.tech_distance_status.config(text="(Optimal: 40-60cm)", foreground='grey')
        
        # Video
        if hasattr(self, 'video_label'):
            self.video_label.config(image=None) # Clear video feed

    def generate_report(self):
        """Generates a clinical report based on the current data."""
        if not self.llm_client:
            messagebox.showwarning("LLM Not Configured", "OpenRouter API key not found. Cannot generate report.", parent=self.root)
            return
        if not self.current_patient_id:
             messagebox.showwarning("No Patient", "Load a patient before generating a report.", parent=self.root)
             return

        # Load data from the last completed session for this patient
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
        
        # Switch to clinical tab to show the report generation progress
        self.notebook.select(self.tab_frames['clinical'])

    def calibrate_camera(self):
        """Calibrates the eye tracking camera."""
        if not self.session_active:
            messagebox.showinfo("Not Active", "Start a session before calibrating.", parent=self.root)
            return
            
        # In a real implementation, this would trigger a calibration sequence
        messagebox.showinfo("Calibration", "Camera calibration would start here.\nThis is a placeholder.", parent=self.root)
        self.tech_calibration.config(text="Calibrated")

    def export_data(self):
        """Exports session data in the selected format."""
        if not self.current_patient_id:
            messagebox.showwarning("No Patient", "Load a patient before exporting data.", parent=self.root)
            return
            
        export_format = self.export_format.get()
        messagebox.showinfo("Export", f"Data would be exported in {export_format} format.\nThis is a placeholder.", parent=self.root)

    def email_report(self):
        """Emails the generated report."""
        messagebox.showinfo("Email", "Report would be emailed.\nThis is a placeholder.", parent=self.root)

    def emr_integration(self):
        """Integrates with Electronic Medical Record system."""
        messagebox.showinfo("EMR Integration", "Data would be sent to EMR system.\nThis is a placeholder.", parent=self.root)

    def create_new_patient(self):
        """Creates a new patient record."""
        try:
            # Get basic patient info
            patient_name = simpledialog.askstring("New Patient", "Enter patient name:", parent=self.root)
            if not patient_name:
                return  # User cancelled
                
            # Get ethnicity
            ethnicity = simpledialog.askstring("New Patient", "Enter ethnicity (or leave blank for 'Other'):", parent=self.root)
            if not ethnicity:
                ethnicity = "Other"
                
            # Add patient to database
            patient_id = self.storage.add_patient(
                name=patient_name,
                ethnicity=ethnicity
            )
            
            if patient_id:
                messagebox.showinfo("Success", f"New patient created with ID: {patient_id}", parent=self.root)
                
                # Get the full patient record
                patient_info = self.storage.get_patient(patient_id)
                
                # Load the new patient
                self.current_patient_id = patient_id
                self.current_patient_info = patient_info
                self.update_patient_display()
                self.start_btn.config(state=tk.NORMAL)
                logger.info(f"Created new patient ID: {patient_id} ({patient_name})")
            else:
                messagebox.showerror("Error", "Failed to create new patient.", parent=self.root)
                
        except Exception as e:
            logger.error(f"Error creating new patient: {e}", exc_info=True)
            messagebox.showerror("Error", f"An unexpected error occurred: {e}", parent=self.root)
                
        except Exception as e:
            logger.error(f"Error creating new patient: {e}", exc_info=True)
            messagebox.showerror("Error", f"An unexpected error occurred: {e}", parent=self.root)
    
    def save_patient_data(self):
        """Saves the patient data from the UI to the database."""
        if not self.current_patient_id:
            messagebox.showwarning("No Patient", "Please load a patient first or create a new one.", parent=self.root)
            return
            
        try:
            # Get values from UI
            name = self.patient_name_var.get()
            ethnicity = self.ethnicity_var.get()
            
            # Format DOB if all fields are filled
            dob = None
            if self.dob_day.get() and self.dob_month.get() and self.dob_year.get():
                dob = f"{self.dob_day.get()}-{self.dob_month.get()}-{self.dob_year.get()}"
            
            # Create medical history object
            medical_history = {
                'symptom_year': self.symptoms_year.get(),
                'contact': self.contact_entry.get()
            }
            
            # Update patient in database
            success = self.storage.update_patient(
                self.current_patient_id,
                name=name,
                dob=dob,
                ethnicity=ethnicity,
                medical_history=medical_history
            )
            
            if success:
                messagebox.showinfo("Success", "Patient data saved successfully.", parent=self.root)
                # Refresh patient info
                patient_info = self.storage.get_patient(self.current_patient_id)
                if patient_info:
                    self.current_patient_info = patient_info
                logger.info(f"Updated patient ID: {self.current_patient_id}")
            else:
                messagebox.showerror("Error", "Failed to save patient data.", parent=self.root)
                
        except Exception as e:
            logger.error(f"Error saving patient data: {e}", exc_info=True)
            messagebox.showerror("Error", f"An unexpected error occurred: {e}", parent=self.root)
    
    def toggle_mesh_view(self):
        """Toggles between normal webcam view and mesh overlay view."""
        self.show_mesh = not self.show_mesh
        if self.processing_thread:
            self.processing_thread.toggle_mesh(self.show_mesh)
        self.webcam_toggle.config(text="Toggle Normal View" if self.show_mesh else "Toggle Mesh View")

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

# --- Main Application Entry Point ---
if __name__ == '__main__':
    root = tk.Tk()
    app = Dashboard(root)
    root.mainloop()
    logger.info("Application exited.")
