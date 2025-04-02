# src/frontend/dashboard.py
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
import time
from src.utils.visualization import create_metrics_visualization, create_risk_meter

class Dashboard:
    def __init__(self, eye_tracker, pd_detector, metrics_history, window_title="GenomicGuard: Early Parkinson's Detection"):
        # Store components
        self.eye_tracker = eye_tracker
        self.pd_detector = pd_detector
        self.metrics_history = metrics_history

        # Create main window
        self.window = tk.Tk()
        self.window.title(window_title)
        self.window.minsize(1200, 700)

        # Create tab control
        self.tab_control = ttk.Notebook(self.window)

        # Eye tracking tab
        self.eye_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.eye_tab, text="Eye Tracking")
        self.setup_eye_tracking_tab(self.eye_tab)

        # Genomic analysis tab
        self.genomic_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.genomic_tab, text="Genomic Analysis")
        self.setup_genomic_tab(self.genomic_tab)

        # Patient analysis tab
        self.patient_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.patient_tab, text="Patient Analysis")
        self.setup_patient_tab(self.patient_tab)

        # Pack the tab control
        self.tab_control.pack(expand=1, fill="both")

        # Variables
        self.running = False
        self.cap = None
        self.after_id = None

    def setup_eye_tracking_tab(self, parent):
        # Create frames
        self.left_frame = ttk.Frame(parent, padding=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_frame = ttk.Frame(parent, padding=10)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create canvas for webcam feed
        self.canvas = tk.Canvas(self.left_frame, width=640, height=480)
        self.canvas.pack(padx=10, pady=10)

        # Create controls
        self.controls_frame = ttk.LabelFrame(self.left_frame, text="Controls", padding=10)
        self.controls_frame.pack(fill=tk.X, padx=10, pady=10)

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

    def setup_genomic_tab(self, parent):
        # Create BioNeMo analysis interface
        frame = ttk.Frame(parent, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # Left column - gene selection
        left_frame = ttk.Frame(frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Gene selection
        gene_frame = ttk.LabelFrame(left_frame, text="PD-Associated Genes")
        gene_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.gene_listbox = tk.Listbox(gene_frame, height=10, selectmode=tk.MULTIPLE)
        self.gene_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add some genes
        pd_genes = ["LRRK2", "GBA", "SNCA", "PARK2", "PINK1", "DJ1", "ATP13A2", "VPS35", "FBXO7"]
        for gene in pd_genes:
            self.gene_listbox.insert(tk.END, gene)

        # Select default genes
        self.gene_listbox.selection_set(0, 2)  # Select first three genes

        # Right column - analysis and visualization
        right_frame = ttk.Frame(frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Analysis controls
        control_frame = ttk.Frame(right_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(control_frame, text="Patient ID:").pack(side=tk.LEFT, padx=5)
        self.patient_id_var = tk.StringVar(value="PD12345")
        ttk.Entry(control_frame, textvariable=self.patient_id_var, width=10).pack(side=tk.LEFT, padx=5)

        self.analyze_btn = ttk.Button(control_frame, text="Run BioNeMo Analysis", command=self.run_bionemo_analysis)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)

        # Results area
        results_frame = ttk.LabelFrame(right_frame, text="Genomic Analysis Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.genomic_results_text = scrolledtext.ScrolledText(results_frame, width=50, height=20)
        self.genomic_results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_patient_tab(self, parent):
        """Set up patient analysis tab with LLM integration"""
        # Create patient analysis interface with LLM integration
        frame = ttk.Frame(parent, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # Patient info area
        info_frame = ttk.LabelFrame(frame, text="Patient Information")
        info_frame.pack(fill=tk.X, padx=10, pady=10)

        form = ttk.Frame(info_frame)
        form.pack(padx=10, pady=10, fill=tk.X)

        # Patient fields - name, age, sex
        ttk.Label(form, text="Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.patient_name_var = tk.StringVar(value="John Doe")
        ttk.Entry(form, textvariable=self.patient_name_var, width=20).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(form, text="Age:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.patient_age_var = tk.StringVar(value="65")
        ttk.Entry(form, textvariable=self.patient_age_var, width=5).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(form, text="Sex:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.patient_sex_var = tk.StringVar(value="Male")
        sex_combo = ttk.Combobox(form, textvariable=self.patient_sex_var, values=["Male", "Female", "Other"], width=10)
        sex_combo.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

        # Medical history
        ttk.Label(form, text="Medical History:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.medical_history_var = tk.StringVar(value="Hypertension, family history of tremor")
        ttk.Entry(form, textvariable=self.medical_history_var, width=40).grid(row=3, column=1, columnspan=3, sticky=tk.W, padx=5, pady=5)

        # LLM analysis controls
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Try to get API key from environment first
        env_api_key = os.getenv("OPENROUTER_API_KEY")
        self.api_key_var = tk.StringVar(value=env_api_key if env_api_key else "")
        
        if not env_api_key:
            # Only show API key input if not found in .env
            ttk.Label(control_frame, text="OpenRouter API Key:").pack(side=tk.LEFT, padx=5)
            self.api_key_entry = ttk.Entry(control_frame, textvariable=self.api_key_var, width=30, show="*")
            self.api_key_entry.pack(side=tk.LEFT, padx=5)
        else:
            ttk.Label(control_frame, text="Using API key from .env").pack(side=tk.LEFT, padx=5)

        # Model info label
        ttk.Label(control_frame, text="Model: DeepSeek V3 0324").pack(side=tk.LEFT, padx=5)

        # Analysis button
        self.analyze_patient_btn = ttk.Button(control_frame, text="Generate Clinical Analysis", command=self.run_llm_analysis)
        self.analyze_patient_btn.pack(side=tk.LEFT, padx=5)

        # Progress indicator for API calls
        self.progress_var = tk.StringVar(value="")
        progress_label = ttk.Label(control_frame, textvariable=self.progress_var)
        progress_label.pack(side=tk.LEFT, padx=5)

        # Analysis results
        results_frame = ttk.LabelFrame(frame, text="AI Clinical Assessment")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Use a scrolled text widget to display formatted analysis
        self.analysis_text = tk.Text(results_frame, width=80, height=25, wrap=tk.WORD)
        self.analysis_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.analysis_text, command=self.analysis_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.analysis_text.config(yscrollcommand=scrollbar.set)

        # Default instructions
        self.analysis_text.insert(tk.END, "Enter your OpenRouter API key and click 'Generate Clinical Analysis' to receive an AI-powered clinical assessment based on the eye tracking and genomic data.")

    def run_bionemo_analysis(self):
        """Run BioNeMo genomic analysis"""
        # Import BioNeMo client here to avoid circular imports
        from src.genomic.bionemo_client import BioNeMoClient

        # Clear results area and show loading message
        self.genomic_results_text.delete(1.0, tk.END)
        self.genomic_results_text.insert(tk.END, "Running BioNeMo genomic analysis...\n\n")
        self.window.update_idletasks()

        # Get selected genes
        selected_indices = self.gene_listbox.curselection()
        selected_genes = [self.gene_listbox.get(i) for i in selected_indices]

        # Get eye tracking risk level
        analysis = self.pd_detector.analyze_metrics(None)
        risk_level = analysis.get('risk_level', 0.5)

        # Create BioNeMo client and run analysis
        bionemo = BioNeMoClient()
        results = bionemo.analyze_genomic_data(risk_level)

        # Display results
        self.genomic_results_text.delete(1.0, tk.END)
        self.genomic_results_text.insert(tk.END, f"# BioNeMo Analysis Results\n\n")
        self.genomic_results_text.insert(tk.END, f"Patient ID: {self.patient_id_var.get()}\n")
        self.genomic_results_text.insert(tk.END, f"Analysis Model: {results['model_used']}\n")
        self.genomic_results_text.insert(tk.END, f"Input Eye Risk Level: {risk_level*100:.1f}%\n\n")

        # Display detected variants
        self.genomic_results_text.insert(tk.END, f"## Detected Variants\n\n")

        if results['patient_variants']:
            for gene, data in results['patient_variants'].items():
                self.genomic_results_text.insert(tk.END, f"- {gene}: {data['variant']} (heterozygous)\n")
                self.genomic_results_text.insert(tk.END, f"  - Risk contribution: {data['risk_contribution']:.2f}x\n\n")
        else:
            self.genomic_results_text.insert(tk.END, "No pathogenic variants detected.\n\n")

        # Overall genomic risk score
        self.genomic_results_text.insert(tk.END, f"## Genomic Risk Assessment\n\n")
        self.genomic_results_text.insert(tk.END, f"Overall Genomic Risk Score: {results['genomic_risk_score']*100:.1f}%\n\n")

        # Protein analysis
        self.genomic_results_text.insert(tk.END, f"## Protein Structure Analysis\n\n")
        self.genomic_results_text.insert(tk.END, f"Alpha-synuclein aggregation potential: {(risk_level*0.8+0.1)*100:.1f}%\n")
        self.genomic_results_text.insert(tk.END, f"Dopamine transporter expression: {(1-risk_level*0.6)*100:.1f}% of normal\n\n")

        self.genomic_results_text.insert(tk.END, f"Analysis complete at {time.strftime('%H:%M:%S')}.")

    # Add the LLM Analysis Method and Genomic Data Simulation Method
    def run_llm_analysis(self):
        """Run LLM analysis via OpenRouter with DeepSeek V3"""
        # Import OpenRouter client
        from src.llm.openrouter_client import OpenRouterClient

        # Clear current text
        self.analysis_text.delete(1.0, tk.END)

        # Show loading message
        self.progress_var.set("Generating analysis...")
        self.analysis_text.insert(tk.END, "Generating clinical assessment with DeepSeek V3...\n")
        self.window.update_idletasks()

        # Gather patient info
        patient_info = {
            "name": self.patient_name_var.get(),
            "age": self.patient_age_var.get(),
            "sex": self.patient_sex_var.get(),
            "medical_history": self.medical_history_var.get()
        }

        # Get eye metrics and analysis from current session
        eye_metrics = {}
        # Ensure metrics_history is not empty before accessing the last element
        if self.metrics_history and len(self.metrics_history) > 0:
             # Get the latest available metrics
            latest_metrics = next((m for m in reversed(self.metrics_history) if m), None)
            if latest_metrics:
                 eye_metrics = latest_metrics.copy() # Use a copy

        # Get risk level from PD detector using the latest metrics if available
        analysis_result = self.pd_detector.analyze_metrics(eye_metrics if eye_metrics else None) # Pass latest metrics or None
        # Add risk level to eye_metrics dict for the prompt
        eye_metrics['risk_level'] = analysis_result.get('risk_level', 0.5) # Default if no analysis yet

        # Get genomic data if available
        # For the hackathon, we'll create simulated data based on eye risk
        genomic_data = self._generate_genomic_data(eye_metrics['risk_level'])

        # Create OpenRouter client
        openrouter = OpenRouterClient(api_key=self.api_key_var.get())

        # Get start time to calculate response time
        start_time = time.time()

        # Get analysis from OpenRouter's DeepSeek V3
        result = openrouter.analyze_patient(patient_info, eye_metrics, genomic_data)

        # Calculate response time
        response_time = time.time() - start_time

        # Update progress indicator
        self.progress_var.set(f"Complete ({response_time:.2f}s)")

        # Display the analysis
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, result["analysis"])
        self.analysis_text.insert(tk.END, f"\n\n---\nGenerated using: {result['model_used']} in {response_time:.2f} seconds")

        # Add formatted citation of where information came from
        self.analysis_text.insert(tk.END, "\n\nAnalysis based on:")
        self.analysis_text.insert(tk.END, f"\n- Eye tracking data ({len(self.metrics_history)} samples)")
        self.analysis_text.insert(tk.END, f"\n- Fixation stability: {eye_metrics.get('fixation_stability', 'N/A')}")
        self.analysis_text.insert(tk.END, f"\n- Saccade velocity: {eye_metrics.get('avg_saccade_velocity', 'N/A')}")
        self.analysis_text.insert(tk.END, f"\n- Vertical Saccade velocity: {eye_metrics.get('avg_vertical_saccade_velocity', 'N/A')}") # Added vertical

    def _generate_genomic_data(self, risk_level):
        """Generate simulated genomic data based on risk level"""
        # For hackathon demo, create genomic data that correlates with eye risk
        variants = {}

        # Higher risk = more likely to have PD gene variants
        if risk_level > 0.3:
            variants["LRRK2"] = {
                "variant": "G2019S",
                "risk_contribution": 2.4 * risk_level
            }

        if risk_level > 0.5:
            variants["GBA"] = {
                "variant": "N370S",
                "risk_contribution": 5.4 * risk_level
            }

        if risk_level > 0.7:
            variants["SNCA"] = {
                "variant": "A53T",
                "risk_contribution": 8.1 * risk_level
            }

        # Calculate genomic risk score
        genomic_risk = 0.1  # Baseline risk
        if variants:
            variant_risks = [v["risk_contribution"] for v in variants.values()]
            # Ensure division by non-zero and scale appropriately
            genomic_risk = min(sum(variant_risks) / max(len(variants) * 5, 1), 1.0) # Adjusted scaling


        return {
            "patient_variants": variants,
            "genomic_risk_score": genomic_risk,
            "model_used": "BioNeMo Evo-2 (simulated)",
            "analysis_timestamp": time.time()
        }

    def on_start(self):
        """Start webcam capture"""
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.update_frame()

    def on_stop(self):
        """Stop webcam capture"""
        if self.running:
            if self.after_id:
                self.window.after_cancel(self.after_id)
            self.running = False
            if self.cap:
                self.cap.release()
                self.cap = None

    def update_frame(self):
        """Read frame, process, update UI, and schedule next call."""
        if self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Process frame with eye tracker, passing the debug mode state
                processed_frame, metrics = self.eye_tracker.process_frame(frame, debug_mode=self.debug_mode)

                # Update metrics history
                if metrics:
                    self.metrics_history.append(metrics)

                # Process cycles if enabled
                if self.cycle_var.get():
                    # This would use the CycleBuffer in a real implementation
                    pass

                # Get Parkinson's analysis
                analysis = self.pd_detector.analyze_metrics(metrics)

                # Create visualizations
                risk_meter_img = None
                metrics_vis_img = None

                if analysis.get('analysis_complete', False):
                    risk_level = analysis.get('risk_level', 0.0)
                    risk_meter_img = create_risk_meter(risk_level)

                if len(self.metrics_history) > 10:
                    metrics_vis_img = create_metrics_visualization(list(self.metrics_history))

                # Update dashboard UI elements
                self.update_metrics(metrics)
                self.update_risk_assessment(analysis)
                self.update_visualization(risk_meter_img, metrics_vis_img)

                # Convert PROCESSED frame for display
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(image=img)

                # Update canvas
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                self.canvas.image = img_tk

            # Schedule next frame processing
            self.after_id = self.window.after(33, self.update_frame)

    def update_metrics(self, metrics):
        """Update metrics display"""
        if metrics:
            # Clear existing text
            self.metrics_text.delete(1.0, tk.END)

            # Add new metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.metrics_text.insert(tk.END, f"{key}: {value:.4f}\n")
                else:
                    self.metrics_text.insert(tk.END, f"{key}: {value}\n")

    def update_risk_assessment(self, assessment):
        """Update risk assessment display"""
        if assessment.get('analysis_complete', False):
            risk_level = assessment.get('risk_level', 0.0)
            risk_factors = assessment.get('risk_factors', [])

            # Update risk label
            self.risk_label.config(text=f"Risk Level: {risk_level*100:.1f}%")

            # Clear and update risk factors
            self.factors_text.delete(1.0, tk.END)
            if risk_factors:
                for factor in risk_factors:
                    self.factors_text.insert(tk.END, f"• {factor}\n")
            else:
                self.factors_text.insert(tk.END, "No risk factors detected.")
        else:
            # Still collecting data
            self.risk_label.config(text=f"Risk Level: {assessment.get('message', 'Processing...')}")

    def update_visualization(self, risk_meter_img, metrics_vis_img):
        """Update visualization canvases"""
        if risk_meter_img is not None:
            # Convert to format suitable for tkinter
            img = Image.fromarray(cv2.cvtColor(risk_meter_img, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(image=img)

            # Update risk meter canvas
            self.risk_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.risk_canvas.image = img_tk

        if metrics_vis_img is not None:
            # Convert to format suitable for tkinter
            img = Image.fromarray(cv2.cvtColor(metrics_vis_img, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(image=img)

            # Update visualization canvas
            self.vis_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.vis_canvas.vis_image = img_tk

    def run(self):
        """Run the dashboard"""
        self.window.mainloop()

    def toggle_debug(self):
        """Toggle debug visualization mode"""
        self.debug_mode = not self.debug_mode
        if self.debug_mode:
            self.btn_debug.configure(text="Normal View")
        else:
            self.btn_debug.configure(text="Debug View")

        # Force immediate UI update
        self.window.update_idletasks()

    def close(self):
        """Close the dashboard and release resources"""
        self.on_stop()
        self.window.destroy()
