import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class Dashboard:
    def __init__(self, window_title="GenomicGuard: Early Parkinson's Detection"):
        # Create main window
        self.window = tk.Tk()
        self.window.title(window_title)
        self.window.minsize(1200, 700)
        
        # Create frames
        self.left_frame = ttk.Frame(self.window, padding=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.right_frame = ttk.Frame(self.window, padding=10)
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
        
        # Variables
        self.running = False
        self.cap = None
        self.after_id = None
        
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
        """Update webcam frame on canvas"""
        if self.running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Convert frame to format suitable for tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(image=img)
                
                # Update canvas
                self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                self.canvas.image = img_tk  # Keep reference
            
            # Schedule next update
            self.after_id = self.window.after(10, self.update_frame)
    
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
            self.risk_canvas.image = img_tk  # Keep reference
        
        if metrics_vis_img is not None:
            # Convert to format suitable for tkinter
            img = Image.fromarray(cv2.cvtColor(metrics_vis_img, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(image=img)
            
            # Update visualization canvas
            self.vis_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.vis_canvas.vis_image = img_tk  # Keep reference
    
    def run(self):
        """Run the dashboard"""
        self.window.mainloop()
    
    def close(self):
        """Close the dashboard and release resources"""
        self.on_stop()
        self.window.destroy()
