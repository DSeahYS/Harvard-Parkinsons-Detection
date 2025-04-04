import sqlite3
import json
import os
import time
import numpy as np
from datetime import datetime

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)

class DataStorage:
    def __init__(self, db_path="data/patients.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id TEXT PRIMARY KEY,
            name TEXT,
            age INTEGER,
            gender TEXT,
            medical_history TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            patient_id TEXT,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            risk_level REAL,
            FOREIGN KEY (patient_id) REFERENCES patients (id)
        )
        ''')
        
        conn.commit()
        conn.close()

    def get_all_patients(self):
        """Retrieve all patient records from database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Convert rows to dict-like objects
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM patients ORDER BY name") # Order by name as suggested
        # Convert Row objects to standard dicts for consistency
        patients = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return patients

    def get_patient(self, patient_id):
        """Get single patient by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
        patient = cursor.fetchone()
        conn.close()
        return dict(patient) if patient else None

    def save_patient(self, patient_data):
        """Save patient profile to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if 'id' not in patient_data:
            patient_data['id'] = f"PT{int(time.time())}"
        
        cursor.execute(
            "INSERT OR REPLACE INTO patients (id, name, age, gender, medical_history) VALUES (?, ?, ?, ?, ?)",
            (
                patient_data['id'],
                patient_data.get('name', ''),
                patient_data.get('age', 0),
                patient_data.get('gender', ''),
                patient_data.get('medical_history', '')
            )
        )
        
        conn.commit()
        conn.close()
        return patient_data['id']

    def save_session(self, patient_id, metrics_history, risk_level=0):
        """Save eye tracking session data with numpy array handling"""
        session_id = f"S{int(time.time())}"
        
        # Save session metadata to SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO sessions (session_id, patient_id, start_time, end_time, risk_level) VALUES (?, ?, ?, ?, ?)",
            (
                session_id,
                patient_id,
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                risk_level
            )
        )
        
        conn.commit()
        conn.close()
        
        # Save metrics data to JSON file
        os.makedirs(f"data/sessions/{patient_id}", exist_ok=True)
        session_file = f"data/sessions/{patient_id}/{session_id}.json"
        
        with open(session_file, 'w') as f:
            json.dump({
                'session_id': session_id,
                'patient_id': patient_id,
                'metrics': metrics_history,
                'timestamp': datetime.now().isoformat(),
                'risk_level': risk_level
            }, f, cls=NumpyEncoder, indent=2)
        
        return session_id

    def get_patient_sessions(self, patient_id):
        """Retrieve all sessions for a patient"""
        sessions = []
        session_dir = f"data/sessions/{patient_id}"
        
        if os.path.exists(session_dir):
            for filename in os.listdir(session_dir):
                if filename.endswith('.json'):
                    with open(os.path.join(session_dir, filename)) as f:
                        sessions.append(json.load(f))
        
        return sorted(sessions, key=lambda x: x['timestamp'], reverse=True)
