import sqlite3
import json
import numpy as np
import threading
import datetime
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Numpy Aware JSON Encoder/Decoder ---
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def numpy_decoder(dct):
    """ Custom decoder hook (can be extended if specific structures need reconversion) """
    # This is basic; more complex structures might need specific checks
    # For now, it doesn't automatically convert lists back to arrays,
    # as we don't know which lists *should* be arrays.
    return dct

# --- Storage Manager ---
class StorageManager:
    """
    Manages data storage using SQLite for structured data and JSON for details.
    Ensures thread safety for SQLite connections.
    """
    _instance = None
    _lock = threading.Lock()

    # Use Singleton pattern to ensure single manager instance
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StorageManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        """
        Initializes the StorageManager using central configuration.
        """
        if self._initialized:
            return

        from ..utils.config import get_config
        config = get_config()

        self.db_path = config.abs_path(config.db_path)
        self.sessions_dir = config.abs_path(config.sessions_dir)
        self._local = threading.local() # Thread-local storage for connections

        # Initialize database schema if it doesn't exist
        self._init_db()
        self._initialized = True
        logging.info(f"StorageManager initialized. DB: '{self.db_path}', Sessions Dir: '{self.sessions_dir}'")

    def _get_connection(self):
        """Gets a thread-safe SQLite connection."""
        # Check if connection exists for this thread
        if not hasattr(self._local, 'connection'):
            try:
                # Create a new connection for this thread
                conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10) # Increased timeout
                conn.execute("PRAGMA journal_mode=WAL;") # Enable Write-Ahead Logging
                conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
                self._local.connection = conn
                logging.debug(f"Created new SQLite connection for thread {threading.current_thread().name}")
            except sqlite3.Error as e:
                logging.error(f"Error connecting to database '{self.db_path}': {e}")
                raise
        return self._local.connection

    def close_connection(self):
        """Closes the connection for the current thread."""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            del self._local.connection
            logging.debug(f"Closed SQLite connection for thread {threading.current_thread().name}")

    def _init_db(self):
        """Initializes the database schema if tables don't exist."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Patients Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    dob TEXT, -- Date of Birth (ISO format YYYY-MM-DD)
                    ethnicity TEXT,
                    medical_history TEXT, -- Store as JSON? Or simple text?
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Sessions Table (Summary Data)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    avg_risk_level REAL,
                    -- Add other summary metrics as needed
                    -- e.g., avg_saccade_velocity REAL, avg_fixation_stability REAL
                    json_log_filename TEXT, -- Link to detailed JSON log
                    FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE SET NULL
                )
            ''')

            conn.commit()
            logging.info("Database schema initialized successfully.")
        except sqlite3.Error as e:
            logging.error(f"Error initializing database schema: {e}")
            # No need to close connection here, handled by context or close_connection()
        # No finally block needed to close connection, as _get_connection manages it per thread

    # --- Patient Management ---
    def add_patient(self, name, dob=None, ethnicity=None, medical_history=None):
        """Adds a new patient to the database."""
        sql = '''INSERT INTO patients(name, dob, ethnicity, medical_history)
                 VALUES(?, ?, ?, ?)'''
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(sql, (name, dob, ethnicity, json.dumps(medical_history) if medical_history else None))
            conn.commit()
            patient_id = cursor.lastrowid
            logging.info(f"Added patient '{name}' with ID: {patient_id}")
            return patient_id
        except sqlite3.Error as e:
            logging.error(f"Error adding patient '{name}': {e}")
            return None

    def get_patient(self, patient_id):
        """Retrieves a patient by their ID."""
        sql = "SELECT * FROM patients WHERE id = ?"
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(sql, (patient_id,))
            patient_row = cursor.fetchone()
            if patient_row:
                patient_dict = dict(patient_row)
                if patient_dict.get('medical_history'):
                    try:
                        patient_dict['medical_history'] = json.loads(patient_dict['medical_history'])
                    except json.JSONDecodeError:
                        logging.warning(f"Could not decode medical history JSON for patient {patient_id}")
                        # Keep raw string if decode fails
                return patient_dict
            else:
                return None
        except sqlite3.Error as e:
            logging.error(f"Error getting patient {patient_id}: {e}")
            return None

    def list_patients(self):
        """Lists all patients."""
        sql = "SELECT id, name, dob, ethnicity FROM patients ORDER BY name"
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(sql)
            patients = [dict(row) for row in cursor.fetchall()]
            return patients
        except sqlite3.Error as e:
            logging.error(f"Error listing patients: {e}")
            return []

    def update_patient(self, patient_id, name=None, dob=None, ethnicity=None, medical_history=None):
        """Updates an existing patient's details."""
        fields = []
        params = []
        if name is not None:
            fields.append("name = ?")
            params.append(name)
        if dob is not None:
            fields.append("dob = ?")
            params.append(dob)
        if ethnicity is not None:
            fields.append("ethnicity = ?")
            params.append(ethnicity)
        if medical_history is not None:
            fields.append("medical_history = ?")
            params.append(json.dumps(medical_history))

        if not fields:
            logging.warning("Update patient called with no fields to update.")
            return False

        sql = f"UPDATE patients SET {', '.join(fields)} WHERE id = ?"
        params.append(patient_id)

        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(sql, tuple(params))
            conn.commit()
            logging.info(f"Updated patient ID: {patient_id}")
            return cursor.rowcount > 0 # Return True if a row was updated
        except sqlite3.Error as e:
            logging.error(f"Error updating patient {patient_id}: {e}")
            return False

    # --- Session Management ---
    def start_session(self, patient_id=None):
        """Starts a new session and returns its ID."""
        start_time = datetime.datetime.now().isoformat()
        sql = '''INSERT INTO sessions(patient_id, start_time) VALUES(?, ?)'''
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(sql, (patient_id, start_time))
            conn.commit()
            session_id = cursor.lastrowid
            logging.info(f"Started session ID: {session_id} for patient ID: {patient_id}")
            return session_id
        except sqlite3.Error as e:
            logging.error(f"Error starting session for patient {patient_id}: {e}")
            return None

    def save_session_details(self, session_id, detailed_data):
        """Saves detailed session data (e.g., metrics over time) to a JSON file."""
        filename = f"session_{session_id}_{datetime.datetime.now():%Y%m%d%H%M%S}.json"
        filepath = os.path.join(self.sessions_dir, filename)
        try:
            with open(filepath, 'w') as f:
                json.dump(detailed_data, f, cls=NumpyEncoder, indent=4)
            logging.info(f"Saved detailed session log to '{filepath}'")
            return filename # Return filename to link in DB
        except IOError as e:
            logging.error(f"Error saving session details to JSON '{filepath}': {e}")
            return None
        except TypeError as e:
             logging.error(f"Error serializing session details to JSON: {e}")
             return None


    def end_session(self, session_id, avg_risk_level=None, json_log_filename=None):
        """Ends a session, updating summary data in the database."""
        end_time = datetime.datetime.now().isoformat()
        sql = '''UPDATE sessions SET end_time = ?, avg_risk_level = ?, json_log_filename = ?
                 WHERE id = ?'''
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(sql, (end_time, avg_risk_level, json_log_filename, session_id))
            conn.commit()
            logging.info(f"Ended session ID: {session_id}")
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logging.error(f"Error ending session {session_id}: {e}")
            return False

    def get_session_summary(self, session_id):
        """Retrieves summary data for a specific session."""
        sql = "SELECT * FROM sessions WHERE id = ?"
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(sql, (session_id,))
            session_row = cursor.fetchone()
            return dict(session_row) if session_row else None
        except sqlite3.Error as e:
            logging.error(f"Error getting session summary {session_id}: {e}")
            return None

    def get_session_details(self, session_id):
        """Loads detailed session data from its JSON file."""
        summary = self.get_session_summary(session_id)
        if not summary or not summary.get('json_log_filename'):
            logging.warning(f"No JSON log filename found for session {session_id}")
            return None

        filepath = os.path.join(self.sessions_dir, summary['json_log_filename'])
        try:
            with open(filepath, 'r') as f:
                details = json.load(f, object_hook=numpy_decoder)
            return details
        except FileNotFoundError:
            logging.error(f"Session detail file not found: '{filepath}'")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding session detail JSON '{filepath}': {e}")
            return None
        except IOError as e:
            logging.error(f"Error reading session detail file '{filepath}': {e}")
            return None

    def get_patient_sessions(self, patient_id):
        """Retrieves all session summaries for a given patient."""
        sql = "SELECT * FROM sessions WHERE patient_id = ? ORDER BY start_time DESC"
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(sql, (patient_id,))
            sessions = [dict(row) for row in cursor.fetchall()]
            return sessions
        except sqlite3.Error as e:
            logging.error(f"Error getting sessions for patient {patient_id}: {e}")
            return []

# Example Usage (can be removed or kept for testing)
if __name__ == '__main__':
    storage = StorageManager(db_path="test_genomeguard.db", data_dir="test_data/sessions")

    # Clean up previous test DB if exists
    if os.path.exists("test_genomeguard.db"):
        os.remove("test_genomeguard.db")
    if os.path.exists("test_data/sessions"):
        import shutil
        shutil.rmtree("test_data/sessions")
    storage = StorageManager(db_path="test_genomeguard.db", data_dir="test_data/sessions") # Re-initialize after cleanup


    # --- Test Patient Management ---
    print("\n--- Testing Patient Management ---")
    p_id1 = storage.add_patient("Alice Wonderland", dob="1990-05-15", ethnicity="Caucasian", medical_history={"allergies": ["penicillin"]})
    p_id2 = storage.add_patient("Bob The Builder", dob="1985-11-01", ethnicity="Hispanic")
    print("Patients List:", storage.list_patients())
    alice = storage.get_patient(p_id1)
    print("Get Alice:", alice)
    storage.update_patient(p_id2, name="Robert Builder", medical_history={"conditions": ["hypertension"]})
    print("Get Updated Bob:", storage.get_patient(p_id2))

    # --- Test Session Management ---
    print("\n--- Testing Session Management ---")
    session_id = storage.start_session(patient_id=p_id1)
    print(f"Started session: {session_id}")

    # Simulate some detailed data
    detailed_log = {
        'timestamps': [time.time() + i for i in range(5)],
        'risk_levels': np.random.rand(5).tolist(), # Use tolist() or NumpyEncoder
        'metrics_log': [
            {'saccade': np.random.rand() * 200, 'stability': np.random.rand()},
            {'saccade': np.random.rand() * 200, 'stability': np.random.rand()},
            {'saccade': np.random.rand() * 200, 'stability': np.random.rand()},
            {'saccade': np.random.rand() * 200, 'stability': np.random.rand()},
            {'saccade': np.random.rand() * 200, 'stability': np.random.rand()},
        ]
    }
    json_file = storage.save_session_details(session_id, detailed_log)
    print(f"Saved details to: {json_file}")

    time.sleep(1) # Ensure end time is different
    storage.end_session(session_id, avg_risk_level=np.mean(detailed_log['risk_levels']), json_log_filename=json_file)
    print("Ended session.")

    summary = storage.get_session_summary(session_id)
    print("Session Summary:", summary)
    details_loaded = storage.get_session_details(session_id)
    # print("Session Details Loaded:", details_loaded) # Can be verbose
    print(f"Details loaded successfully: {details_loaded is not None}")

    patient_sessions = storage.get_patient_sessions(p_id1)
    print(f"Sessions for Patient {p_id1}:", patient_sessions)

    # --- Test Thread Safety (Conceptual) ---
    print("\n--- Testing Thread Safety (Conceptual) ---")
    def worker(patient_name):
        local_storage = StorageManager() # Get singleton instance
        new_id = local_storage.add_patient(patient_name, ethnicity="Test")
        print(f"Thread {threading.current_thread().name} added patient {patient_name} with ID {new_id}")
        retrieved = local_storage.get_patient(new_id)
        print(f"Thread {threading.current_thread().name} retrieved: {retrieved['name']}")
        # Connections are implicitly managed per thread by _get_connection
        # local_storage.close_connection() # Optional: close explicitly if thread is ending

    threads = []
    for i in range(3):
        t = threading.Thread(target=worker, args=(f"ThreadPatient_{i}",), name=f"Worker-{i}")
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("Final Patients List:", storage.list_patients())

    # Explicitly close main thread connection if needed (usually not necessary for short scripts)
    storage.close_connection()
    print("Main thread connection closed.")
