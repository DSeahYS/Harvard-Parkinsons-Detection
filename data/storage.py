import sqlite3
import json
import numpy as np
import threading
import datetime
import os
import logging
from pathlib import Path # Use pathlib for better path handling

# Configure logging (can be overridden by main app)
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Numpy Aware JSON Encoder/Decoder ---
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            # Handle potential NaN/Inf values gracefully
            if np.isnan(obj):
                return None # Or 'NaN' as a string if preferred
            elif np.isinf(obj):
                return None # Or 'Infinity'/' -Infinity'
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime.date, datetime.datetime)):
             return obj.isoformat() # Ensure datetimes are serialized
        return json.JSONEncoder.default(self, obj)

def numpy_decoder(dct):
    """ Custom decoder hook (can be extended if specific structures need reconversion) """
    # This is basic; more complex structures might need specific checks
    # For now, it doesn't automatically convert lists back to arrays.
    return dct

# --- Storage Manager ---
class StorageManager:
    """
    Manages data storage using SQLite for structured data and JSON for details.
    Ensures thread safety for SQLite connections. Uses Singleton pattern.
    Relies on configuration loaded via config.py.
    """
    _instance = None
    _lock = threading.Lock()
    _initialized = False # Class variable to track initialization

    # Use Singleton pattern to ensure single manager instance
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StorageManager, cls).__new__(cls)
                # Reset initialized flag for the new instance before __init__ is called
                cls._instance._initialized_instance = False
            return cls._instance

    def __init__(self):
        """
        Initializes the StorageManager using central configuration.
        Prevents re-initialization.
        """
        # Prevent re-initialization of the same instance
        if hasattr(self, '_initialized_instance') and self._initialized_instance:
            return

        with self._lock: # Ensure thread-safe initialization check
             if hasattr(self, '_initialized_instance') and self._initialized_instance:
                 return

             # Import config dynamically to avoid circular dependencies if config uses storage
             try:
                 from ..utils.config import Config
                 config = Config() # Get config instance
             except ImportError:
                 logger.error("Failed to import Config from ..utils.config. StorageManager cannot be initialized.")
                 # Raise an error or handle appropriately
                 raise ImportError("StorageManager requires ..utils.config.Config")

             self.db_path = config.get_db_path()
             self.sessions_dir = config.get_sessions_dir()

             # Ensure directories exist
             self.db_path.parent.mkdir(parents=True, exist_ok=True)
             self.sessions_dir.mkdir(parents=True, exist_ok=True)

             self._local = threading.local() # Thread-local storage for connections

             # Initialize database schema if it doesn't exist
             try:
                 self._init_db()
                 self._initialized_instance = True # Mark this instance as initialized
                 logger.info(f"StorageManager initialized. DB: '{self.db_path}', Sessions Dir: '{self.sessions_dir}'")
             except Exception as e:
                 logger.error(f"Failed to initialize database: {e}", exc_info=True)
                 # Decide how to handle initialization failure. Raise error?
                 raise RuntimeError(f"StorageManager failed to initialize database: {e}")


    def _get_connection(self):
        """Gets a thread-safe SQLite connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            try:
                # Create a new connection for this thread
                conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)
                conn.execute("PRAGMA journal_mode=WAL;") # Enable Write-Ahead Logging
                conn.execute("PRAGMA foreign_keys = ON;") # Enforce foreign key constraints
                conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
                self._local.connection = conn
                logger.debug(f"Created new SQLite connection for thread {threading.current_thread().name}")
            except sqlite3.Error as e:
                logger.error(f"Error connecting to database '{self.db_path}': {e}", exc_info=True)
                raise ConnectionError(f"Failed to connect to database: {e}")
        return self._local.connection

    def close_connection(self):
        """Closes the connection for the current thread, if it exists."""
        if hasattr(self._local, 'connection') and self._local.connection is not None:
            try:
                self._local.connection.close()
                logger.debug(f"Closed SQLite connection for thread {threading.current_thread().name}")
            except sqlite3.Error as e:
                 logger.error(f"Error closing SQLite connection: {e}", exc_info=True)
            finally:
                 self._local.connection = None # Ensure it's marked as closed


    def _init_db(self):
        """Initializes the database schema if tables don't exist."""
        conn = None
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
                    medical_history TEXT, -- Store as JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Add trigger for updated_at on patients
            cursor.execute('''
                CREATE TRIGGER IF NOT EXISTS update_patient_timestamp
                AFTER UPDATE ON patients FOR EACH ROW
                BEGIN
                    UPDATE patients SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
                END;
            ''')

            # Sessions Table (Summary Data)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    duration_seconds REAL, -- Calculated duration
                    avg_risk_level REAL,
                    avg_saccade_velocity REAL,
                    avg_fixation_stability REAL,
                    avg_blink_rate REAL,
                    genetic_risk_score REAL, -- Store result from BioNeMo if available
                    json_log_filename TEXT UNIQUE, -- Link to detailed JSON log, ensure uniqueness
                    notes TEXT, -- Optional notes for the session
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE CASCADE -- Cascade delete sessions if patient is deleted
                )
            ''')
            # Add index for faster patient session lookups
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_sessions_patient_id ON sessions (patient_id);
            ''')

            # History Table (for longitudinal trend data used by PDDetector)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metric_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL, -- e.g., 'daily_avg_saccade_velocity', 'daily_avg_fixation_stability'
                    value REAL NOT NULL,
                    timestamp TIMESTAMP NOT NULL, -- Timestamp for the aggregated value (e.g., end of day)
                    UNIQUE(metric_name, timestamp) -- Prevent duplicate entries for the same metric/time
                )
            ''')
            cursor.execute('''
                 CREATE INDEX IF NOT EXISTS idx_metric_history_name_time ON metric_history (metric_name, timestamp);
            ''')


            conn.commit()
            logger.info("Database schema initialized/verified successfully.")
        except sqlite3.Error as e:
            logger.error(f"Error initializing database schema: {e}", exc_info=True)
            if conn:
                conn.rollback() # Rollback changes on error
            raise # Re-raise the exception
        # No finally block needed to close connection, managed per thread

    # --- Patient Management ---
    def add_patient(self, name, dob=None, ethnicity=None, medical_history=None):
        """Adds a new patient to the database."""
        sql = '''INSERT INTO patients(name, dob, ethnicity, medical_history)
                 VALUES(?, ?, ?, ?)'''
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            history_json = json.dumps(medical_history, cls=NumpyEncoder) if medical_history else None
            cursor.execute(sql, (name, dob, ethnicity, history_json))
            conn.commit()
            patient_id = cursor.lastrowid
            logger.info(f"Added patient '{name}' with ID: {patient_id}")
            return patient_id
        except sqlite3.Error as e:
            logger.error(f"Error adding patient '{name}': {e}", exc_info=True)
            if conn: conn.rollback()
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
                        logger.warning(f"Could not decode medical history JSON for patient {patient_id}. Returning raw string.")
                return patient_dict
            else:
                logger.warning(f"Patient with ID {patient_id} not found.")
                return None
        except sqlite3.Error as e:
            logger.error(f"Error getting patient {patient_id}: {e}", exc_info=True)
            return None

    def list_patients(self):
        """Lists all patients (ID, name, dob, ethnicity)."""
        sql = "SELECT id, name, dob, ethnicity FROM patients ORDER BY name"
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(sql)
            patients = [dict(row) for row in cursor.fetchall()]
            return patients
        except sqlite3.Error as e:
            logger.error(f"Error listing patients: {e}", exc_info=True)
            return []

    def update_patient(self, patient_id, name=None, dob=None, ethnicity=None, medical_history=None):
        """Updates an existing patient's details."""
        fields = []
        params = []
        if name is not None: fields.append("name = ?"); params.append(name)
        if dob is not None: fields.append("dob = ?"); params.append(dob)
        if ethnicity is not None: fields.append("ethnicity = ?"); params.append(ethnicity)
        if medical_history is not None: fields.append("medical_history = ?"); params.append(json.dumps(medical_history, cls=NumpyEncoder))

        if not fields:
            logger.warning("Update patient called with no fields to update.")
            return False

        sql = f"UPDATE patients SET {', '.join(fields)} WHERE id = ?"
        params.append(patient_id)
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(sql, tuple(params))
            conn.commit()
            updated = cursor.rowcount > 0
            if updated:
                logger.info(f"Updated patient ID: {patient_id}")
            else:
                 logger.warning(f"Attempted to update patient ID {patient_id}, but no rows were affected (patient might not exist).")
            return updated
        except sqlite3.Error as e:
            logger.error(f"Error updating patient {patient_id}: {e}", exc_info=True)
            if conn: conn.rollback()
            return False

    def delete_patient(self, patient_id):
        """Deletes a patient and their associated sessions (due to CASCADE)."""
        sql = "DELETE FROM patients WHERE id = ?"
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(sql, (patient_id,))
            conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted patient ID: {patient_id} and associated sessions.")
            else:
                logger.warning(f"Attempted to delete patient ID {patient_id}, but no rows were affected.")
            return deleted
        except sqlite3.Error as e:
            logger.error(f"Error deleting patient {patient_id}: {e}", exc_info=True)
            if conn: conn.rollback()
            return False


    # --- Session Management ---
    def start_session(self, patient_id=None):
        """Starts a new session and returns its ID."""
        start_time = datetime.datetime.now().isoformat()
        sql = '''INSERT INTO sessions(patient_id, start_time) VALUES(?, ?)'''
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(sql, (patient_id, start_time))
            conn.commit()
            session_id = cursor.lastrowid
            logger.info(f"Started session ID: {session_id} for patient ID: {patient_id}")
            return session_id
        except sqlite3.Error as e:
            logger.error(f"Error starting session for patient {patient_id}: {e}", exc_info=True)
            if conn: conn.rollback()
            return None

    def save_session_details(self, session_id, detailed_data):
        """Saves detailed session data (e.g., metrics over time) to a JSON file."""
        # Ensure sessions directory exists (might be called before __init__ fully completes in some scenarios)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        filename = f"session_{session_id}_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
        filepath = self.sessions_dir / filename
        try:
            with open(filepath, 'w') as f:
                json.dump(detailed_data, f, cls=NumpyEncoder, indent=2) # Use indent=2 for smaller files
            logger.info(f"Saved detailed session log to '{filepath}'")
            return filename # Return filename to link in DB
        except IOError as e:
            logger.error(f"Error saving session details to JSON '{filepath}': {e}", exc_info=True)
            return None
        except TypeError as e:
             logger.error(f"Error serializing session details to JSON: {e}", exc_info=True)
             return None


    def end_session(self, session_id, summary_metrics=None, json_log_filename=None, notes=None):
        """Ends a session, updating summary data in the database."""
        end_time_dt = datetime.datetime.now()
        end_time_iso = end_time_dt.isoformat()

        # Calculate duration
        session_summary = self.get_session_summary(session_id)
        duration_seconds = None
        if session_summary and session_summary.get('start_time'):
            try:
                start_time_dt = datetime.datetime.fromisoformat(session_summary['start_time'])
                duration_seconds = (end_time_dt - start_time_dt).total_seconds()
            except (ValueError, TypeError):
                 logger.warning(f"Could not parse start_time '{session_summary.get('start_time')}' to calculate duration for session {session_id}.")


        sql = '''UPDATE sessions SET
                    end_time = ?, duration_seconds = ?, avg_risk_level = ?,
                    avg_saccade_velocity = ?, avg_fixation_stability = ?, avg_blink_rate = ?,
                    genetic_risk_score = ?, json_log_filename = ?, notes = ?
                 WHERE id = ?'''

        # Extract metrics safely from summary_metrics dict
        metrics = summary_metrics or {}
        params = (
            end_time_iso,
            duration_seconds,
            metrics.get('avg_risk_level'),
            metrics.get('avg_saccade_velocity'),
            metrics.get('avg_fixation_stability'),
            metrics.get('avg_blink_rate'),
            metrics.get('genetic_risk_score'),
            json_log_filename,
            notes,
            session_id
        )
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            updated = cursor.rowcount > 0
            if updated:
                logger.info(f"Ended session ID: {session_id}")
            else:
                logger.warning(f"Attempted to end session ID {session_id}, but no rows were affected.")
            return updated
        except sqlite3.Error as e:
            logger.error(f"Error ending session {session_id}: {e}", exc_info=True)
            if conn: conn.rollback()
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
            logger.error(f"Error getting session summary {session_id}: {e}", exc_info=True)
            return None

    def get_session_details(self, session_id):
        """Loads detailed session data from its JSON file."""
        summary = self.get_session_summary(session_id)
        if not summary or not summary.get('json_log_filename'):
            # logger.warning(f"No JSON log filename found for session {session_id}")
            return None # Return None silently if no log file linked

        filepath = self.sessions_dir / summary['json_log_filename']
        if not filepath.is_file():
             logger.error(f"Session detail file not found: '{filepath}'")
             return None

        try:
            with open(filepath, 'r') as f:
                details = json.load(f, object_hook=numpy_decoder)
            return details
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding session detail JSON '{filepath}': {e}", exc_info=True)
            return None
        except IOError as e:
            logger.error(f"Error reading session detail file '{filepath}': {e}", exc_info=True)
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
            logger.error(f"Error getting sessions for patient {patient_id}: {e}", exc_info=True)
            return []

    # --- History Management (for PDDetector trends) ---
    def save_metric_history(self, metric_name, value, timestamp):
        """Saves a single aggregated metric value to the history table."""
        sql = '''INSERT INTO metric_history(metric_name, value, timestamp)
                 VALUES(?, ?, ?)
                 ON CONFLICT(metric_name, timestamp) DO UPDATE SET value = excluded.value'''
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            # Ensure timestamp is in ISO format string
            ts_iso = timestamp.isoformat() if isinstance(timestamp, (datetime.date, datetime.datetime)) else str(timestamp)
            cursor.execute(sql, (metric_name, value, ts_iso))
            conn.commit()
            logger.debug(f"Saved/Updated metric history: {metric_name} = {value} at {ts_iso}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Error saving metric history for {metric_name}: {e}", exc_info=True)
            if conn: conn.rollback()
            return False

    def load_metric_history(self, metric_name, days_limit=None):
        """Loads historical data for a specific metric, optionally limited by days."""
        sql = f"SELECT value, timestamp FROM metric_history WHERE metric_name = ? "
        params = [metric_name]
        if days_limit and isinstance(days_limit, int) and days_limit > 0:
            limit_date = (datetime.datetime.now() - datetime.timedelta(days=days_limit)).isoformat()
            sql += "AND timestamp >= ? "
            params.append(limit_date)
        sql += "ORDER BY timestamp ASC" # Load in chronological order

        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(sql, tuple(params))
            # Return list of values, assuming PDDetector's CycleBuffer only needs the values
            history_values = [row['value'] for row in cursor.fetchall()]
            logger.info(f"Loaded {len(history_values)} history points for metric '{metric_name}' (Limit: {days_limit} days)")
            return history_values
        except sqlite3.Error as e:
            logger.error(f"Error loading metric history for {metric_name}: {e}", exc_info=True)
            return []


# --- Cleanup Function ---
# Optional: Define a function to close connections for all threads if needed at app shutdown
# This is complex to manage correctly, often relying on thread lifecycle management.
# For simpler apps, connections might just close when threads exit.

# Example Usage (requires config setup)
if __name__ == '__main__':
    print("Running StorageManager example...")
    print("NOTE: This example requires a valid configuration setup in src/utils/config.py")

    try:
        # This will likely fail if config isn't set up correctly
        storage = StorageManager()

        # --- Test Patient Management ---
        print("\n--- Testing Patient Management ---")
        p_id1 = storage.add_patient("Alice Test", dob="1991-01-01", ethnicity="Test", medical_history={"notes": "Test case"})
        if p_id1:
            print("Patients List:", storage.list_patients())
            alice = storage.get_patient(p_id1)
            print("Get Alice:", alice)
            storage.update_patient(p_id1, name="Alice Updated")
            print("Get Updated Alice:", storage.get_patient(p_id1))

            # --- Test Session Management ---
            print("\n--- Testing Session Management ---")
            session_id = storage.start_session(patient_id=p_id1)
            if session_id:
                print(f"Started session: {session_id}")
                detailed_log = {'data': [1, 2, 3], 'metric': np.random.rand()}
                json_file = storage.save_session_details(session_id, detailed_log)
                print(f"Saved details to: {json_file}")
                time.sleep(0.1)
                summary_data = {'avg_risk_level': 0.55, 'avg_saccade_velocity': 210.5}
                storage.end_session(session_id, summary_metrics=summary_data, json_log_filename=json_file, notes="Test session notes.")
                print("Ended session.")
                summary = storage.get_session_summary(session_id)
                print("Session Summary:", summary)
                details_loaded = storage.get_session_details(session_id)
                print(f"Details loaded successfully: {details_loaded is not None}")
                patient_sessions = storage.get_patient_sessions(p_id1)
                print(f"Sessions for Patient {p_id1}:", patient_sessions)

            # --- Test History Management ---
            print("\n--- Testing History Management ---")
            ts = datetime.datetime.now()
            storage.save_metric_history('daily_avg_saccade_velocity', 205.5, ts - datetime.timedelta(days=2))
            storage.save_metric_history('daily_avg_saccade_velocity', 215.0, ts - datetime.timedelta(days=1))
            storage.save_metric_history('daily_avg_fixation_stability', 0.45, ts - datetime.timedelta(days=1))
            history = storage.load_metric_history('daily_avg_saccade_velocity', days_limit=30)
            print("Loaded Velocity History:", history)
            history_fix = storage.load_metric_history('daily_avg_fixation_stability', days_limit=30)
            print("Loaded Fixation History:", history_fix)


            # --- Test Deletion ---
            print("\n--- Testing Deletion ---")
            # storage.delete_patient(p_id1)
            # print("Patients after delete:", storage.list_patients())
            # print(f"Sessions for Patient {p_id1} after delete:", storage.get_patient_sessions(p_id1))


        else:
            print("Failed to add initial patient.")

    except (ImportError, RuntimeError, ConnectionError) as e:
        print(f"\n*** Example Usage Failed: {e} ***")
        print("*** Ensure src/utils/config.py is present and configured correctly. ***")
    except Exception as e:
         print(f"\n*** An unexpected error occurred during example usage: {e} ***")

    finally:
        # Attempt to close the connection for the main thread
        if StorageManager._instance:
            try:
                StorageManager._instance.close_connection()
                print("Main thread storage connection closed.")
            except Exception as e:
                print(f"Error closing main thread connection: {e}")