import sqlite3
import json
import numpy as np
import threading
import datetime
import os
import logging
from pathlib import Path # Use pathlib for better path handling
import uuid

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

             # Verify directory permissions
             logger.info(f"Absolute sessions directory: {self.sessions_dir.absolute()}")
             logger.info(f"Directory writable: {os.access(self.sessions_dir, os.W_OK)}")

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
                    id TEXT PRIMARY KEY, -- Use UUID as TEXT primary key
                    patient_id INTEGER,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP, -- Store end time when session finishes
                    duration_seconds REAL, -- Calculated duration at the end
                    session_dir_path TEXT NOT NULL, -- Path to the directory holding JSON files
                    -- Store final calculated averages for quick lookup/display if needed
                    final_avg_risk_level REAL,
                    final_avg_saccade_velocity REAL,
                    final_avg_fixation_stability REAL,
                    final_avg_blink_rate REAL,
                    final_genetic_risk_score REAL,
                    notes TEXT, -- Optional notes for the session
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE CASCADE
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
        """
        Starts a new session:
        1. Generates a UUID for the session ID.
        2. Creates a dedicated directory for the session's JSON files.
        3. Inserts a basic record into the SQLite 'sessions' table.
        Returns the session ID (UUID string).
        """
        session_id = str(uuid.uuid4())
        start_time = datetime.datetime.now()
        start_time_iso = start_time.isoformat()

        # Create session directory using the configured base path
        try:
            session_dir = self.sessions_dir / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created session directory: {session_dir}")
        except OSError as e:
            logger.error(f"Failed to create session directory '{session_dir}': {e}", exc_info=True)
            return None # Cannot proceed without session directory

        # Insert lightweight record into SQLite
        sql = '''INSERT INTO sessions(id, patient_id, start_time, session_dir_path)
                 VALUES(?, ?, ?, ?)'''
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(sql, (session_id, patient_id, start_time_iso, str(session_dir)))
            conn.commit()
            logger.info(f"Started session ID: {session_id} for patient ID: {patient_id} in DB.")
            return session_id # Return the UUID string
        except sqlite3.Error as e:
            logger.error(f"Error starting session {session_id} in DB for patient {patient_id}: {e}", exc_info=True)
            if conn: conn.rollback()
            return None


    def save_metric_stream(self, session_id, metrics):
        """Appends a single metric dictionary to the session's metrics.jsonl file."""
        if not session_id:
            logger.error("Cannot save metric stream: session_id is None.")
            return

        try:
            session_dir = self.sessions_dir / session_id
            if not session_dir.exists():
                logger.error(f"Session directory {session_dir} does not exist!")
                return

            # Use temp file for atomic writes
            temp_file = session_dir / 'metrics.tmp'
            final_file = session_dir / 'metrics.jsonl'

            # Use lock for thread safety
            with threading.Lock():
                # Write to temp file first
                with open(temp_file, 'a', encoding='utf-8') as f:
                    json.dump({
                        'timestamp': datetime.datetime.now().isoformat(),
                        **metrics
                    }, f, cls=NumpyEncoder)
                    f.write('\n')
                    f.flush()  # Ensure data is written
                    os.fsync(f.fileno())  # Force OS-level flush

                # Rename temp file to final file (atomic on Unix)
                temp_file.rename(final_file)

        except Exception as e:
            logger.error(f"Critical error saving metric stream: {e}", exc_info=True)
            # Clean up temp file if it exists
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass


    def save_genomic_results(self, session_id, results):
        """Saves the genomic analysis results to genomic.json for the session."""
        if not session_id:
            logger.error("Cannot save genomic results: session_id is None.")
            return

        try:
            session_dir = self.sessions_dir / session_id
            session_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            genomic_file = session_dir / 'genomic.json'

            with open(genomic_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, cls=NumpyEncoder, indent=2)
            logger.info(f"Saved genomic results for session {session_id} to {genomic_file}")

        except Exception as e:
            logger.error(f"Error saving genomic results for session {session_id}: {e}", exc_info=True)


    # Note: save_session_details is now deprecated/removed in favor of streaming.


    def end_session(self, session_id, notes=None, error=None):
        """
        Ends a session:
        1. Gets session directory from database
        2. Reads metrics.jsonl to calculate final average metrics
        3. Reads genomic.json to get the final genetic score
        4. Updates the SQLite session record

        Args:
            session_id: The session ID to end
            notes: Optional notes about the session
            error: Optional error message if session ended abnormally
        """
        if not session_id:
            logger.error("Cannot end session: session_id is None.")
            return False

        # Get session directory from DB
        session_info = self.get_session_summary(session_id)
        if not session_info:
            logger.error(f"Session {session_id} not found in database")
            return False
            
        try:
            session_dir = Path(session_info['session_dir_path'])
        except Exception as e:
            logger.error(f"Invalid session directory path: {e}")
            return False

        end_time_dt = datetime.datetime.now()
        end_time_iso = end_time_dt.isoformat()

        # Calculate duration
        duration_seconds = None
        try:
            if session_info.get('start_time'):
                start_time_dt = datetime.datetime.fromisoformat(session_info['start_time'])
                duration_seconds = (end_time_dt - start_time_dt).total_seconds()
        except Exception as e:
            logger.warning(f"Error calculating duration: {e}")

        # Calculate averages from metrics
        metrics = []
        try:
            metric_file = session_dir / 'metrics.jsonl'
            if metric_file.exists():
                with open(metric_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            metrics.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON in metrics: {e}")
        except Exception as e:
            logger.error(f"Error reading metrics: {e}")

        # Calculate averages with null handling
        saccades = [m.get('saccade_velocity_deg_s', 0) for m in metrics if m]
        fixations = [m.get('fixation_stability_deg', 0) for m in metrics if m]
        blinks = [m.get('blink_rate_bpm', 0) for m in metrics if m]
        
        avg_saccade = sum(saccades)/len(saccades) if saccades else 0
        avg_fixation = sum(fixations)/len(fixations) if fixations else 0
        avg_blink = sum(blinks)/len(blinks) if blinks else 0

        # Get genetic risk score
        genetic_score = None
        try:
            genomic_file = session_dir / 'genomic.json'
            if genomic_file.is_file():
                with open(genomic_file, 'r', encoding='utf-8') as f:
                    genomic_data = json.load(f)
                    final_genetic_score = genomic_data.get('genetic_risk_score')
                    logger.info(f"Loaded final genetic score for session {session_id}: {final_genetic_score}")
            else:
                 logger.warning(f"Genomic file not found for session {session_id}: {genomic_file}")
        except Exception as e:
             logger.error(f"Error reading genomic file for session {session_id}: {e}", exc_info=True)


        # --- Update SQLite Record ---
        sql = '''UPDATE sessions SET
                    end_time = ?, duration_seconds = ?,
                    final_avg_risk_level = ?, final_avg_saccade_velocity = ?,
                    final_avg_fixation_stability = ?, final_avg_blink_rate = ?,
                    final_genetic_risk_score = ?, notes = ?
                 WHERE id = ?'''

        # Combine notes and error if both exist
        final_notes = notes or ""
        if error:
            final_notes = f"{final_notes}\nERROR: {error}".strip()

        params = (
            end_time_iso,
            duration_seconds,
            final_avg_risk if final_avg_risk is not None else 0.0,  # Default to 0.0 if None
            avg_saccade if avg_saccade is not None else 0.0,
            avg_fixation if avg_fixation is not None else 0.0,
            avg_blink if avg_blink is not None else 0.0,
            genetic_score if genetic_score is not None else 1.0,  # Default genetic score
            final_notes,
            session_id
        )

        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            
            if cursor.rowcount > 0:
                logger.info(f"Successfully updated session {session_id} with end metrics")
                return True
            else:
                logger.warning(f"No session found with ID {session_id} to update")
                return False
                
        except sqlite3.Error as e:
            logger.error(f"Database error updating session {session_id}: {e}")
            if conn: conn.rollback()
            return False
        except Exception as e:
            logger.error(f"Unexpected error updating session {session_id}: {e}")
            return False
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
        """Loads historical metric data for trend analysis"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            sql = "SELECT * FROM metric_history WHERE metric_name = ?"
            params = [metric_name]
            
            if days_limit:
                sql += " AND timestamp >= datetime('now', ?)"
                params.append(f"-{days_limit} days")
                
            sql += " ORDER BY timestamp"
            
            cursor.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]
            
        except sqlite3.Error as e:
            logger.error(f"Error loading metric history: {e}")
            return []

    def get_session_dir(self, session_id):
        """Returns Path object for session directory"""
        session_info = self.get_session_summary(session_id)
        if not session_info or not session_info.get('session_dir_path'):
            raise ValueError(f"Invalid session ID or missing directory path: {session_id}")
        return Path(session_info['session_dir_path'])

    def get_session_details(self, session_id):
        """Retrieves basic session metadata from database"""
        sql = "SELECT * FROM sessions WHERE id = ?"
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(sql, (session_id,))
            result = cursor.fetchone()
            return dict(result) if result else None
        except sqlite3.Error as e:
            logger.error(f"Error getting session details: {e}")
            return None

    def get_session_data(self, session_id):
        """Get complete session data from database and JSON files"""
        session_info = self.get_session_details(session_id)
        if not session_info:
            return None

        session_dir = Path(session_info['session_dir_path'])
        data = {
            'metadata': dict(session_info),
            'metrics': [],
            'genomic': {}
        }

        # Load metrics from JSONL
        metrics_file = session_dir / 'metrics.jsonl'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                data['metrics'] = [json.loads(line) for line in f]

        # Load genomic data
        genomic_file = session_dir / 'genomic.json'
        if genomic_file.exists():
            with open(genomic_file, 'r') as f:
                data['genomic'] = json.load(f)

        return data

    def get_session_data(self, session_id):
        """Retrieves complete session data from database and JSON files"""
        try:
            # Get basic session info from SQLite
            session_details = self.get_session_details(session_id)
            if not session_details:
                return None
            
            # Get metrics from JSONL
            session_dir = self.get_session_dir(session_id)
            metrics = []
            metrics_file = session_dir / 'metrics.jsonl'
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    for line in f:
                        try:
                            metrics.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON in metrics file: {e}")
            
            # Get genomic results
            genomic_file = session_dir / 'genomic.json'
            genomic = {}
            if genomic_file.exists():
                with open(genomic_file, 'r') as f:
                    try:
                        genomic = json.load(f)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in genomic file: {e}")
            
            return {
                'metadata': session_details,
                'metrics': metrics,
                'genomic': genomic
            }
            
        except Exception as e:
            logger.error(f"Error loading session data: {e}")
            return None
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
