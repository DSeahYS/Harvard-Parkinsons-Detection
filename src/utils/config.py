import os
from pathlib import Path
import logging
from dotenv import load_dotenv
import threading

# Configure logging (can be overridden by main app)
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    """
    Central configuration manager that loads from .env (optional) and provides
    consistent paths and settings across the application. Uses Singleton pattern.
    """
    _instance = None
    _lock = threading.Lock()
    _initialized = False # Class variable to track initialization

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Config, cls).__new__(cls)
                # Reset initialized flag for the new instance before __init__ is called
                cls._instance._initialized_instance = False
            return cls._instance

    def __init__(self):
        """
        Initializes the Config. Prevents re-initialization.
        Loads settings from environment variables or defaults.
        """
        # Prevent re-initialization of the same instance
        if hasattr(self, '_initialized_instance') and self._initialized_instance:
            return

        with self._lock: # Ensure thread-safe initialization check
            if hasattr(self, '_initialized_instance') and self._initialized_instance:
                return

            # Determine project root (assuming this file is in TrueGenomeGuard/src/utils)
            # Go up three levels: utils -> src -> TrueGenomeGuard
            self.project_root = Path(__file__).resolve().parent.parent.parent
            logger.debug(f"Project root determined as: {self.project_root}")

            # Load .env file from project root if it exists
            dotenv_path = self.project_root / '.env'
            if dotenv_path.exists():
                load_dotenv(dotenv_path=dotenv_path)
                logger.info(f"Loaded environment variables from: {dotenv_path}")
            else:
                logger.info(".env file not found at project root, using defaults and environment variables.")

            # --- Define Configuration Settings ---
            # Get from environment variables or use defaults

            # Database path (relative to project root)
            self._db_path_relative = os.getenv('DB_PATH', 'truegenomeguard.db')

            # Data directories (relative to project root)
            self._data_dir_relative = os.getenv('DATA_DIR', 'data')
            self._sessions_dir_relative = os.getenv('SESSIONS_DIR', os.path.join(self._data_dir_relative, 'sessions'))
            self._patient_data_dir_relative = os.getenv('PATIENT_DATA_DIR', os.path.join(self._data_dir_relative, 'patients'))
            self._logs_dir_relative = os.getenv('LOGS_DIR', 'logs') # Added logs directory

            # BioNeMo API Key (Example of other config)
            self.bionemo_api_key = os.getenv('BIONEMO_API_KEY')
            # OpenRouter API Key
            self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY') # <<< ADDED

            # Ensure essential directories exist
            self._ensure_dirs_exist()

            self._initialized_instance = True # Mark this instance as initialized
            logger.info("Configuration initialized.")
            logger.info(f"Database path: {self.get_db_path()}")
            logger.info(f"Sessions directory: {self.get_sessions_dir()}")
            logger.info(f"Logs directory: {self.get_logs_dir()}")


    def _ensure_dirs_exist(self):
        """Creates all configured directories if they don't exist."""
        dirs_to_create = [
            self.get_data_dir(),
            self.get_sessions_dir(),
            self.get_patient_data_dir(),
            self.get_logs_dir(),
            self.get_db_path().parent # Ensure DB directory exists
        ]
        for dir_path in dirs_to_create:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {dir_path}")
            except OSError as e:
                logger.error(f"Failed to create directory {dir_path}: {e}", exc_info=True)
                # Decide if this is critical - maybe raise error for essential dirs?

    def _get_abs_path(self, relative_path_str):
        """Converts relative path string to absolute Path object relative to project root."""
        return self.project_root / relative_path_str

    # --- Getter methods for paths ---
    # These return absolute Path objects

    def get_project_root(self) -> Path:
        """Returns the absolute path to the project root directory."""
        return self.project_root

    def get_db_path(self) -> Path:
        """Returns the absolute path to the SQLite database file."""
        return self._get_abs_path(self._db_path_relative)

    def get_data_dir(self) -> Path:
        """Returns the absolute path to the main data directory."""
        return self._get_abs_path(self._data_dir_relative)

    def get_sessions_dir(self) -> Path:
        """Returns validated sessions directory Path object"""
        sessions_dir = self._get_abs_path(self._sessions_dir_relative).resolve()
        
        sessions_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using sessions directory: {sessions_dir}")
        
        # Verify directory is writable
        test_file = sessions_dir / '.permission_test'
        try:
            test_file.touch()
            test_file.unlink()
            logger.debug("Session directory permissions verified")
        except Exception as e:
            logger.error(f"Session directory not writable: {sessions_dir} - {e}")
            raise PermissionError(f"Cannot write to sessions directory: {sessions_dir}")
            
        return sessions_dir

    def get_patient_data_dir(self) -> Path:
        """Returns the absolute path to the patient data directory."""
        return self._get_abs_path(self._patient_data_dir_relative)

    def get_logs_dir(self) -> Path:
        """Returns the absolute path to the logs directory."""
        return self._get_abs_path(self._logs_dir_relative)

    # --- Getter for other settings ---
    def get_bionemo_api_key(self) -> str | None:
        """Returns the BioNeMo API key, if configured."""
        return self.bionemo_api_key

    def get_openrouter_api_key(self) -> str | None: # <<< ADDED
        """Returns the OpenRouter API key, if configured.""" # <<< ADDED
        return self.openrouter_api_key # <<< ADDED

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        """
        Retrieves a configuration setting from environment variables.

        Args:
            key (str): The environment variable key (e.g., 'APP_URL').
            default (str | None): The default value if the key is not found.

        Returns:
            str | None: The value of the environment variable or the default.
        """
        return os.getenv(key, default)

# Helper function remains for easy access, though direct instantiation works too
def get_config():
    """Helper function to get the Config instance."""
    return Config()

# Example Usage (can be removed or kept for testing)
if __name__ == '__main__':
    print("Running Config example...")
    config1 = Config()
    config2 = get_config() # Use helper

    print(f"Config is Singleton: {config1 is config2}")
    print(f"Project Root: {config1.get_project_root()}")
    print(f"DB Path: {config1.get_db_path()}")
    print(f"Sessions Dir: {config1.get_sessions_dir()}")
    print(f"Logs Dir: {config1.get_logs_dir()}")
    print(f"BioNeMo Key: {config1.get_bionemo_api_key() or 'Not Set'}")

    # Test directory creation (should already be done by __init__)
    print("\nVerifying directories exist...")
    print(f"Data Dir exists: {config1.get_data_dir().exists()}")
    print(f"Sessions Dir exists: {config1.get_sessions_dir().exists()}")
    print(f"Logs Dir exists: {config1.get_logs_dir().exists()}")
    print(f"DB Dir exists: {config1.get_db_path().parent.exists()}")
