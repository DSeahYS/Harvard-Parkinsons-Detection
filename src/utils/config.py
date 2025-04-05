import os
from pathlib import Path
import logging
from dotenv import load_dotenv

class ConfigManager:
    """
    Central configuration manager that loads from .env and provides 
    consistent paths across the application.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.project_root = Path(__file__).parent.parent.parent
        load_dotenv(self.project_root / '.env')
        
        # Default paths relative to project root
        self.db_path = os.getenv('DB_PATH', 'genomeguard.db')
        self.data_dir = os.getenv('DATA_DIR', 'data')
        self.sessions_dir = os.getenv('SESSIONS_DIR', 'data/sessions')
        self.patient_data_dir = os.getenv('PATIENT_DATA_DIR', 'data/patients')
        
        # Create directories if they don't exist
        self._ensure_dirs_exist()
        
    def _ensure_dirs_exist(self):
        """Creates all configured directories if they don't exist."""
        os.makedirs(self.abs_path(self.data_dir), exist_ok=True)
        os.makedirs(self.abs_path(self.sessions_dir), exist_ok=True) 
        os.makedirs(self.abs_path(self.patient_data_dir), exist_ok=True)
        
    def abs_path(self, relative_path):
        """Converts relative paths to absolute paths relative to project root."""
        return str(self.project_root / relative_path)
        
def get_config():
    """Helper function to get the ConfigManager instance."""
    return ConfigManager()