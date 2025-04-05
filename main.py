import tkinter as tk
import logging
import os
from dotenv import load_dotenv
import sys
from src.utils.config import ConfigManager

# Initialize configuration first (will setup paths)
config = ConfigManager()

# Ensure the 'src' directory is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from frontend.dashboard import Dashboard
except ImportError as e:
    logging.error(f"Failed to import Dashboard. Ensure 'src' is in PYTHONPATH or run from GenomeGuard root. Error: {e}")
    sys.exit(1)

logging.info("Configuration initialized from .env")

# Configure basic logging for the main entry point
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    logging.info("Starting GenomeGuard Application...")
    root = tk.Tk()
    try:
        app = Dashboard(root)
        root.mainloop()
        logging.info("GenomeGuard Application exited normally.")
    except Exception as e:
        logging.critical(f"An unhandled exception occurred in the main application loop: {e}", exc_info=True)
        # Optionally show an error message to the user
        tk.messagebox.showerror("Fatal Error", f"An unexpected error occurred:\n{e}\n\nPlease check the logs.")
    finally:
        # Ensure cleanup, though on_close in Dashboard should handle most of it
        logging.info("Application shutdown sequence initiated.")
