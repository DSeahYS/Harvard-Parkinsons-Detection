import tkinter as tk
from tkinter import messagebox
import logging
import os
import sys
from pathlib import Path
import traceback

# --- Ensure 'src' is in the Python path ---
# Allows running the script directly from the project root (WorkingGenomeGuard)
try:
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / 'src'
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
        print(f"Added '{src_dir}' to sys.path") # Debug print
    # Test import after path modification
    from utils.config import Config
except ImportError as e:
     print(f"Error: Could not modify sys.path or import initial config. Ensure script is run from WorkingGenomeGuard root. Details: {e}", file=sys.stderr)
     sys.exit(1)
except Exception as e:
    print(f"Error setting up sys.path: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)


# --- Initialize Configuration ---
# This needs to happen before other imports that might use the config
try:
    config = Config() # Assumes config.py is now in WorkingGenomeGuard/src/utils
except Exception as e:
    print(f"Error: Failed to initialize configuration. Details: {e}", file=sys.stderr)
    traceback.print_exc()
    # Attempt to show a graphical error if tkinter is available
    try:
        root = tk.Tk()
        root.withdraw() # Hide the main window
        messagebox.showerror("Configuration Error", f"Failed to initialize configuration:\n{e}\n\nCheck logs and .env file.")
        root.destroy()
    except Exception:
        pass # Ignore if tkinter fails here
    sys.exit(1)


# --- Configure Logging ---
# Setup logging to file and console based on config
log_dir = config.get_logs_dir() # Assumes logs dir is defined in config or defaults
log_file = log_dir / 'workinggenomeguard.log' # Use a new log file name
log_level_str = os.environ.get('LOG_LEVEL', 'INFO').upper()
log_level = getattr(logging, log_level_str, logging.INFO)

# Ensure log directory exists (should be WorkingGenomeGuard/logs)
log_dir.mkdir(parents=True, exist_ok=True)

# Basic config first to catch early issues
# Configure root logger
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # Log to console initially
)

# Add file handler
try:
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler) # Add to root logger
except Exception as e:
    logging.error(f"Failed to configure file logging to {log_file}: {e}")

logger = logging.getLogger(__name__) # Get logger for this module


# --- Import Dashboard (after path setup and config) ---
try:
    # Import from the frontend module within the src directory
    from frontend.tabbed_dashboard import Dashboard
except ImportError as e:
    logger.critical(f"Failed to import Dashboard from frontend.tabbed_dashboard. Error: {e}", exc_info=True)
    messagebox.showerror("Import Error", f"Failed to import application components:\n{e}\n\nPlease check installation and logs.")
    sys.exit(1)
except Exception as e:
     logger.critical(f"An unexpected error occurred during Dashboard import: {e}", exc_info=True)
     messagebox.showerror("Import Error", f"An unexpected error occurred during import:\n{e}")
     sys.exit(1)


# --- Main Application Execution ---
if __name__ == '__main__':
    logger.info("=======================================")
    logger.info("Starting WorkingGenomeGuard Application...")
    logger.info(f"Project Root: {config.get_project_root()}")
    logger.info(f"Log Level: {log_level_str}")
    logger.info("=======================================")

    root = tk.Tk()
    app_instance = None # To hold the Dashboard instance

    try:
        # Set a global exception handler for Tkinter errors
        def handle_tk_exception(exc, val, tb):
            logger.critical("Unhandled Tkinter exception:", exc_info=(exc, val, tb))
            messagebox.showerror("Application Error", f"An unexpected error occurred in the UI:\n{val}\n\nPlease check logs.")
            # Optionally try to close gracefully
            if app_instance:
                try:
                    app_instance._on_close() # Attempt cleanup
                except Exception as close_err:
                     logger.error(f"Error during forced close after UI exception: {close_err}")
            root.destroy() # Force close

        root.report_callback_exception = handle_tk_exception

        app_instance = Dashboard(root) # Create the application instance
        root.mainloop() # Start the Tkinter event loop
        logger.info("WorkingGenomeGuard Application exited normally.")

    except Exception as e:
        logger.critical(f"An unhandled exception occurred in the main application scope: {e}", exc_info=True)
        # Show error message if possible (Tkinter might already be closed)
        try:
            messagebox.showerror("Fatal Error", f"An unexpected error occurred:\n{e}\n\nPlease check the logs in '{log_dir}'.")
        except Exception:
             print(f"FATAL ERROR: {e}. Check logs in '{log_dir}'.", file=sys.stderr)
    finally:
        # Final cleanup attempt (though _on_close should handle most)
        logger.info("Application shutdown sequence complete.")
