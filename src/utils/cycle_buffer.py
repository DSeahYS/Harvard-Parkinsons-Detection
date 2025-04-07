import threading
from collections import deque
import numpy as np # Added for mean calculation

class CycleBuffer:
    """A thread-safe circular buffer."""
    def __init__(self, maxlen):
        """
        Initializes the CycleBuffer.

        Args:
            maxlen (int): The maximum size of the buffer.
        """
        if not isinstance(maxlen, int) or maxlen <= 0:
            raise ValueError("maxlen must be a positive integer")
        self.buffer = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def append(self, item):
        """
        Appends an item to the buffer in a thread-safe manner.

        Args:
            item: The item to append. Should be numeric for mean calculation.
        """
        # Optional: Add type checking if only numbers are expected
        # if not isinstance(item, (int, float)):
        #     print(f"Warning: Appending non-numeric item '{item}' to CycleBuffer.")
        with self.lock:
            self.buffer.append(item)

    def mean(self):
        """
        Calculates the mean of the items in the buffer in a thread-safe manner.

        Returns:
            float: The mean of the items, or 0.0 if the buffer is empty or contains non-numeric data.
        """
        with self.lock:
            if not self.buffer:
                return 0.0
            # Use numpy for potentially better handling of different numeric types and edge cases
            try:
                return float(np.mean(list(self.buffer)))
            except (TypeError, ValueError):
                # Handle cases where buffer might contain non-numeric data if type check wasn't enforced
                print("Warning: Could not calculate mean due to non-numeric data in buffer.")
                # Attempt to calculate mean of numeric items only
                numeric_items = [item for item in self.buffer if isinstance(item, (int, float))]
                return float(np.mean(numeric_items)) if numeric_items else 0.0


    def get_all(self):
        """
        Returns a copy of all items currently in the buffer in a thread-safe manner.

        Returns:
            list: A list containing all items in the buffer.
        """
        with self.lock:
            return list(self.buffer)

    def __len__(self):
        """Returns the current number of items in the buffer."""
        with self.lock:
            return len(self.buffer)

    def is_full(self):
        """Checks if the buffer is full."""
        with self.lock:
            return len(self.buffer) == self.buffer.maxlen

    def clear(self):
        """Removes all items from the buffer."""
        with self.lock:
            self.buffer.clear() 
