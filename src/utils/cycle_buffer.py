import threading
from collections import deque

class CycleBuffer:
    """A thread-safe circular buffer."""
    def __init__(self, maxlen):
        """
        Initializes the CycleBuffer.

        Args:
            maxlen (int): The maximum size of the buffer.
        """
        self.buffer = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def append(self, item):
        """
        Appends an item to the buffer in a thread-safe manner.

        Args:
            item: The item to append.
        """
        with self.lock:
            self.buffer.append(item)

    def mean(self):
        """
        Calculates the mean of the items in the buffer in a thread-safe manner.

        Returns:
            float: The mean of the items, or 0 if the buffer is empty.
        """
        with self.lock:
            return sum(self.buffer) / len(self.buffer) if self.buffer else 0

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
