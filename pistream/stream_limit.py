"""Cap concurrent MJPEG stream clients."""

from threading import Lock


class StreamLimiter:
    def __init__(self, max_clients: int = 4):
        self.max_clients = max(1, int(max_clients))
        self._n = 0
        self._lock = Lock()

    def acquire(self) -> bool:
        with self._lock:
            if self._n >= self.max_clients:
                return False
            self._n += 1
            return True

    def release(self) -> None:
        with self._lock:
            self._n = max(0, self._n - 1)
