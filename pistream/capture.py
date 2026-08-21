"""Threaded video capture with reconnect."""

import logging
from threading import Event, Lock, Thread

import cv2

logger = logging.getLogger(__name__)

BURST_RECONNECTS = 10
BURST_DELAY = 1.0
BACKOFF_DELAY = 5.0


def reconnect_wait(reconnects: int) -> float:
    """Seconds to wait before the next reconnect. Never 0 (never give up)."""
    if reconnects > 0 and reconnects % BURST_RECONNECTS == 0:
        return BACKOFF_DELAY
    return BURST_DELAY


class VideoCapture:
    """Threaded video capture with low-latency buffering and auto-reconnect."""

    MAX_FAILURES = 30        # consecutive read failures before reconnect
    MAX_RECONNECTS = BURST_RECONNECTS

    def __init__(self, source: str, buffer_size: int = 2):
        self.source = source
        self.buffer_size = buffer_size
        self._cap = None
        self._frame = None
        self._ret = False
        self._lock = Lock()
        self._stop = Event()
        self._thread = None
        self.width = 0
        self.height = 0
        self.fps = 30.0
        self._is_http = source.startswith(('http://', 'https://'))

    def _open_capture(self) -> bool:
        """Open the video capture with appropriate backend."""
        self._release_capture()

        if self._is_http:
            # For HTTP MJPEG streams, FFMPEG is the correct backend
            backends = [cv2.CAP_FFMPEG, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_V4L2, cv2.CAP_FFMPEG, cv2.CAP_ANY]

        for backend in backends:
            candidate = None
            try:
                if self._is_http:
                    candidate = cv2.VideoCapture(
                        self.source,
                        backend,
                        [
                            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
                        ],
                    )
                else:
                    candidate = cv2.VideoCapture(self.source, backend)
            except Exception:
                # Older OpenCV builds do not expose the constructor params.
                try:
                    candidate = cv2.VideoCapture(self.source, backend)
                except Exception:
                    candidate = None
            if candidate is not None and candidate.isOpened():
                self._cap = candidate
                break
            if candidate is not None:
                candidate.release()

        if not self._cap or not self._cap.isOpened():
            return False

        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        return True

    def _release_capture(self) -> None:
        cap, self._cap = self._cap, None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                logger.debug("Capture release failed", exc_info=True)

    def start(self) -> bool:
        """Start video capture."""
        if not self._open_capture():
            logger.error(f"Failed to open: {self.source}")
            return False

        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

        self._ret, self._frame = self._cap.read()
        if not self._ret:
            logger.error("Failed to read initial frame")
            self._release_capture()
            return False
        if self.width <= 0 or self.height <= 0:
            self.height, self.width = self._frame.shape[:2]

        self._stop.clear()
        self._thread = Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        logger.info(f"Capture started: {self.width}x{self.height} @ {self.fps:.1f} FPS")
        return True

    def _capture_loop(self):
        failures = 0
        reconnects = 0
        try:
            while not self._stop.is_set():
                cap = self._cap
                ret, frame = cap.read() if cap is not None else (False, None)
                if ret:
                    with self._lock:
                        self._ret, self._frame = ret, frame
                    failures = 0
                    reconnects = 0
                else:
                    failures += 1
                    if failures >= self.MAX_FAILURES:
                        with self._lock:
                            self._ret = False
                        reconnects += 1
                        delay = reconnect_wait(reconnects)
                        logger.warning(
                            f"Stream lost ({failures} failures), "
                            f"reconnect {reconnects} in {delay:.0f}s"
                        )
                        self._stop.wait(delay)
                        if self._stop.is_set():
                            return
                        if self._open_capture():
                            logger.info("Stream reconnected")
                            failures = 0
                            reconnects = 0
                        else:
                            logger.warning(f"Reconnect attempt {reconnects} failed")
                    else:
                        self._stop.wait(0.005)
        finally:
            if self._stop.is_set():
                self._release_capture()

    def read(self):
        """Get latest frame."""
        with self._lock:
            return self._ret, self._frame

    def stop(self):
        """Stop capture."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=6.0)
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Capture thread did not stop within the read timeout")
        else:
            self._release_capture()
        with self._lock:
            self._ret = False

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def stream_lost(self) -> bool:
        with self._lock:
            return not self._ret
