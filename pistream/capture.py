"""Threaded video capture with reconnect."""

import logging
from threading import Event, Lock, Thread

import cv2

logger = logging.getLogger(__name__)


class VideoCapture:
    """Threaded video capture with low-latency buffering and auto-reconnect."""

    MAX_FAILURES = 30        # consecutive read failures before reconnect
    RECONNECT_DELAY = 1.0    # seconds between reconnect attempts
    MAX_RECONNECTS = 10      # give up after this many consecutive reconnects

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
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        if self._is_http:
            # For HTTP MJPEG streams, FFMPEG is the correct backend
            backends = [cv2.CAP_FFMPEG, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_V4L2, cv2.CAP_FFMPEG, cv2.CAP_ANY]

        for backend in backends:
            try:
                self._cap = cv2.VideoCapture(self.source, backend)
                if self._cap.isOpened():
                    break
            except Exception:
                continue

        if not self._cap or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.source)

        if not self._cap or not self._cap.isOpened():
            return False

        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        # For HTTP streams: reduce internal FFMPEG buffer for lower latency
        if self._is_http:
            self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            self._cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

        return True

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
            return False

        self._stop.clear()
        self._thread = Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        logger.info(f"Capture started: {self.width}x{self.height} @ {self.fps:.1f} FPS")
        return True

    def _capture_loop(self):
        failures = 0
        reconnects = 0
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._ret, self._frame = ret, frame
                failures = 0
                reconnects = 0
            else:
                failures += 1
                if failures >= self.MAX_FAILURES:
                    if reconnects >= self.MAX_RECONNECTS:
                        logger.error("Max reconnect attempts reached, giving up")
                        with self._lock:
                            self._ret = False
                        return
                    logger.warning(f"Stream lost ({failures} failures), reconnecting...")
                    self._stop.wait(self.RECONNECT_DELAY)
                    if self._stop.is_set():
                        return
                    if self._open_capture():
                        logger.info("Stream reconnected")
                        failures = 0
                        reconnects += 1
                    else:
                        reconnects += 1
                        logger.warning(f"Reconnect attempt {reconnects}/{self.MAX_RECONNECTS} failed")
                else:
                    self._stop.wait(0.005)

    def read(self):
        """Get latest frame."""
        with self._lock:
            return self._ret, self._frame

    def stop(self):
        """Stop capture."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
