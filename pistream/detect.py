"""MoveNet pose detection and async worker."""

import logging
import os
import ssl
import time
import urllib.request
from threading import Event, Lock, Thread
from typing import Optional

import cv2
import numpy as np

from pistream.config import project_root

logger = logging.getLogger(__name__)


class PoseDetector:
    """MoveNet Lightning pose detector optimized for Raspberry Pi."""

    # Primary: TFHub redirect (follows 302 to the actual GCS object)
    # Fallback: original GCS direct path (kept in case it is restored)
    MODEL_URLS = [
        "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite",
        "https://storage.googleapis.com/tfhub-lite-models/google/lite-model/movenet/singlepose/lightning/tflite/float16/4.tflite",
    ]
    # Minimum expected size for a valid MoveNet Lightning float16 model (~1.8 MB)
    MODEL_MIN_BYTES = 1_500_000

    def __init__(self, model_path: Optional[str] = None, threads: Optional[int] = None,
                 confidence: float = 0.5, keypoint_threshold: float = 0.3):
        self.confidence = confidence
        self.keypoint_threshold = keypoint_threshold
        self.threads = threads or max(1, (os.cpu_count() or 2) // 2)
        self._interpreter = None
        self._input_size = (192, 192)
        self._input_buffer = None
        self._keypoints_buffer = np.empty((17, 3), dtype=np.float32)

        self._init_model(model_path)

    def _init_model(self, model_path: Optional[str]):
        """Initialize TFLite interpreter."""
        try:
            # Try tflite-runtime first (faster on Pi)
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                try:
                    from ai_edge_litert.interpreter import Interpreter
                except ImportError:
                    try:
                        from tensorflow.lite.python.interpreter import Interpreter
                    except ImportError as exc:
                        raise ImportError(
                            'No TensorFlow Lite interpreter is installed. On a desktop, run '
                            'pip install -e ".[desktop]"; on Raspberry Pi, install '
                            'tflite-runtime in the tracker environment.'
                        ) from exc

            path = self._get_model(model_path)
            if not path:
                return

            try:
                self._interpreter = Interpreter(model_path=path, num_threads=self.threads)
            except TypeError:
                self._interpreter = Interpreter(model_path=path)

            self._interpreter.allocate_tensors()
            self._input = self._interpreter.get_input_details()[0]
            self._output = self._interpreter.get_output_details()[0]

            shape = self._input['shape']
            self._input_size = (int(shape[1]), int(shape[2]))
            self._input_buffer = np.zeros((1, *self._input_size, 3), dtype=self._input['dtype'])

            logger.info(f"PoseDetector ready: {self._input_size}, {self.threads} threads")

        except Exception as e:
            logger.error(f"Detector init failed: {e}")
            self._interpreter = None

    def _get_model(self, model_path: Optional[str]) -> Optional[str]:
        """Get or download model file."""
        if model_path and os.path.exists(model_path):
            return model_path

        models_dir = project_root() / "models"
        models_dir.mkdir(exist_ok=True)
        path = models_dir / "movenet_lightning.tflite"

        # Validate any cached file before trusting it
        if path.exists():
            if path.stat().st_size >= self.MODEL_MIN_BYTES:
                return str(path)
            logger.warning(
                f"Cached model too small ({path.stat().st_size} B < {self.MODEL_MIN_BYTES} B) — "
                "deleting and re-downloading"
            )
            path.unlink()

        ctx = ssl.create_default_context()
        for url in self.MODEL_URLS:
            logger.info(f"Downloading MoveNet model from {url} ...")
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'PiTracker/1.0'})
                with urllib.request.urlopen(req, context=ctx, timeout=90) as resp:
                    data = resp.read()
                if len(data) >= self.MODEL_MIN_BYTES:
                    path.write_bytes(data)
                    logger.info(f"Model saved: {path} ({len(data) // 1024} KB)")
                    return str(path)
                logger.warning(f"Download from {url} returned only {len(data)} B — skipping")
            except Exception as e:
                logger.warning(f"Download failed ({url}): {e}")

        logger.error("All model download URLs failed")
        return None

    def detect(self, frame: np.ndarray) -> Optional[dict]:
        """Detect person in frame. Returns dict with bbox, confidence, keypoints."""
        if self._interpreter is None:
            return None

        h, w = frame.shape[:2]
        input_h, input_w = self._input_size

        # Preprocess
        resized = cv2.resize(frame, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        cv2.cvtColor(resized, cv2.COLOR_BGR2RGB, resized)

        if self._input['dtype'] == np.float32:
            self._input_buffer[0] = resized.astype(np.float32) / 255.0
        else:
            self._input_buffer[0] = resized

        # Inference
        try:
            self._interpreter.set_tensor(self._input['index'], self._input_buffer)
            self._interpreter.invoke()
            output = self._interpreter.get_tensor(self._output['index']).astype(np.float32)
        except Exception:
            return None

        # Parse keypoints [1, 1, 17, 3] -> [17, 3]
        raw = output[0, 0]
        scores = raw[:, 2]
        valid = scores >= self.keypoint_threshold

        if not np.any(valid):
            return None

        # Scale to frame coordinates
        kp = self._keypoints_buffer
        kp[:, 0] = raw[:, 1] * w  # x
        kp[:, 1] = raw[:, 0] * h  # y
        kp[:, 2] = scores

        # Bounding box from valid keypoints
        xs, ys = kp[valid, 0], kp[valid, 1]
        x1, y1 = max(0, int(np.min(xs))), max(0, int(np.min(ys)))
        x2, y2 = min(w - 1, int(np.max(xs))), min(h - 1, int(np.max(ys)))

        conf = float(np.mean(scores[valid]))
        if conf < self.confidence:
            return None

        return {
            'bbox': (x1, y1, x2 - x1, y2 - y1),
            'confidence': min(conf, 1.0),
            'keypoints': kp.copy()
        }

    @property
    def ready(self) -> bool:
        return self._interpreter is not None


class AsyncDetector:
    """Runs pose detection in a background thread so the main loop never blocks.

    The main loop submits frames via :meth:`submit` (non-blocking) and polls
    for results via :meth:`poll_result` (also non-blocking).  The heavy
    MoveNet inference happens entirely inside the worker thread.
    """

    def __init__(self, detector: PoseDetector, process_scale: float,
                 frame_width: int, frame_height: int, max_result_age: float = 1.0):
        self._detector = detector
        self._process_scale = process_scale

        # Inter-thread communication
        self._submit_lock = Lock()
        self._result_lock = Lock()
        self._pending_frame: Optional[np.ndarray] = None
        self._result: Optional[dict] = None
        self._new_result = False
        self._busy = False
        self._generation = 0
        self._pending_generation = 0
        self._pending_at = 0.0
        self._result_at = 0.0
        self._max_result_age = max_result_age

        self._stop = Event()
        self._has_frame = Event()
        self._thread = Thread(target=self._run, daemon=True,
                              name="AsyncDetector")

        # Pre-allocate scaled buffer once
        if process_scale < 1.0:
            sh = max(1, int(frame_height * process_scale))
            sw = max(1, int(frame_width * process_scale))
            self._scaled_buf = np.empty((sh, sw, 3), dtype=np.uint8)
        else:
            self._scaled_buf = None

    # -- public API (called from main thread) --------------------------------

    def start(self):
        self._stop.clear()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self.invalidate()
        self._has_frame.set()  # unblock the worker if waiting
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        return self.closed

    @property
    def closed(self) -> bool:
        return not self._thread.is_alive()

    def invalidate(self):
        """Cancel pending/inflight work across Reset, stream recovery and stop."""
        with self._submit_lock, self._result_lock:
            self._generation += 1
            self._pending_frame = None
            self._result = None
            self._new_result = False

    def submit(self, frame: np.ndarray, captured_at: float | None = None):
        """Submit a frame for background detection (non-blocking).

        The frame is **copied** internally so the caller can reuse its buffer.
        If the detector is already busy, the new frame silently replaces the
        pending one (latest-wins).
        """
        with self._submit_lock:
            if self._stop.is_set():
                return
            self._pending_frame = frame.copy()
            self._pending_generation = self._generation
            self._pending_at = time.monotonic() if captured_at is None else captured_at
        self._has_frame.set()

    def poll_result(self) -> tuple[bool, Optional[dict]]:
        """Return ``(is_new, result)`` so a new negative result is observable."""
        with self._result_lock:
            if self._new_result:
                self._new_result = False
                if time.monotonic() - getattr(self, '_result_at', 0) > getattr(
                    self, '_max_result_age', float('inf')
                ):
                    return False, None
                return True, self._result
        return False, None

    @property
    def busy(self) -> bool:
        return self._busy

    # -- worker thread -------------------------------------------------------

    def _run(self):
        while not self._stop.is_set():
            self._has_frame.wait(timeout=0.5)
            if self._stop.is_set():
                break
            self._has_frame.clear()

            with self._submit_lock:
                frame = self._pending_frame
                self._pending_frame = None
                generation = self._pending_generation
                captured_at = self._pending_at
            if frame is None:
                continue

            self._busy = True
            try:
                result = self._detect(frame)
                with self._submit_lock, self._result_lock:
                    if generation == self._generation and not self._stop.is_set():
                        self._result = result
                        self._result_at = captured_at
                        self._new_result = True
            except Exception:
                logger.exception('Async detection failed')
            finally:
                self._busy = False

    def _detect(self, frame: np.ndarray) -> Optional[dict]:
        """Run detection (called inside worker thread)."""
        h, w = frame.shape[:2]
        if self._process_scale < 1.0:
            shape = (max(1, int(h * self._process_scale)),
                     max(1, int(w * self._process_scale)), 3)
            if self._scaled_buf is None or self._scaled_buf.shape != shape:
                self._scaled_buf = np.empty(shape, dtype=np.uint8)
            cv2.resize(frame,
                       (self._scaled_buf.shape[1], self._scaled_buf.shape[0]),
                       dst=self._scaled_buf,
                       interpolation=cv2.INTER_AREA)
            small = self._scaled_buf
        else:
            small = frame

        det = self._detector.detect(small)
        if det is None:
            return None

        # Scale coordinates back to original frame size
        x, y, bw, bh = det['bbox']
        scale_x, scale_y = w / small.shape[1], h / small.shape[0]
        x = int(x * scale_x)
        y = int(y * scale_y)
        bw = int(bw * scale_x)
        bh = int(bh * scale_y)

        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = min(bw, w - x)
        bh = min(bh, h - y)

        if det['keypoints'] is not None:
            det['keypoints'][:, :2] *= (scale_x, scale_y)

        det['bbox'] = (x, y, bw, bh)
        return det
