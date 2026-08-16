"""MoveNet pose detection and async worker."""

import logging
import os
import ssl
import urllib.request
from threading import Event, Lock, Thread
from typing import Optional

import cv2
import numpy as np

from pistream.config import project_root

logger = logging.getLogger(__name__)


class PoseDetector:
    """MoveNet Lightning pose detector optimized for Raspberry Pi."""

    KEYPOINTS = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

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
            path = self._get_model(model_path)
            if not path:
                return

            # Try tflite-runtime first (faster on Pi)
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                from tensorflow.lite.python.interpreter import Interpreter

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
    for results via :meth:`get_result` (also non-blocking).  The heavy
    MoveNet inference happens entirely inside the worker thread.
    """

    def __init__(self, detector: PoseDetector, process_scale: float,
                 frame_width: int, frame_height: int):
        self._detector = detector
        self._process_scale = process_scale
        self._scale_inv = 1.0 / process_scale
        self._frame_w = frame_width
        self._frame_h = frame_height

        # Inter-thread communication
        self._submit_lock = Lock()
        self._result_lock = Lock()
        self._pending_frame: Optional[np.ndarray] = None
        self._result: Optional[dict] = None
        self._new_result = False
        self._busy = False

        self._stop = Event()
        self._has_frame = Event()
        self._thread = Thread(target=self._run, daemon=True,
                              name="AsyncDetector")

        # Pre-allocate scaled buffer once
        if process_scale < 1.0:
            sh = int(frame_height * process_scale)
            sw = int(frame_width * process_scale)
            self._scaled_buf = np.empty((sh, sw, 3), dtype=np.uint8)
        else:
            self._scaled_buf = None

    # -- public API (called from main thread) --------------------------------

    def start(self):
        self._stop.clear()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._has_frame.set()  # unblock the worker if waiting
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def submit(self, frame: np.ndarray):
        """Submit a frame for background detection (non-blocking).

        The frame is **copied** internally so the caller can reuse its buffer.
        If the detector is already busy, the new frame silently replaces the
        pending one (latest-wins).
        """
        with self._submit_lock:
            self._pending_frame = frame.copy()
        self._has_frame.set()

    def get_result(self) -> Optional[dict]:
        """Return the latest detection result, or *None* if nothing new."""
        with self._result_lock:
            if self._new_result:
                self._new_result = False
                return self._result
        return None

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
            if frame is None:
                continue

            self._busy = True
            result = self._detect(frame)
            self._busy = False

            with self._result_lock:
                self._result = result
                self._new_result = True

    def _detect(self, frame: np.ndarray) -> Optional[dict]:
        """Run detection (called inside worker thread)."""
        if self._process_scale < 1.0 and self._scaled_buf is not None:
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
        w, h = self._frame_w, self._frame_h
        x, y, bw, bh = det['bbox']
        x = int(x * self._scale_inv)
        y = int(y * self._scale_inv)
        bw = int(bw * self._scale_inv)
        bh = int(bh * self._scale_inv)

        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = min(bw, w - x)
        bh = min(bh, h - y)

        if det['keypoints'] is not None:
            det['keypoints'][:, :2] *= self._scale_inv

        det['bbox'] = (x, y, bw, bh)
        return det
