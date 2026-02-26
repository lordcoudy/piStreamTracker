#!/usr/bin/env python3
"""
piStreamTracker - Human Tracking System for Raspberry Pi
Optimized MoveNet pose detection with EV3 motor control
"""

import argparse
import logging
import os
import shutil
import ssl
import subprocess
import time
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Optional

import cv2
import numpy as np
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file with defaults."""
    defaults = {
        'network': {'camera_ip': '192.168.100.1', 'tracker_ip': '192.168.100.2'},
        'camera': {'port': 8000},
        'tracker': {
            'output_dir': 'recordings',
            'detection': {'interval': 10, 'scale': 0.4, 'confidence': 0.5, 'keypoint_threshold': 0.3},
            'movenet': {'model_path': None, 'threads': None}
        },
        'ev3': {
            'enabled': True, 'deadzone': {'x': 90, 'y': 90},
            'speed_factor': 1.0, 'max_speed': 50, 'cooldown': 0.5,
            'invert': {'x': False, 'y': False}, 'ports': {'pan': 'a', 'tilt': 'b'}
        },
        'web': {'enabled': True, 'host': '0.0.0.0', 'port': 5000},
        'logging': {'level': 'INFO', 'verbose_shifts': False}
    }

    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            user_config = yaml.safe_load(f) or {}
        # Deep merge
        def merge(base, override):
            for k, v in override.items():
                if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                    merge(base[k], v)
                else:
                    base[k] = v
        merge(defaults, user_config)

    return defaults


# =============================================================================
# Video Capture
# =============================================================================

class VideoCapture:
    """Threaded video capture with low-latency buffering."""

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

    def start(self) -> bool:
        """Start video capture."""
        for backend in [cv2.CAP_V4L2, cv2.CAP_FFMPEG, cv2.CAP_ANY]:
            try:
                self._cap = cv2.VideoCapture(self.source, backend)
                if self._cap.isOpened():
                    break
            except Exception:
                continue

        if not self._cap or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            logger.error(f"Failed to open: {self.source}")
            return False

        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
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
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._ret, self._frame = ret, frame
            else:
                self._stop.wait(0.001)

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


# =============================================================================
# MoveNet Detector
# =============================================================================

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

        models_dir = Path(__file__).parent / "models"
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


# =============================================================================
# Async Detection Thread
# =============================================================================

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


# =============================================================================
# EV3 Motor Controller
# =============================================================================

class MotorController:
    """EV3 motor controller for pan/tilt tracking."""

    def __init__(self, config: dict):
        self.deadzone_x = config['deadzone']['x']
        self.deadzone_y = config['deadzone']['y']
        self.speed_factor = min(config['speed_factor'], 2.0)
        self.max_speed = min(config['max_speed'], 100)
        self.invert_x = config['invert']['x']
        self.invert_y = config['invert']['y']
        self.cooldown = config['cooldown']
        self.pan_port = config['ports']['pan']
        self.tilt_port = config['ports']['tilt']

        self._ev3 = None
        self._pan = None
        self._tilt = None
        self._last_cmd = 0.0
        self._cam_w = 1280
        self._cam_h = 960
        self.connected = False

        if config['enabled']:
            self.connect()

    def connect(self) -> bool:
        """Connect to EV3 via USB."""
        try:
            from ev3_usb import EV3_USB
            logger.info("Connecting to EV3...")
            self._ev3 = EV3_USB()
            self._pan = self._ev3.Motor(self.pan_port)
            self._tilt = self._ev3.Motor(self.tilt_port)
            self.connected = True

            try:
                self._ev3.Led('green', 'pulse')
            except Exception:
                pass

            self.stop()
            logger.info("EV3 connected")
            return True

        except Exception as e:
            logger.warning(f"EV3 connection failed: {e}")
            self.connected = False
            return False

    def set_frame_size(self, width: int, height: int):
        """Update frame dimensions for motor calculations."""
        self._cam_w = width
        self._cam_h = height

    def _drive_axis(self, motor, shift: int, deadzone: int, invert: bool,
                    frame_dim: int, scale: float):
        """Drive a single motor axis based on shift from center."""
        if abs(shift) >= deadzone:
            speed = int(min(abs(shift) / 100.0 * self.speed_factor, self.max_speed))
            degrees = int((-shift if invert else shift) / (frame_dim / scale))
            if speed > 0:
                motor.run_to(degrees=degrees, speed=speed)
        else:
            motor.stop()

    def update(self, shift_x: int, shift_y: int):
        """Update motors based on target offset from center."""
        if not self.connected:
            return

        now = time.monotonic()
        if now - self._last_cmd < self.cooldown:
            return

        try:
            self._drive_axis(self._pan, shift_x, self.deadzone_x,
                             self.invert_x, self._cam_w, 128.0)
            self._drive_axis(self._tilt, shift_y, self.deadzone_y,
                             self.invert_y, self._cam_h, 96.0)
            self._last_cmd = now
        except Exception as e:
            logger.debug(f"Motor error: {e}")

    def stop(self):
        """Stop all motors."""
        if not self.connected:
            return
        try:
            if self._pan:
                self._pan.stop()
            if self._tilt:
                self._tilt.stop()
        except Exception:
            pass

    def disconnect(self):
        """Disconnect from EV3."""
        if self.connected:
            self.stop()
            try:
                if self._ev3:
                    self._ev3.Led('orange', 'static')
            except Exception:
                pass
            try:
                if self._ev3:
                    self._ev3.close()
            except Exception:
                pass
            self._pan = self._tilt = None
            self._ev3 = None
            self.connected = False
            logger.info("EV3 disconnected")


# =============================================================================
# Object Tracker
# =============================================================================

class ObjectTracker:
    """OpenCV-based object tracker."""

    TRACKERS = ["MOSSE", "KCF", "CSRT"]

    def __init__(self):
        self._tracker = None
        self._type = None

    def init(self, frame: np.ndarray, bbox: tuple) -> bool:
        """Initialize tracker with bounding box."""
        x, y, w, h = bbox
        if w < 10 or h < 10:
            return False

        fh, fw = frame.shape[:2]
        x = max(0, min(x, fw - 1))
        y = max(0, min(y, fh - 1))
        w = min(w, fw - x)
        h = min(h, fh - y)

        for name in self.TRACKERS:
            try:
                if hasattr(cv2, 'legacy'):
                    create = getattr(cv2.legacy, f'Tracker{name}_create', None)
                else:
                    create = getattr(cv2, f'Tracker{name}_create', None)

                if create:
                    tracker = create()
                    if tracker.init(frame, (x, y, w, h)):
                        self._tracker = tracker
                        self._type = name
                        return True
            except Exception:
                continue

        return False

    def update(self, frame: np.ndarray) -> Optional[tuple]:
        """Update tracker. Returns bbox or None if lost."""
        if self._tracker is None:
            return None

        try:
            ok, bbox = self._tracker.update(frame)
            if ok:
                return tuple(int(v) for v in bbox)
        except Exception:
            pass

        self._tracker = None
        return None

    def reset(self):
        """Reset tracker."""
        self._tracker = None
        self._type = None

    @property
    def active(self) -> bool:
        return self._tracker is not None

    @property
    def tracker_type(self) -> Optional[str]:
        return self._type


# =============================================================================
# Recording — ffmpeg pipe with hardware-accelerated H.264
# =============================================================================

def _probe_encoder() -> str:
    """Return the best available ffmpeg H.264 encoder.

    Preference order:
      1. h264_v4l2m2m  — Pi hardware encoder (near-zero CPU)
      2. libx264       — CPU-based but vastly more efficient than MJPG

    Returns the encoder name or '' if ffmpeg is not available.
    """
    if not shutil.which('ffmpeg'):
        return ''

    for enc in ('h264_v4l2m2m', 'libx264'):
        try:
            r = subprocess.run(
                ['ffmpeg', '-hide_banner', '-encoders'],
                capture_output=True, text=True, timeout=5,
            )
            if enc in r.stdout:
                logger.info(f"Detected ffmpeg encoder: {enc}")
                return enc
        except Exception:
            continue
    return ''


class _RecordingThread:
    """Writes frames at a fixed framerate via an ffmpeg subprocess pipe.

    Uses hardware-accelerated H.264 when available (h264_v4l2m2m on Pi 5),
    falling back to software libx264 ultrafast, then to the legacy OpenCV
    MJPG writer as a last resort.

    The thread ticks at exactly ``1/fps`` intervals.  If the main loop is
    slower than the target FPS the last available frame is repeated so the
    output file always has a constant frame rate.
    """

    def __init__(self, path: str, width: int, height: int, fps: float,
                 encoder: str = 'auto'):
        self._interval = 1.0 / max(fps, 1.0)
        self._frame: Optional[np.ndarray] = None
        self._lock = Lock()
        self._stop = Event()
        self._thread: Optional[Thread] = None
        self._proc: Optional[subprocess.Popen] = None
        self._cv_writer: Optional[cv2.VideoWriter] = None
        self._width = width
        self._height = height

        self._init_writer(path, width, height, fps, encoder)

    # ---- writer init with fallback chain -----------------------------

    def _init_writer(self, path: str, w: int, h: int, fps: float,
                     encoder: str):
        """Try ffmpeg pipe first, fall back to OpenCV MJPG."""
        if encoder == 'auto':
            encoder = _probe_encoder()

        if encoder:
            try:
                self._start_ffmpeg(path, w, h, fps, encoder)
                return
            except Exception as e:
                logger.warning(f"ffmpeg ({encoder}) failed: {e} — falling back")

        # Fallback: OpenCV MJPG
        logger.info("Using fallback OpenCV MJPG recorder")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        fallback_path = path.rsplit('.', 1)[0] + '.avi'
        self._cv_writer = cv2.VideoWriter(fallback_path, fourcc, fps, (w, h))

    def _start_ffmpeg(self, path: str, w: int, h: int, fps: float,
                      encoder: str):
        """Launch ffmpeg as a subprocess accepting raw BGR frames on stdin."""
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{w}x{h}',
            '-r', str(fps),
            '-i', 'pipe:0',
            '-c:v', encoder,
        ]
        if encoder == 'libx264':
            cmd += ['-preset', 'ultrafast', '-crf', '23']
        elif encoder == 'h264_v4l2m2m':
            cmd += ['-b:v', '4M']

        cmd += [
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            path,
        ]

        logger.info(f"Recording via ffmpeg [{encoder}]: {path}")
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )

    # ---- public API --------------------------------------------------

    def update_frame(self, frame: np.ndarray):
        """Feed the latest frame (non-blocking)."""
        with self._lock:
            self._frame = frame

    def start(self):
        """Start the recording thread."""
        self._stop.clear()
        self._thread = Thread(target=self._run, daemon=True, name="RecordingThread")
        self._thread.start()

    def stop(self):
        """Stop the recording thread and flush/close the writer."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=max(self._interval * 4, 2.0))

        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=10)
            except Exception as e:
                logger.debug(f"ffmpeg close: {e}")
                self._proc.kill()

        if self._cv_writer:
            self._cv_writer.release()

    # ---- internal ----------------------------------------------------

    def _write(self, frame: np.ndarray):
        """Write one frame to whichever backend is active."""
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                logger.warning("ffmpeg pipe broken — stopping recording")
                self._stop.set()
        elif self._cv_writer:
            self._cv_writer.write(frame)

    def _run(self):
        next_tick = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            sleep_for = next_tick - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            next_tick += self._interval

            with self._lock:
                frame = self._frame
            if frame is not None:
                self._write(frame)


# =============================================================================
# Main Tracker Application
# =============================================================================

class HumanTracker:
    """Main tracking application."""

    def __init__(self, config: dict):
        self.config = config

        # Build stream URL
        net = config['network']
        cam = config['camera']
        tracker_cfg = config['tracker']
        url = tracker_cfg.get('stream_url')
        if not url:
            url = f"http://{net['camera_ip']}:{cam['port']}/stream"
        self.stream_url = url

        # Settings (all from config — CLI overrides are merged before construction)
        det = tracker_cfg['detection']
        self.detection_interval = det['interval']
        self.process_scale = det['scale']
        self.output_dir = Path(tracker_cfg['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        self.recording_fps: float = float(tracker_cfg.get('recording_fps', 30))

        # Components
        self.capture = None
        self.detector = PoseDetector(
            model_path=tracker_cfg['movenet'].get('model_path'),
            threads=tracker_cfg['movenet'].get('threads'),
            confidence=det['confidence'],
            keypoint_threshold=det['keypoint_threshold']
        )
        self.tracker = ObjectTracker()
        self.motors = MotorController(config['ev3'])

        # State
        self.running = False
        self.recording = False
        self.current_detection = None
        self.frame_count = 0
        self._rec_thread: Optional[_RecordingThread] = None
        self._fps_count = 0
        self._fps_time = time.monotonic()
        self._fps = 0.0

        # Recording mode: 'local' (Pi 5) or 'camera' (Pi 3B+)
        self.recording_mode = tracker_cfg.get('recording_mode', 'local')
        self._camera_base_url = f"http://{net['camera_ip']}:{cam['port']}"

        # Async detector (created in connect() once we know frame size)
        self._async_detector: Optional[AsyncDetector] = None

        # Shift logger
        self._shift_logger = None
        if config['logging'].get('verbose_shifts'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._shift_logger = ShiftLogger(self.output_dir / f"shifts_{ts}.txt")

    def connect(self) -> bool:
        """Connect to video stream."""
        self.capture = VideoCapture(self.stream_url)
        if not self.capture.start():
            return False

        self.motors.set_frame_size(self.capture.width, self.capture.height)

        # Start async detection thread
        if self.detector.ready:
            self._async_detector = AsyncDetector(
                self.detector, self.process_scale,
                self.capture.width, self.capture.height,
            )
            self._async_detector.start()
            logger.info("Async detection thread started")

        return True

    def process_frame(self, frame: np.ndarray) -> tuple:
        """Process frame, return (annotated_frame, detection).

        Detection runs **asynchronously** in a background thread so this
        method never blocks on MoveNet inference.  Between detections the
        lightweight MOSSE tracker keeps the bounding box up-to-date.
        """
        self.frame_count += 1

        # --- 1. Pick up async detection result (non-blocking) ---------------
        det_result = None
        if self._async_detector is not None:
            det_result = self._async_detector.get_result()
        if det_result:
            self.tracker.init(frame, det_result['bbox'])
            self.current_detection = det_result

        # --- 2. Submit frame for detection when needed ----------------------
        #   • Periodically (every N frames) to refresh the pose estimate
        #   • Urgently every frame when there is nothing to track yet
        if self._async_detector is not None:
            need_detect = (
                self.frame_count % self.detection_interval == 0
                or not self.tracker.active
            )
            if need_detect and not self._async_detector.busy:
                self._async_detector.submit(frame)

        # --- 3. Fast tracker update (MOSSE ≈ 0.5 ms) -----------------------
        detection = None
        tracked_bbox = self.tracker.update(frame) if self.tracker.active else None

        if tracked_bbox is not None:
            detection = {
                'bbox': tracked_bbox,
                'confidence': (
                    self.current_detection.get('confidence', 0.5)
                    if self.current_detection else 0.5
                ),
                'keypoints': (
                    self.current_detection.get('keypoints')
                    if self.current_detection else None
                ),
            }
            self.current_detection = detection
        elif det_result:
            # Tracker init + update failed, but we have a fresh detection
            detection = det_result

        annotated = self._draw(frame.copy(), detection)
        return annotated, detection

    def _draw(self, frame: np.ndarray, detection: Optional[dict]) -> np.ndarray:
        """Draw tracking overlay."""
        cx, cy = self.capture.width // 2, self.capture.height // 2

        # Center crosshair
        cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (255, 0, 0), 1)
        cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (255, 0, 0), 1)

        if detection:
            x, y, w, h = detection['bbox']
            tx, ty = x + w // 2, y + h // 2
            shift_x, shift_y = tx - cx, ty - cy

            # Bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (tx, ty), 4, (0, 0, 255), -1)
            cv2.line(frame, (cx, cy), (tx, ty), (255, 255, 0), 1)

            # Keypoints
            kp = detection.get('keypoints')
            if kp is not None:
                for px, py, conf in kp:
                    if conf >= self.detector.keypoint_threshold:
                        cv2.circle(frame, (int(px), int(py)), 3, (255, 0, 255), -1)

            # Shift text
            cv2.putText(frame, f"x={shift_x:+d} y={shift_y:+d}", (x, y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Update motors and log
            if self._shift_logger:
                self._shift_logger.log(shift_x, shift_y)
            self.motors.update(shift_x, shift_y)
        else:
            self.motors.stop()

        return frame

    def start_recording(self):
        """Start video recording.

        In 'local' mode: records on Pi 5 via ffmpeg with hardware H.264
        encoding when available, falling back to software H.264 / MJPG.

        In 'camera' mode: triggers hardware H.264 recording on Pi 3B+
        via HTTP API — zero CPU cost on the tracker Pi.
        """
        if self.recording:
            return

        if self.recording_mode == 'camera':
            try:
                req = urllib.request.Request(
                    f"{self._camera_base_url}/record/start",
                    method='POST', data=b'',
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    import json
                    info = json.loads(resp.read())
                self.recording = True
                logger.info(f"Camera-side recording started: {info.get('file')}")
            except Exception as e:
                logger.error(f"Failed to start camera recording: {e}")
                logger.info("Falling back to local recording")
                self._start_local_recording()
        else:
            self._start_local_recording()

    def _start_local_recording(self):
        """Start local ffmpeg-based recording on tracker Pi."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"rec_{ts}.mp4"
        encoder = self.config['tracker'].get('recording_encoder', 'auto')

        self._rec_thread = _RecordingThread(
            str(path), self.capture.width, self.capture.height,
            self.recording_fps, encoder=encoder,
        )
        self._rec_thread.start()
        self.recording = True
        logger.info(f"Recording: {path}  ({self.recording_fps:.0f} fps)")

    def stop_recording(self):
        """Stop video recording."""
        if not self.recording:
            return
        self.recording = False

        if self.recording_mode == 'camera':
            try:
                req = urllib.request.Request(
                    f"{self._camera_base_url}/record/stop",
                    method='POST', data=b'',
                )
                urllib.request.urlopen(req, timeout=5)
                logger.info("Camera-side recording stopped")
            except Exception as e:
                logger.warning(f"Failed to stop camera recording: {e}")

        if self._rec_thread:
            self._rec_thread.stop()
            self._rec_thread = None
        logger.info("Recording stopped")

    def screenshot(self, frame: np.ndarray):
        """Save screenshot."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"screenshot_{ts}.jpg"
        cv2.imwrite(str(path), frame)
        logger.info(f"Screenshot: {path}")

    @property
    def fps(self) -> float:
        """Current frames per second."""
        return self._fps

    def update_fps(self):
        """Update FPS counter. Call once per frame."""
        self._fps_count += 1
        now = time.monotonic()
        if now - self._fps_time >= 1.0:
            self._fps = self._fps_count / (now - self._fps_time)
            self._fps_count = 0
            self._fps_time = now

    def write_frame(self, frame: Optional[np.ndarray]):
        """Feed the latest clean (un-annotated) frame to the recording thread."""
        if frame is not None and self.recording and self._rec_thread:
            self._rec_thread.update_frame(frame)

    def run(self, display: bool = True, auto_record: bool = False):
        """Main processing loop."""
        if not self.connect():
            logger.error("Failed to connect to stream")
            return

        self.running = True
        if auto_record:
            self.start_recording()

        logger.info("Tracking started. Keys: q=quit, r=record, s=screenshot, d=reset, e=EV3")

        try:
            while self.running:
                ret, frame = self.capture.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                rec_frame = frame.copy() if self.recording else None
                self.write_frame(rec_frame)
                annotated, _ = self.process_frame(frame)
                self.update_fps()

                if display:
                    # Status overlay
                    cv2.putText(annotated, f"FPS: {self.fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    ev3_text = "EV3: ON" if self.motors.connected else "EV3: OFF"
                    ev3_color = (0, 255, 0) if self.motors.connected else (0, 0, 255)
                    cv2.putText(annotated, ev3_text, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, ev3_color, 2)

                    if self.tracker.tracker_type:
                        cv2.putText(annotated, f"Tracker: {self.tracker.tracker_type}", (10, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                    if self.recording:
                        cv2.circle(annotated, (self.capture.width - 20, 20), 8, (0, 0, 255), -1)

                    cv2.imshow('Human Tracker', annotated)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        self.stop_recording() if self.recording else self.start_recording()
                    elif key == ord('s'):
                        self.screenshot(annotated)
                    elif key == ord('d'):
                        self.tracker.reset()
                        self.current_detection = None
                        logger.info("Detection reset")
                    elif key == ord('e'):
                        self.motors.disconnect() if self.motors.connected else self.motors.connect()

        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            self.cleanup()

    def cleanup(self):
        """Release all resources."""
        logger.info("Cleaning up...")
        self.running = False
        if self._async_detector:
            self._async_detector.stop()
            self._async_detector = None
        self.motors.disconnect()
        self.stop_recording()
        if self.capture:
            self.capture.stop()
        if self._shift_logger:
            self._shift_logger.flush()
        cv2.destroyAllWindows()


# =============================================================================
# Utilities
# =============================================================================

class ShiftLogger:
    """Batched shift logger to reduce I/O."""

    def __init__(self, path: Path, batch_size: int = 50):
        self.path = path
        self.batch_size = batch_size
        self.buffer = deque(maxlen=batch_size * 2)
        self._lock = Lock()
        self._write_header()

    def _write_header(self):
        with open(self.path, 'w') as f:
            f.write("Position Shifts from Center\n")
            f.write("=" * 40 + "\n\n")

    def log(self, x: int, y: int):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            self.buffer.append(f"{ts} | x={x:+6d} y={y:+6d}\n")
            if len(self.buffer) >= self.batch_size:
                self._flush()

    def _flush(self):
        if self.buffer:
            with open(self.path, 'a') as f:
                f.writelines(self.buffer)
            self.buffer.clear()

    def flush(self):
        with self._lock:
            self._flush()


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Human Tracker for Raspberry Pi')

    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--url', help='Stream URL (overrides config)')
    parser.add_argument('--output-dir', help='Output directory')

    # Detection
    parser.add_argument('--detection-interval', type=int, help='Detection interval (frames)')
    parser.add_argument('--process-scale', type=float, help='Detection scale (0.2-1.0)')
    parser.add_argument('--confidence', type=float, help='Confidence threshold')
    parser.add_argument('--movenet-threads', type=int, help='Inference threads')

    # Display
    parser.add_argument('--no-display', action='store_true', help='Headless mode')
    parser.add_argument('--auto-record', action='store_true', help='Auto-start recording')

    # EV3
    parser.add_argument('--no-ev3', action='store_true', help='Disable EV3')

    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Merge CLI overrides into config (single source of truth)
    if args.url:
        config['tracker']['stream_url'] = args.url
    if args.output_dir:
        config['tracker']['output_dir'] = args.output_dir
    if args.detection_interval:
        config['tracker']['detection']['interval'] = args.detection_interval
    if args.process_scale:
        config['tracker']['detection']['scale'] = args.process_scale
    if args.confidence:
        config['tracker']['detection']['confidence'] = args.confidence
    if args.movenet_threads:
        config['tracker']['movenet']['threads'] = args.movenet_threads
    if args.no_ev3:
        config['ev3']['enabled'] = False

    tracker = HumanTracker(config)
    tracker.run(display=not args.no_display, auto_record=args.auto_record)


if __name__ == "__main__":
    main()
