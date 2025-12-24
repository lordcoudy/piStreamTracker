import argparse
import logging
import os
from tabnanny import verbose
import time
import urllib.request
from collections import deque
from datetime import datetime
from threading import Event, Lock, Thread
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from ev3_usb import EV3_USB

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tracker.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


try:
    cv2.setNumThreads(1)
except Exception:
    pass


class ThreadedVideoCapture:
    
    def __init__(self, src: str, buffer_size: int = 2):
        self.src = src
        self.cap = None
        self.frame = None
        self.ret = False
        self.lock = Lock()
        self.stopped = Event()
        self.thread = None
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 30.0
        self.buffer_size = buffer_size
        
    def start(self) -> bool:
        """Initialize capture and start background thread."""
        # Try multiple backends for Pi compatibility
        for backend in [cv2.CAP_V4L2, cv2.CAP_FFMPEG, cv2.CAP_ANY]:
            try:
                self.cap = cv2.VideoCapture(self.src, backend)
                if self.cap.isOpened():
                    break
            except:
                continue
        
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.src)
            
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.src}")
            return False
        
        # Optimize buffer settings for low latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        
        # Get stream properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        # Read initial frame
        self.ret, self.frame = self.cap.read()
        if not self.ret:
            logger.error("Failed to read initial frame")
            return False
        
        # Start capture thread
        self.stopped.clear()
        self.thread = Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        logger.info(f"Capture started: {self.frame_width}x{self.frame_height} @ {self.fps:.1f} FPS")
        return True
    
    def _capture_loop(self):
        """Background thread for continuous frame capture."""
        while not self.stopped.is_set():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                self.stopped.wait(timeout=0.001)
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get the latest frame (non-blocking).

        Returns a reference to the last captured frame to avoid an expensive
        full-frame copy per read.
        """
        with self.lock:
            return self.ret, self.frame
    
    def read_direct(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.read()
    
    def stop(self):
        """Stop capture thread and release resources."""
        self.stopped.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
    
    def isOpened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()
    
    def __del__(self):
        """Ensure resources are released on garbage collection."""
        self.stop()
    
    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self.frame_width
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.frame_height
        elif prop == cv2.CAP_PROP_FPS:
            return self.fps
        return self.cap.get(prop) if self.cap else 0


class BatchedLogWriter:
    """Batched file writer to reduce I/O overhead."""
    
    def __init__(self, filepath: str, batch_size: int = 50):
        self.filepath = filepath
        self.batch_size = batch_size
        self.buffer = deque(maxlen=batch_size * 2)
        self.lock = Lock()
        self._write_header()
    
    def _write_header(self):
        with open(self.filepath, 'w') as f:
            f.write("Human Position Shifts from Frame Center\n")
            f.write("=" * 50 + "\n")
            f.write("Format: timestamp | x=<shift> ; y=<shift>\n")
            f.write("=" * 50 + "\n\n")
    
    def log(self, shift_x: int, shift_y: int):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = f"{timestamp} | x={shift_x:+6d} ; y={shift_y:+6d}\n"
        
        with self.lock:
            self.buffer.append(entry)
            if len(self.buffer) >= self.batch_size:
                self._flush_buffer()
    
    def _flush_buffer(self):
        if not self.buffer:
            return
        try:
            with open(self.filepath, 'a') as f:
                f.writelines(self.buffer)
            self.buffer.clear()
        except Exception as e:
            logger.error(f"Log write error: {e}")
    
    def flush(self):
        with self.lock:
            self._flush_buffer()


class EV3Controller:
    """Optimized EV3 motor controller with rate limiting."""
    
    def __init__(self, 
                 deadzone_x: int = 50, 
                 deadzone_y: int = 50,
                 speed_factor: float = 1.0, 
                 max_speed: int = 50,
                 invert_x: bool = False, 
                 invert_y: bool = False,
                 command_cooldown: float = 0.5):
        
        self.deadzone_x = deadzone_x
        self.deadzone_y = deadzone_y
        self.speed_factor = min(speed_factor, 2.0)
        self.max_speed = min(max_speed, 100)
        self.invert_x = invert_x
        self.invert_y = invert_y
        
        self.ev3 = None
        self.motor_a = None
        self.motor_b = None
        self.connected = False
        self.last_command_time = 0.0
        self.command_cooldown = command_cooldown
        self.cam_width = 1280
        self.cam_height = 960
        
        # Pre-calculate coefficients
        self._update_coefficients()
        self.connect()
    
    def _update_coefficients(self):
        """Pre-calculate degree coefficients for motor control."""
        # Based on camera FOV (OV5647)
        self.degree_coeff_x = self.cam_width / 128.0
        self.degree_coeff_y = self.cam_height / 96.0
    
    def set_camera_size(self, width: int, height: int):
        """Update camera dimensions and recalculate coefficients."""
        self.cam_width = width
        self.cam_height = height
        self._update_coefficients()
    
    def connect(self) -> bool:
        try:
            logger.info("Connecting to EV3 via USB...")
            self.ev3 = EV3_USB()
            self.motor_a = self.ev3.Motor('a')
            self.motor_b = self.ev3.Motor('b')
            self.connected = True
            
            try:
                self.ev3.Led('green', 'pulse')
            except:
                pass
            
            self.stop_motors()
            logger.info("EV3 connected successfully!")
            return True
            
        except Exception as e:
            logger.error(f"EV3 connection failed: {e}")
            self.connected = False
            self._cleanup()
            return False
    
    def _cleanup(self):
        self.motor_a = None
        self.motor_b = None
        self.ev3 = None
    
    def calculate_motor_params(self, shift: int, deadzone: int, 
                                invert: bool, degree_coeff: float) -> Tuple[Optional[int], int]:
        """Calculate motor degree and speed. Returns (degree, speed) or (None, 0)."""
        if abs(shift) < deadzone:
            return None, 0
        
        speed = int(min(abs(shift) / 100.0 * self.speed_factor * 10 + 5, self.max_speed))
        degree = int((-shift if invert else shift) / degree_coeff)
        
        return degree, speed
    
    def update_motors(self, shift_x: int, shift_y: int):
        """Update motors with rate limiting."""
        if not self.connected:
            return
        
        current_time = time.monotonic()
        if current_time - self.last_command_time < self.command_cooldown:
            return
        
        try:
            degree_a, speed_a = self.calculate_motor_params(
                shift_x, self.deadzone_x, self.invert_x, self.degree_coeff_x
            )
            degree_b, speed_b = self.calculate_motor_params(
                shift_y, self.deadzone_y, self.invert_y, self.degree_coeff_y
            )
            
            if degree_a is not None and speed_a > 0:
                self.motor_a.run_to(degrees=degree_a, speed=speed_a)
            else:
                try:
                    self.motor_a.stop()
                except:
                    pass
            
            if degree_b is not None and speed_b > 0:
                self.motor_b.run_to(degrees=degree_b, speed=speed_b)
            else:
                try:
                    self.motor_b.stop()
                except:
                    pass
            
            self.last_command_time = current_time
            
        except Exception as e:
            logger.debug(f"Motor update error: {e}")
    
    def stop_motors(self):
        if not self.connected:
            return
        try:
            if self.motor_a:
                self.motor_a.stop()
        except:
            pass
        try:
            if self.motor_b:
                self.motor_b.stop()
        except:
            pass
    
    def disconnect(self):
        if self.connected:
            self.stop_motors()
            try:
                if self.ev3:
                    self.ev3.Led('orange', 'static')
            except:
                pass
            self._cleanup()
            self.connected = False
            logger.info("EV3 disconnected")
    
    def __del__(self):
        """Ensure motors stop on garbage collection."""
        self.disconnect()


class MoveNetDetector:
    """
    Optimized MoveNet Lightning detector for Raspberry Pi.
    """
    
    # MoveNet Lightning keypoint names
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 num_threads: Optional[int] = None,
                 confidence_threshold: float = 0.5,
                 keypoint_threshold: float = 0.3):
        
        self.model_path = model_path
        self.num_threads = num_threads or max(1, (os.cpu_count() or 2) // 2)
        self.confidence_threshold = confidence_threshold
        self.keypoint_threshold = keypoint_threshold
        
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_size = (192, 192)
        
        self._input_buffer = None
        self._resized_buffer = None
        self._keypoints_buffer = np.empty((17, 3), dtype=np.float32)
        
        self._initialize()
    
    def _initialize(self):
        """Initialize the TFLite interpreter."""
        try:
            model_path = self._ensure_model()
            if not model_path:
                raise RuntimeError("Model not available")
            
            # Try tflite-runtime first (faster on Pi)
            interpreter_cls = None
            try:
                from tflite_runtime.interpreter import Interpreter
                interpreter_cls = Interpreter
                logger.info("Using tflite-runtime interpreter")
            except ImportError:
                try:
                    from tensorflow.lite.python.interpreter import Interpreter
                    interpreter_cls = Interpreter
                    logger.info("Using TensorFlow Lite interpreter")
                except ImportError:
                    raise RuntimeError("No TFLite interpreter available")
            
            # Create interpreter with optimized settings
            try:
                self.interpreter = interpreter_cls(
                    model_path=model_path, 
                    num_threads=self.num_threads
                )
            except TypeError:
                self.interpreter = interpreter_cls(model_path=model_path)
            
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Get input size from model
            shape = self.input_details[0]['shape']
            if len(shape) >= 3:
                # Model shape is typically [1, height, width, 3]
                self.input_size = (int(shape[1]), int(shape[2]))

            input_h, input_w = self.input_size
            
            # Pre-allocate input buffer
            dtype = self.input_details[0]['dtype']
            self._input_buffer = np.zeros(
                (1, input_h, input_w, 3),
                dtype=dtype
            )

            # Pre-allocate resized RGB buffer (uint8) for a single resize + color conversion.
            self._resized_buffer = np.empty((input_h, input_w, 3), dtype=np.uint8)

            # Optional float workspace for quantized models (allocated once).
            self._float_buffer = None
            quant = self.input_details[0].get('quantization', (0.0, 0))
            if dtype in (np.uint8, np.int8) and quant and quant[0] and quant[0] > 0:
                self._float_buffer = np.empty((input_h, input_w, 3), dtype=np.float32)
            
            logger.info(f"MoveNet initialized: input={self.input_size}, threads={self.num_threads}")
            
        except Exception as e:
            logger.error(f"MoveNet initialization failed: {e}")
            self.interpreter = None
    
    def _ensure_model(self) -> Optional[str]:
        """Ensure model file exists, download if needed."""
        if self.model_path and os.path.exists(self.model_path):
            return self.model_path
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "movenet_lightning.tflite")
        
        if os.path.exists(model_path):
            self.model_path = model_path
            return model_path
        
        urls = [
            # Direct Kaggle/Google Storage mirror (more reliable)
            "https://storage.googleapis.com/tfhub-lite-models/google/lite-model/movenet/singlepose/lightning/tflite/float16/4.tflite",
            # TFHub with explicit format
            "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite",
        ]
        
        for url in urls:
            try:
                logger.info(f"Downloading MoveNet Lightning model from {url[:50]}...")
                
                # Create request with headers to handle redirects properly
                req = urllib.request.Request(
                    url,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (compatible; PiTracker/1.0)',
                        'Accept': '*/*',
                    }
                )
                
                # Download with SSL context for Pi compatibility
                import ssl
                ctx = ssl.create_default_context()
                
                with urllib.request.urlopen(req, context=ctx, timeout=60) as response:
                    # Check if response is valid TFLite model (starts with specific bytes)
                    data = response.read()
                    
                    if len(data) < 1000:
                        logger.warning(f"Downloaded file too small ({len(data)} bytes), trying next URL...")
                        continue
                    
                    with open(model_path, 'wb') as f:
                        f.write(data)
                
                self.model_path = model_path
                logger.info(f"Model downloaded to {model_path} ({len(data) // 1024} KB)")
                return model_path
                
            except Exception as e:
                logger.warning(f"Download failed from {url[:50]}: {e}")
                continue
        
        logger.error("All model download attempts failed. Please download manually.")
        logger.error("Visit: https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/4")
        return None
    
    def detect(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect single person in frame.
        
        Args:
            frame: BGR image (already scaled for processing)
        
        Returns:
            Detection dict with 'bbox', 'confidence', 'keypoints' or None
        """
        if self.interpreter is None:
            return None
        
        frame_h, frame_w = frame.shape[:2]

        input_h, input_w = self.input_size

        # Resize to model input (OpenCV expects dsize=(width, height)).
        cv2.resize(
            frame,
            (input_w, input_h),
            dst=self._resized_buffer,
            interpolation=cv2.INTER_LINEAR,
        )

        # Convert BGR->RGB in-place (keeps a single buffer alive across frames)
        resized = self._resized_buffer
        cv2.cvtColor(resized, cv2.COLOR_BGR2RGB, resized)
        
        # Prepare input tensor
        input_detail = self.input_details[0]
        dtype = input_detail['dtype']

        input_tensor = self._input_buffer
        input_view = input_tensor[0]

        if dtype == np.float32:
            # Normalize to [0, 1] without per-frame allocations.
            input_view[...] = resized
            input_view *= (1.0 / 255.0)
        elif dtype == np.uint8:
            scale, zero_point = input_detail.get('quantization', (0.0, 0))
            if scale and scale > 0:
                # Quantize: q = x/255/scale + zero_point
                float_buf = self._float_buffer
                if float_buf is None:
                    float_buf = resized.astype(np.float32)
                else:
                    float_buf[...] = resized
                float_buf *= (1.0 / 255.0)
                float_buf /= scale
                float_buf += zero_point
                np.clip(float_buf, 0, 255, out=float_buf)
                input_view[...] = float_buf
            else:
                input_view[...] = resized
        elif dtype == np.int8:
            scale, zero_point = input_detail.get('quantization', (0.0, 0))
            if scale and scale > 0:
                float_buf = self._float_buffer
                if float_buf is None:
                    float_buf = resized.astype(np.float32)
                else:
                    float_buf[...] = resized
                float_buf *= (1.0 / 255.0)
                float_buf /= scale
                float_buf += zero_point
                np.clip(float_buf, -128, 127, out=float_buf)
                input_view[...] = float_buf
            else:
                # Fallback: common convention is symmetric int8 with zero_point=-128.
                # Map uint8 [0,255] -> int8 [-128,127] without wraparound.
                if self._float_buffer is not None:
                    self._float_buffer[...] = resized
                    self._float_buffer -= 128.0
                    input_view[...] = self._float_buffer
                else:
                    input_view[...] = resized.astype(np.int16) - 128
        else:
            input_view[...] = resized
        
        # Run inference
        try:
            self.interpreter.set_tensor(input_detail['index'], input_tensor)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        except Exception as e:
            logger.debug(f"Inference error: {e}")
            return None
        
        # Dequantize output if needed
        output_detail = self.output_details[0]
        quant = output_detail.get('quantization', (0.0, 0))
        scale, zero_point = quant
        if scale and scale > 0:
            output_data = (output_data.astype(np.float32) - zero_point) * scale
        else:
            output_data = output_data.astype(np.float32)
        
        # Parse keypoints [1, 1, 17, 3] -> [17, 3] (y, x, confidence)
        raw_keypoints = output_data[0, 0, :, :]
        scores = raw_keypoints[:, 2]
        
        # Filter valid keypoints
        valid_mask = scores >= self.keypoint_threshold
        if not np.any(valid_mask):
            return None
        
        # Scale and swap keypoints into pre-allocated buffer: [y, x, conf] -> [x, y, conf]
        keypoints = self._keypoints_buffer
        keypoints[:, 0] = raw_keypoints[:, 1] * frame_w  # x
        keypoints[:, 1] = raw_keypoints[:, 0] * frame_h  # y
        keypoints[:, 2] = scores  # confidence
        
        # Get bounding box from valid keypoints
        xs_valid = keypoints[valid_mask, 0]
        ys_valid = keypoints[valid_mask, 1]
        
        x_min = max(0, int(np.min(xs_valid)))
        y_min = max(0, int(np.min(ys_valid)))
        x_max = min(frame_w - 1, int(np.max(xs_valid)))
        y_max = min(frame_h - 1, int(np.max(ys_valid)))
        
        width = max(1, x_max - x_min)
        height = max(1, y_max - y_min)
        
        # Calculate mean confidence of valid keypoints
        confidence = float(np.mean(scores[valid_mask]))
        if confidence < self.confidence_threshold:
            return None
        
        # Return detection with keypoints copy (buffer is reused across frames)
        return {
            'bbox': (x_min, y_min, width, height),
            'confidence': min(confidence, 1.0),
            'keypoints': keypoints.copy()
        }
    
    @property
    def is_ready(self) -> bool:
        return self.interpreter is not None


class OptimizedTracker:
    
    def __init__(self,
                 stream_url: str,
                 output_dir: str = "recordings",
                 verbose: bool = False,
                 confidence_threshold: float = 0.5,
                 detection_interval: int = 8,
                 process_scale: float = 0.5,
                 keypoint_threshold: float = 0.3,
                 movenet_model_path: Optional[str] = None,
                 movenet_threads: Optional[int] = None,
                 ev3_deadzone_x: int = 50,
                 ev3_deadzone_y: int = 50,
                 ev3_speed_factor: float = 1.0,
                 ev3_max_speed: int = 50,
                 ev3_invert_x: bool = False,
                 ev3_invert_y: bool = False,
                 ev3_command_cooldown: float = 0.5):
        
        self.stream_url = stream_url
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold
        self.detection_interval = max(1, detection_interval)
        self.process_scale = max(0.2, min(1.0, process_scale))
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Video capture
        self.cap = None
        self.frame_width = 0
        self.frame_height = 0
        self.frame_center_x = 0
        self.frame_center_y = 0
        
        # State
        self.is_running = False
        self.is_recording = False
        self.video_writer = None
        self.output_filename = None
        self.frame_count = 0
        
        # Tracking state
        self.tracker = None
        self.tracked_human = None
        self.tracking_enabled = True
        self._tracker_type = None
        
        # Pre-calculate inverse scale
        self.scale_inv = 1.0 / self.process_scale
        
        # Pre-allocated buffer for scaled detection frame (set after connect)
        self._scaled_frame_buffer = None
        
        # Initialize detector
        self.detector = MoveNetDetector(
            model_path=movenet_model_path,
            num_threads=movenet_threads,
            confidence_threshold=confidence_threshold,
            keypoint_threshold=keypoint_threshold
        )
        
        # Initialize EV3 controller
        self.ev3 = EV3Controller(
            deadzone_x=ev3_deadzone_x,
            deadzone_y=ev3_deadzone_y,
            speed_factor=ev3_speed_factor,
            max_speed=ev3_max_speed,
            invert_x=ev3_invert_x,
            invert_y=ev3_invert_y,
            command_cooldown=ev3_command_cooldown
        )
        
        # Shift logging
        if verbose:
            logger.setLevel(logging.DEBUG)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(output_dir, f"human_shifts_{timestamp}.txt")
            self.shift_logger = BatchedLogWriter(log_path, batch_size=30)
        
        # FPS tracking
        self._fps_frames = 0
        self._fps_start = time.monotonic()
        self._current_fps = 0.0
        
        logger.info(f"Tracker initialized: scale={self.process_scale}, interval={self.detection_interval}")
    
    def connect(self) -> bool:
        """Connect to video stream."""
        self.cap = ThreadedVideoCapture(self.stream_url, buffer_size=2)
        
        if not self.cap.start():
            logger.error("Failed to connect to stream")
            return False
        
        self.frame_width = self.cap.frame_width
        self.frame_height = self.cap.frame_height
        self.frame_center_x = self.frame_width // 2
        self.frame_center_y = self.frame_height // 2
        
        # Update EV3 with camera dimensions
        self.ev3.set_camera_size(self.frame_width, self.frame_height)
        
        # Pre-allocate scaled frame buffer for detection
        if self.process_scale < 1.0:
            scaled_h = int(self.frame_height * self.process_scale)
            scaled_w = int(self.frame_width * self.process_scale)
            self._scaled_frame_buffer = np.empty((scaled_h, scaled_w, 3), dtype=np.uint8)
        
        logger.info(f"Connected: {self.frame_width}x{self.frame_height}, center=({self.frame_center_x}, {self.frame_center_y})")
        return True
    
    def _init_tracker(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """Initialize OpenCV tracker with fastest available type."""
        x, y, w, h = bbox
        
        # Validate bbox
        if w < 10 or h < 10:
            return False
        
        # Clip to frame bounds
        fh, fw = frame.shape[:2]
        x = max(0, min(x, fw - 1))
        y = max(0, min(y, fh - 1))
        w = min(w, fw - x)
        h = min(h, fh - y)
        
        # Try trackers in order of speed (MOSSE is fastest)
        tracker_types = ["MOSSE", "KCF", "CSRT"]
        
        for tracker_name in tracker_types:
            try:
                # Try legacy API first (OpenCV 4.5+)
                if hasattr(cv2, 'legacy'):
                    create_fn = getattr(cv2.legacy, f'Tracker{tracker_name}_create', None)
                    if callable(create_fn):
                        tracker = create_fn()
                        if tracker.init(frame, (x, y, w, h)):
                            self.tracker = tracker
                            self._tracker_type = tracker_name
                            return True
                
                # Try standard API
                create_fn = getattr(cv2, f'Tracker{tracker_name}_create', None)
                if callable(create_fn):
                    tracker = create_fn()
                    if tracker.init(frame, (x, y, w, h)):
                        self.tracker = tracker
                        self._tracker_type = tracker_name
                        return True
                        
            except Exception:
                continue
        
        self.tracker = None
        self.tracking_enabled = False
        return False
    
    def _update_tracker(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Update tracker and return bbox if successful."""
        if self.tracker is None:
            return None
        
        try:
            success, bbox = self.tracker.update(frame)
            if success:
                return tuple(int(v) for v in bbox)
        except Exception:
            pass
        
        self.tracker = None
        return None
    
    def _run_detection(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Run detection on scaled frame and return result in original coordinates."""
        # Scale down for faster detection (INTER_AREA is optimal for downscaling)
        if self.process_scale < 1.0 and self._scaled_frame_buffer is not None:
            cv2.resize(frame, (self._scaled_frame_buffer.shape[1], self._scaled_frame_buffer.shape[0]),
                       dst=self._scaled_frame_buffer, interpolation=cv2.INTER_AREA)
            small_frame = self._scaled_frame_buffer
        else:
            small_frame = frame
        
        # Run detection
        detection = self.detector.detect(small_frame)
        
        if detection is None:
            return None
        
        # Scale bbox back to original coordinates
        x, y, w, h = detection['bbox']
        x = int(x * self.scale_inv)
        y = int(y * self.scale_inv)
        w = int(w * self.scale_inv)
        h = int(h * self.scale_inv)
        
        # Clip to frame bounds
        x = max(0, min(x, self.frame_width - 1))
        y = max(0, min(y, self.frame_height - 1))
        w = min(w, self.frame_width - x)
        h = min(h, self.frame_height - y)
        
        # Scale keypoints (modify in-place since detect() returns fresh array)
        keypoints = detection['keypoints']
        if keypoints is not None:
            keypoints[:, :2] *= self.scale_inv  # Scale x and y together
        
        return {
            'bbox': (x, y, w, h),
            'confidence': detection['confidence'],
            'keypoints': keypoints
        }
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Dict]]:
        """Process single frame with tracking and detection."""
        self.frame_count += 1
        
        tracked = None
        need_detection = False
        
        # Try tracker update first (fast path)
        if self.tracker is not None:
            bbox = self._update_tracker(frame)
            if bbox is not None:
                tracked = {
                    'bbox': bbox,
                    'confidence': self.tracked_human.get('confidence', 0.5) if self.tracked_human else 0.5,
                    'keypoints': self.tracked_human.get('keypoints') if self.tracked_human else None
                }
                # Periodic re-detection to correct drift
                if self.frame_count % self.detection_interval == 0:
                    need_detection = True
            else:
                need_detection = True
        else:
            need_detection = True
        
        # Run detection if needed
        if need_detection:
            detection = self._run_detection(frame)
            if detection is not None:
                tracked = detection
                # Re-initialize tracker with new detection
                if self.tracking_enabled:
                    self._init_tracker(frame, detection['bbox'])
        
        self.tracked_human = tracked
        
        # Draw visualization and update motors
        annotated = self._draw_overlay(frame, tracked)
        
        return annotated, tracked
    
    def _draw_overlay(self, frame: np.ndarray, tracked: Optional[Dict]) -> np.ndarray:
        """Draw tracking overlay on frame."""
        # Draw center crosshair
        cv2.line(frame, (self.frame_center_x - 8, self.frame_center_y),
                 (self.frame_center_x + 8, self.frame_center_y), (255, 0, 0), 1)
        cv2.line(frame, (self.frame_center_x, self.frame_center_y - 8),
                 (self.frame_center_x, self.frame_center_y + 8), (255, 0, 0), 1)
        
        if tracked:
            x, y, w, h = tracked['bbox']
            
            # Calculate center and shift
            bbox_cx = x + w // 2
            bbox_cy = y + h // 2
            shift_x = bbox_cx - self.frame_center_x
            shift_y = bbox_cy - self.frame_center_y
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(frame, (bbox_cx, bbox_cy), 4, (0, 0, 255), -1)
            
            # Draw line to center
            cv2.line(frame, (self.frame_center_x, self.frame_center_y),
                    (bbox_cx, bbox_cy), (255, 255, 0), 1)
            
            # Draw keypoints if available
            keypoints = tracked.get('keypoints')
            if keypoints is not None:
                for kp_x, kp_y, kp_conf in keypoints:
                    if kp_conf >= self.detector.keypoint_threshold:
                        cv2.circle(frame, (int(kp_x), int(kp_y)), 3, (255, 0, 255), -1)
            
            # Display shift text
            shift_text = f"x={shift_x:+d} y={shift_y:+d}"
            cv2.putText(frame, shift_text, (x, y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Log shift and update motors
            if self.verbose: self.shift_logger.log(shift_x, shift_y)
            self.ev3.update_motors(shift_x, shift_y)
        else:
            self.ev3.stop_motors()
        
        return frame
    
    def start_recording(self):
        """Start video recording."""
        if self.is_recording or self.cap is None:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fps = min(self.cap.fps, 30.0)  # Cap at 30 FPS for recording
        
        # Try codecs in order of compatibility
        codecs = [
            ('MJPG', '.avi'),
            ('mp4v', '.mp4'),
            ('XVID', '.avi'),
        ]
        
        for fourcc_str, ext in codecs:
            try:
                self.output_filename = os.path.join(self.output_dir, f"rec_{timestamp}{ext}")
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                self.video_writer = cv2.VideoWriter(
                    self.output_filename, fourcc, fps,
                    (self.frame_width, self.frame_height)
                )
                if self.video_writer.isOpened():
                    self.is_recording = True
                    logger.info(f"Recording started: {self.output_filename}")
                    return
            except Exception:
                continue
        
        logger.error("Failed to initialize video recording")
    
    def stop_recording(self):
        """Stop video recording."""
        if not self.is_recording:
            return
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        self.is_recording = False
        logger.info(f"Recording stopped: {self.output_filename}")
    
    def _handle_key(self, key: int, frame: np.ndarray) -> bool:
        """Handle keyboard input. Returns True to quit."""
        if key == ord('q'):
            return True
        elif key == ord('r'):
            if self.is_recording:
                self.stop_recording()
            else:
                self.start_recording()
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.output_dir, f"screenshot_{timestamp}.jpg")
            cv2.imwrite(path, frame)
            logger.info(f"Screenshot: {path}")
        elif key == ord('d'):
            self.tracker = None
            self.tracked_human = None
            logger.info("Detection reset")
        elif key == ord('e'):
            if self.ev3.connected:
                self.ev3.disconnect()
            else:
                self.ev3.connect()
        return False
    
    def run(self, display: bool = True, auto_record: bool = False):
        """Main processing loop."""
        if not self.connect():
            return
        
        self.is_running = True
        
        if auto_record:
            self.start_recording()
        
        logger.info("Tracking started. Keys: q=quit, r=record, s=screenshot, d=reset, e=EV3")
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                
                # Process frame
                annotated, _ = self.process_frame(frame)
                
                # Update FPS counter
                self._fps_frames += 1
                now = time.monotonic()
                elapsed = now - self._fps_start
                if elapsed >= 1.0:
                    self._current_fps = self._fps_frames / elapsed
                    self._fps_frames = 0
                    self._fps_start = now
                
                # Record if enabled
                if self.is_recording and self.video_writer:
                    self.video_writer.write(annotated)
                
                if display:
                    # Draw FPS and status
                    cv2.putText(annotated, f"FPS: {self._current_fps:.1f}",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    ev3_status = "EV3: ON" if self.ev3.connected else "EV3: OFF"
                    ev3_color = (0, 255, 0) if self.ev3.connected else (0, 0, 255)
                    cv2.putText(annotated, ev3_status, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, ev3_color, 2)
                    
                    if self._tracker_type:
                        cv2.putText(annotated, f"Tracker: {self._tracker_type}",
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    
                    cv2.imshow('Human Tracker', annotated)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if self._handle_key(key, annotated):
                        break
                        
        except KeyboardInterrupt:
            logger.info("Interrupted")
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release all resources."""
        logger.info("Cleaning up...")
        self.is_running = False
        
        self.ev3.disconnect()
        
        if self.is_recording:
            self.stop_recording()
        
        if self.cap:
            self.cap.stop()
        
        if self.verbose: self.shift_logger.flush()
        cv2.destroyAllWindows()
        
        logger.info("Cleanup complete")


def main():
    parser = argparse.ArgumentParser(
        description='Optimized Human Tracker for Raspberry Pi 3B/5'
    )
    
    # Stream settings
    parser.add_argument('--url', default='http://192.168.100.1:8000/stream',
                        help='Video stream URL')
    parser.add_argument('--output-dir', default='recordings',
                        help='Output directory')
    parser.add_argument('--verbose', default='False', action='store_true',
                        help='Enable verbose logging')
    # Detection settings
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
                        help='Detection confidence threshold (0.0-1.0)')
    parser.add_argument('--detection-interval', type=int, default=8,
                        help='Run detection every N frames (higher=faster)')
    parser.add_argument('--process-scale', type=float, default=0.5,
                        help='Frame scale for detection (0.2-1.0, lower=faster)')
    parser.add_argument('--keypoint-threshold', type=float, default=0.3,
                        help='Keypoint confidence threshold')
    
    # MoveNet settings
    parser.add_argument('--movenet-model', type=str, default=None,
                        help='Path to MoveNet TFLite model')
    parser.add_argument('--movenet-threads', type=int, default=None,
                        help='Inference threads (default: half CPU cores)')
    
    # Display settings
    parser.add_argument('--no-display', action='store_true',
                        help='Run headless (no video display)')
    parser.add_argument('--no-auto-record', action='store_true',
                        help='Disable auto-recording on start')
    
    # EV3 settings
    parser.add_argument('--ev3-deadzone-x', type=int, default=90,
                        help='Horizontal deadzone in pixels')
    parser.add_argument('--ev3-deadzone-y', type=int, default=90,
                        help='Vertical deadzone in pixels')
    parser.add_argument('--ev3-speed-factor', type=float, default=1.0,
                        help='Motor speed multiplier (0.1-2.0)')
    parser.add_argument('--ev3-max-speed', type=int, default=50,
                        help='Maximum motor speed (1-100)')
    parser.add_argument('--ev3-invert-x', action='store_true',
                        help='Invert horizontal direction')
    parser.add_argument('--ev3-invert-y', action='store_true',
                        help='Invert vertical direction')
    parser.add_argument('--ev3-cooldown', type=float, default=0.5,
                        help='Motor command cooldown in seconds')
    
    args = parser.parse_args()
    
    tracker = OptimizedTracker(
        stream_url=args.url,
        output_dir=args.output_dir,
        verbose=args.verbose,
        confidence_threshold=args.confidence_threshold,
        detection_interval=args.detection_interval,
        process_scale=args.process_scale,
        keypoint_threshold=args.keypoint_threshold,
        movenet_model_path=args.movenet_model,
        movenet_threads=args.movenet_threads,
        ev3_deadzone_x=args.ev3_deadzone_x,
        ev3_deadzone_y=args.ev3_deadzone_y,
        ev3_speed_factor=args.ev3_speed_factor,
        ev3_max_speed=args.ev3_max_speed,
        ev3_invert_x=args.ev3_invert_x,
        ev3_invert_y=args.ev3_invert_y,
        ev3_command_cooldown=args.ev3_cooldown
    )
    
    tracker.run(
        display=not args.no_display,
        auto_record=not args.no_auto_record
    )


if __name__ == "__main__":
    main()
