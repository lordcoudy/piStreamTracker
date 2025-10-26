import argparse
import math
import logging
import os
import time
from datetime import datetime
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
import socket

import cv2
import numpy as np

from ev3_usb import EV3_USB

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stream.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EV3Controller:
    """Controller for EV3 motors to track human position"""
    def __init__(self, 
                 deadzone_x=50, deadzone_y=50,
                 speed_factor=1.0, max_speed=10,
                 invert_x=False, invert_y=False):
        self.deadzone_x = deadzone_x
        self.deadzone_y = deadzone_y
        self.speed_factor = speed_factor
        self.max_speed = max_speed
        self.invert_x = invert_x
        self.invert_y = invert_y

        self.ev3 = None
        self.motor_a = None  # Horizontal control
        self.motor_b = None  # Vertical control
        self.connected = False
        self.last_command_time = 0
        self.command_cooldown = 15  # Minimum time between commands (seconds)

        self.connect()

    def connect(self):
        """Connect to EV3 brick via USB"""
        try:
            logger.info(f"Connecting to EV3 via USB...")

            # Initialize EV3 connection
            self.ev3 = EV3_USB()

            # Try to initialize motors with error handling
            logger.info("Initializing Motor A (Horizontal)...")
            self.motor_a = self.ev3.Motor('a') 
            logger.info("Initializing Motor B (Vertical)...")
            self.motor_b = self.ev3.Motor('b')

            self.connected = True
            logger.info("EV3 connected successfully!")
            logger.info("Motor A (Port A): Horizontal control")
            logger.info("Motor B (Port B): Vertical control")

            try:
                self.ev3.Led('green', 'pulse')
                logger.info("LED set to green")
            except Exception as e:
                logger.warning(f"Could not set LED: {e}")

            self.stop_motors()

            return

        except Exception as e:
            logger.error(f"Failed to connect to EV3: {e}")
            logger.error("Please check:")
            logger.error("  1. EV3 is turned on")
            logger.error("  2. EV3 is connected via USB")
            logger.error("  3. Motors are connected to ports A and B")
            self.connected = False
            self.cleanup_failed_connection()

    def cleanup_failed_connection(self):
        try:
            if self.motor_a:
                self.motor_a = None
            if self.motor_b:
                self.motor_b = None
            if self.ev3:
                self.ev3 = None
        except:
            pass

    def calculate_motor_direction_speed(self, shift, deadzone, invert=False):
        """
        Calculate motor direction and speed based on shift from center
        Returns: (direction, speed) where direction is 1 or -1
        """
        if abs(shift) < deadzone:
            return None, 0

        direction = 1 if shift > 0 else -1
        if invert:
            direction = -direction

        speed = min(abs(shift) / 100.0 * self.speed_factor, self.max_speed)

        if speed > 0:
            speed = max(speed, 2)  # Minimum speed to overcome motor friction

        return direction, int(speed)

    def update_motors(self, shift_x, shift_y):
        """Update motor positions based on human position shift"""
        if not self.connected or self.ev3 is None:
            return

        current_time = time.time()
        if current_time - self.last_command_time < self.command_cooldown:
            return

        try:
            dir_a, speed_a = self.calculate_motor_direction_speed(
                shift_x / 5, self.deadzone_x, self.invert_x
            )

            dir_b, speed_b = self.calculate_motor_direction_speed(
                shift_y / 50, self.deadzone_y, self.invert_y
            )

            if dir_a is not None and speed_a > 0:
                try:
                    self.motor_a.run(direction=dir_a, speed=speed_a)
                    logger.debug(f"Motor A: dir={dir_a}, speed={speed_a}, shift_x={shift_x}")
                except Exception as e:
                    logger.error(f"Motor A error: {e}")
            else:
                try:
                    self.motor_a.stop()
                except Exception as e:
                    logger.debug(f"Motor A stop error: {e}")

            if dir_b is not None and speed_b > 0:
                try:
                    self.motor_b.run(direction=dir_b, speed=speed_b)
                    logger.debug(f"Motor B: dir={dir_b}, speed={speed_b}, shift_y={shift_y}")
                except Exception as e:
                    logger.error(f"Motor B error: {e}")
            else:
                try:
                    self.motor_b.stop()
                except Exception as e:
                    logger.debug(f"Motor B stop error: {e}")

            self.last_command_time = current_time

        except Exception as e:
            logger.error(f"Error updating motors: {e}")

    def stop_motors(self):
        if not self.connected:
            return

        try:
            # logger.info("Stopping all EV3 motors")
            if self.motor_a:
                try:
                    self.motor_a.stop()
                except Exception as e:
                    logger.debug(f"Error stopping motor A: {e}")
            if self.motor_b:
                try:
                    self.motor_b.stop()
                except Exception as e:
                    logger.debug(f"Error stopping motor B: {e}")
        except Exception as e:
            logger.error(f"Error stopping motors: {e}")

    def disconnect(self):
        """Disconnect from EV3"""
        if self.connected:
            self.stop_motors()

            # Set LED to orange to indicate disconnection
            try:
                if self.ev3:
                    self.ev3.Led('orange', 'static')
                    logger.info("LED set to orange")
            except Exception as e:
                logger.debug(f"Could not set LED on disconnect: {e}")

            # Clean up references
            try:
                self.motor_a = None
                self.motor_b = None
                self.ev3 = None
            except:
                pass

            self.connected = False
            logger.info("EV3 disconnected")



class MJPEGTracker:
    def __init__(self, stream_url, output_dir="recordings",
                 detection_method="haarcascade", confidence_threshold=0.5,
                 detection_interval=10, process_scale=0.4,
                 ev3_deadzone_x=50, ev3_deadzone_y=50,
                 ev3_speed_factor=1.0, ev3_max_speed=30,
                 ev3_invert_x=False, ev3_invert_y=False):
        self.stream_url = stream_url
        self.output_dir = output_dir
        self.detection_method = detection_method
        self.confidence_threshold = confidence_threshold
        self.detection_interval = detection_interval  # Run detection every N frames
        self.process_scale = process_scale  # Scale factor for processing

        os.makedirs(output_dir, exist_ok=True)

        self.cap = None
        self.is_recording = False
        self.is_running = False
        self.video_writer = None
        self.output_filename = None

        # Track only ONE human with highest confidence
        self.tracker = None
        self.tracked_human = None  # Store the tracked human info
        self.tracking_supported = None  # Discover after first attempt
        self._tracker_warned = False    # Log missing support once

        # Frame counter for detection interval
        self.frame_count = 0
        self.last_detection_time = 0

        # Frame center coordinates (will be set when stream connects)
        self.frame_center_x = 0
        self.frame_center_y = 0

        # Text file for shift logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.shift_log_file = os.path.join(output_dir, f"human_shifts_{timestamp}.txt")

        # Initialize shift log file
        with open(self.shift_log_file, 'w') as f:
            f.write("Human Position Shifts from Frame Center\n")
            f.write("=" * 50 + "\n")
            f.write("Format: timestamp | x=<shift> ; y=<shift>\n")
            f.write("=" * 50 + "\n\n")

        # Initialize EV3 controller
        self.ev3_controller = None
        self.ev3_controller = EV3Controller(
                deadzone_x=ev3_deadzone_x,
                deadzone_y=ev3_deadzone_y,
                speed_factor=ev3_speed_factor,
                max_speed=ev3_max_speed,
                invert_x=ev3_invert_x,
                invert_y=ev3_invert_y
            )

        self.initialize_detection()
        try:
            logger.info(f"OpenCV version: {cv2.__version__}")
        except Exception:
            pass
        logger.info(f"MJPEG Tracker initialized with URL: {stream_url}")
        logger.info(f"Detection interval: every {detection_interval} frames")
        logger.info(f"Process scale: {process_scale}")
        logger.info(f"Shift log file: {self.shift_log_file}")
        if self.ev3_controller and self.ev3_controller.connected:
            logger.info(f"EV3 control enabled")

    def initialize_detection(self):
        try:
            if self.detection_method == "haarcascade":
                # Use upper body cascade for better performance
                self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
                if self.body_cascade.empty():
                    self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
                logger.info("Haar Cascade initialized successfully")
            elif self.detection_method == "hog":
                self.hog = cv2.HOGDescriptor()
                self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                logger.info("HOG people detector initialized successfully")
            else:
                # Default to Haar Cascade (faster on RPi)
                self.detection_method = "haarcascade"
                self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
                if self.body_cascade.empty():
                    self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
                logger.info("Default Haar Cascade initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing detection: {e}")

    def connect_to_stream(self):
        try:
            self.cap = cv2.VideoCapture(self.stream_url)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Calculate frame center
                self.frame_center_x = width // 2
                self.frame_center_y = height // 2

                logger.info(f"Connected to stream: {width}x{height} @ {fps} FPS")
                logger.info(f"Frame center: ({self.frame_center_x}, {self.frame_center_y})")
                return True
            else:
                logger.error("Failed to connect to MJPEG stream")
                return False
        except Exception as e:
            logger.error(f"Error connecting to stream: {e}")
            return False

    def detect_humans_hog(self, frame):
        """Detect humans using HOG descriptor - optimized for single detection"""
        # Detect people with larger stride for speed
        boxes, weights = self.hog.detectMultiScale(frame, winStride=(16, 16),
                                                    padding=(8, 8), scale=1.1)

        if len(boxes) == 0:
            return None

        # Find human with maximum confidence
        # weights can be shape (N,) or (N,1) - normalize accordingly
        w_arr = np.array(weights).reshape(-1)
        max_idx = int(np.argmax(w_arr))
        x, y, w, h = boxes[max_idx]
        # Map SVM score to probability-like confidence [0,1]
        svm_score = float(w_arr[max_idx])
        confidence = 1.0 / (1.0 + math.exp(-svm_score))

        if confidence >= self.confidence_threshold:
            return {
                'bbox': (int(x), int(y), int(w), int(h)),
                'label': 'human',
                'confidence': confidence
            }
        return None

    def detect_humans_haarcascade(self, frame):
        """Detect humans using Haar Cascade - optimized for single detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optimize parameters for speed
        bodies = None
        level_weights = None

        # Prefer APIs that return weights to estimate confidence
        try:
            if hasattr(self.body_cascade, 'detectMultiScale3'):
                bodies, rejectLevels, level_weights = self.body_cascade.detectMultiScale3(
                    gray,
                    scaleFactor=1.5,
                    minNeighbors=3,
                    flags=0,
                    minSize=(30, 30),
                    maxSize=()
                )
        except Exception:
            bodies = None
            level_weights = None

        if bodies is None:
            try:
                if hasattr(self.body_cascade, 'detectMultiScale2'):
                    bodies, level_weights = self.body_cascade.detectMultiScale2(
                        gray,
                        scaleFactor=1.5,
                        minNeighbors=3,
                        minSize=(30, 30)
                    )
            except Exception:
                bodies = None
                level_weights = None

        if bodies is None:
            # Final fallback without weights
            bodies = self.body_cascade.detectMultiScale(
                gray,
                scaleFactor=1.5,
                minNeighbors=3,
                minSize=(30, 30)
            )
            level_weights = None

        if len(bodies) == 0:
            return None

        # Find largest detection (usually most confident)
        areas = [w * h for (x, y, w, h) in bodies]
        max_idx = np.argmax(areas)
        x, y, w, h = bodies[max_idx]

        # Compute confidence from available weights; map to [0,1]
        confidence = 0.75
        if level_weights is not None and len(level_weights) > 0:
            lw = np.array(level_weights).reshape(-1)
            raw_w = float(lw[max_idx])
            # Convert OpenCV level weight (often >0) to bounded [0,1]
            confidence = raw_w / (1.0 + abs(raw_w))

        if confidence < self.confidence_threshold:
            return None

        return {
            'bbox': (int(x), int(y), int(w), int(h)),
            'label': 'human',
            'confidence': confidence
        }

    def detect_single_human(self, frame):
        """Detect single human with highest confidence"""
        if self.detection_method == "haarcascade":
            return self.detect_humans_haarcascade(frame)
        else:
            return self.detect_humans_hog(frame)

    def calculate_shift_from_center(self, bbox):
        """Calculate the shift of bounding box center from frame center"""
        x, y, w, h = bbox
        # Calculate center of bounding box
        bbox_center_x = x + w // 2
        bbox_center_y = y + h // 2

        # Calculate shift from frame center
        shift_x = bbox_center_x - self.frame_center_x
        shift_y = bbox_center_y - self.frame_center_y

        return shift_x, shift_y, bbox_center_x, bbox_center_y

    def log_shift_to_file(self, shift_x, shift_y):
        """Log the shift values to text file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"{timestamp} | x={shift_x:+6d} ; y={shift_y:+6d}\n"

        try:
            with open(self.shift_log_file, 'a') as f:
                f.write(log_entry)
        except Exception as e:
            logger.error(f"Error writing to shift log file: {e}")

    def init_tracker(self, frame, bbox):
        """Initialize an OpenCV tracker robustly across versions/builds.

        Tries KCF, then MOSSE, then CSRT, MIL, MedianFlow in both legacy and
        non-legacy namespaces. Also guards against invalid bounding boxes.
        """
        # Validate bbox inside frame bounds and non-trivial size
        h, w = frame.shape[:2]
        x, y, bw, bh = [int(v) for v in bbox]
        # Clip to frame
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = max(1, min(bw, w - x))
        bh = max(1, min(bh, h - y))
        if bw < 5 or bh < 5:
            logger.debug("Skipped tracker init: bbox too small after clipping")
            return False

        def _factory(name):
            # Try legacy first (OpenCV >=4.5 moved trackers here)
            try:
                if hasattr(cv2, 'legacy'):
                    ctor = getattr(cv2.legacy, f'Tracker{name}_create', None)
                    if callable(ctor):
                        return ctor()
            except Exception:
                pass
            # Try top-level (older builds)
            try:
                ctor = getattr(cv2, f'Tracker{name}_create', None)
                if callable(ctor):
                    return ctor()
            except Exception:
                pass
            return None

        # Preference order: KCF (fast/accurate), MOSSE (fast), CSRT (accurate), MIL/MedianFlow (fallbacks)
        for name in ("KCF", "MOSSE", "CSRT", "MIL", "MedianFlow"):
            try:
                tracker = _factory(name)
                if tracker is None:
                    continue
                ok = tracker.init(frame, (x, y, bw, bh))
                # Some OpenCV versions return bool; others raise on failure
                if ok is False:
                    continue
                self.tracker = tracker
                self.tracking_supported = True
                logger.debug(f"Initialized {name} tracker with bbox {(x, y, bw, bh)}")
                return True
            except Exception:
                # Try next tracker type
                continue

        # None of the trackers are available/initialized
        self.tracker = None
        if self.tracking_supported is None:
            self.tracking_supported = False
        if not self._tracker_warned:
            self._tracker_warned = True
            logger.warning(
                "OpenCV tracking APIs not available or failed to initialize. "
                "Continuing in detection-only mode. If desired, install a build "
                "with tracking modules (e.g., opencv-contrib-python)."
            )
        return False

    def update_tracking(self, frame):
        """Update tracker and run detection periodically"""
        current_time = time.time()

        # Try to update existing tracker
        if self.tracker is not None:
            success, bbox = self.tracker.update(frame)
            if success:
                self.tracked_human = {
                    'bbox': tuple(int(v) for v in bbox),
                    'label': 'human',
                    'confidence': 0.9
                }
                # Re-detect every N frames or every 2 seconds to correct drift
                if (self.frame_count % self.detection_interval == 0 or
                    current_time - self.last_detection_time > 2.0):
                    self.run_detection_update(frame)
                return self.tracked_human

        # No tracker or tracking failed - run detection
        if self.frame_count % max(1, self.detection_interval // 2) == 0:
            return self.run_detection_update(frame)

        return None

    def run_detection_update(self, frame):
        """Run detection on downscaled frame"""
        self.last_detection_time = time.time()

        # Downscale for detection
        small_frame = cv2.resize(frame, None, fx=self.process_scale, fy=self.process_scale)

        # Detect human
        detected_human = self.detect_single_human(small_frame)

        if detected_human:
            # Scale bbox back to original size
            dx, dy, dw, dh = detected_human['bbox']
            scale_inv = 1.0 / self.process_scale
            # Compute scaled bbox
            x = int(dx * scale_inv)
            y = int(dy * scale_inv)
            w = int(dw * scale_inv)
            h = int(dh * scale_inv)

            # Clip bbox to frame bounds to prevent tracker init errors
            fh, fw = frame.shape[:2]
            x = max(0, min(x, fw - 1))
            y = max(0, min(y, fh - 1))
            w = max(1, min(w, fw - x))
            h = max(1, min(h, fh - y))
            bbox = (x, y, w, h)

            # Reinitialize tracker with new detection if supported
            tracker_ok = self.tracking_supported is not False and self.init_tracker(frame, bbox)

            self.tracked_human = {
                'bbox': bbox,
                'label': 'human',
                'confidence': detected_human['confidence']
            }
            if tracker_ok:
                logger.debug(f"Human detected and tracker initialized: conf={detected_human['confidence']:.2f}")
            else:
                logger.debug(f"Human detected (detection-only mode): conf={detected_human['confidence']:.2f}")
            return self.tracked_human

        return None

    def draw_tracking(self, frame):
        """Draw minimal tracking visualization for performance"""
        # Draw small frame center crosshair
        cv2.line(frame, (self.frame_center_x - 5, self.frame_center_y),
                 (self.frame_center_x + 5, self.frame_center_y), (255, 0, 0), 1)
        cv2.line(frame, (self.frame_center_x, self.frame_center_y - 5),
                 (self.frame_center_x, self.frame_center_y + 5), (255, 0, 0), 1)

        if self.tracked_human:
            x, y, w, h = self.tracked_human['bbox']
            confidence = self.tracked_human['confidence']

            # Calculate shift
            shift_x, shift_y, bbox_center_x, bbox_center_y = self.calculate_shift_from_center((x, y, w, h))

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw center point
            cv2.circle(frame, (bbox_center_x, bbox_center_y), 3, (0, 0, 255), -1)

            # Draw line from frame center to human center
            cv2.line(frame, (self.frame_center_x, self.frame_center_y),
                     (bbox_center_x, bbox_center_y), (255, 255, 0), 1)

            # Display shift information
            shift_text = f"x={shift_x:+d} y={shift_y:+d}"
            cv2.putText(frame, shift_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Log shift to file
            self.log_shift_to_file(shift_x, shift_y)

            # Update EV3 motors
            if self.ev3_controller and self.ev3_controller.connected:
                self.ev3_controller.update_motors(shift_x, shift_y)
        else:
            # No human detected - stop motors
            if self.ev3_controller and self.ev3_controller.connected:
                self.ev3_controller.stop_motors()

        return frame

    def start_recording(self):
        """Start recording with Raspberry Pi compatible codecs"""
        if not self.is_recording and self.cap is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Try multiple codec options for Raspberry Pi compatibility
            codec_options = [
                ('MJPG', '.avi', 'MJPEG'),  # Most compatible
                ('mp4v', '.mp4', 'MP4V'),   # MP4 fallback
                ('XVID', '.avi', 'XVID'),   # AVI fallback
                ('H264', '.mp4', 'H264'),   # Try H264 as last resort
            ]

            for fourcc_str, extension, name in codec_options:
                try:
                    self.output_filename = os.path.join(self.output_dir, f"recording_{timestamp}{extension}")
                    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                    self.video_writer = cv2.VideoWriter(self.output_filename, fourcc, fps, (width, height))

                    if self.video_writer.isOpened():
                        self.is_recording = True
                        logger.info(f"Started recording with {name} codec: {self.output_filename}")
                        logger.info(f"Recording at {width}x{height} @ {fps} FPS")
                        return
                    else:
                        logger.warning(f"Failed to initialize {name} codec, trying next...")
                        self.video_writer = None
                except Exception as e:
                    logger.warning(f"Error with {name} codec: {e}")
                    self.video_writer = None
                    continue

            logger.error("Failed to initialize video writer with any codec")
            logger.info("Recording disabled - tracking will continue without recording")

    def stop_recording(self):
        if self.is_recording and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False
            logger.info(f"Stopped recording: {self.output_filename}")

    def process_frame(self, frame):
        """Process frame with optimized tracking"""
        self.frame_count += 1

        # Update tracking
        tracked_human = self.update_tracking(frame)

        # Draw visualization
        annotated_frame = self.draw_tracking(frame)

        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(annotated_frame)

        return annotated_frame, tracked_human

    def run(self, display_video=True, auto_record=False):
        if not self.connect_to_stream():
            logger.error("Failed to connect to stream")
            return

        self.is_running = True
        if auto_record:
            self.start_recording()

        logger.info("Starting optimized human tracking...")
        logger.info("Tracking SINGLE human with maximum confidence")
        logger.info("Press 'q' to quit, 'r' to toggle recording, 's' to take screenshot, 'd' to force detection, 'e' to toggle EV3")

        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0.0

        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame from stream")
                    time.sleep(0.1)
                    continue

                annotated_frame, tracked_human = self.process_frame(frame)

                # Calculate and display FPS
                fps_frame_count += 1
                if fps_frame_count >= 30:
                    fps_end_time = time.time()
                    current_fps = fps_frame_count / (fps_end_time - fps_start_time)
                    logger.info(f"FPS: {current_fps:.2f}")
                    fps_start_time = fps_end_time
                    fps_frame_count = 0

                if display_video:
                    # Show FPS on frame
                    cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # Show EV3 status
                    if self.ev3_controller and self.ev3_controller.connected:
                        cv2.putText(annotated_frame, "EV3: ON",
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    elif self.ev3_controller:
                        cv2.putText(annotated_frame, "EV3: OFF",
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    cv2.imshow('MJPEG Human Tracker', annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        if self.is_recording:
                            self.stop_recording()
                        else:
                            self.start_recording()
                    elif key == ord('s'):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        screenshot_path = os.path.join(self.output_dir, f"screenshot_{timestamp}.jpg")
                        cv2.imwrite(screenshot_path, annotated_frame)
                        logger.info(f"Screenshot saved: {screenshot_path}")
                    elif key == ord('d'):
                        # Force detection
                        logger.info("Forcing detection...")
                        self.tracker = None
                        self.tracked_human = None
                    elif key == ord('e'):
                        # Toggle EV3 connection
                        if self.ev3_controller:
                            if self.ev3_controller.connected:
                                self.ev3_controller.disconnect()
                                logger.info("EV3 disabled")
                            else:
                                self.ev3_controller.connect()
                                logger.info("EV3 enabled")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        logger.info("Cleaning up...")
        self.is_running = False

        # Stop EV3 motors
        if self.ev3_controller:
            self.ev3_controller.disconnect()

        if self.is_recording:
            self.stop_recording()
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info(f"Shift log saved to: {self.shift_log_file}")
        logger.info("Cleanup completed")

def main():
    parser = argparse.ArgumentParser(description='Optimized MJPEG Human Tracker for Raspberry Pi 5 with EV3 Control')
    parser.add_argument('--url', default='http://192.168.100.1:8000/stream.mjpg',
                        help='MJPEG stream URL')
    parser.add_argument('--output-dir', default='recordings',
                        help='Output directory for recordings')
    parser.add_argument('--detection-method', choices=['hog', 'haarcascade'],
                        default='haarcascade',
                        help='Human detection method (haarcascade is faster on RPi)')
    parser.add_argument('--confidence-threshold', type=float, default=0.75,
                        help='Confidence threshold for human detection')
    parser.add_argument('--detection-interval', type=int, default=15,
                        help='Run detection every N frames (higher = faster but less responsive)')
    parser.add_argument('--process-scale', type=float, default=0.3,
                        help='Scale factor for detection processing (lower = faster)')
    parser.add_argument('--no-display', action='store_true',
                        help='Run without displaying video (saves CPU)')
    parser.add_argument('--no-auto-record', action='store_true',
                        help='Do not start recording automatically')

    # EV3 arguments
    parser.add_argument('--ev3-deadzone-x', type=int, default=100,
                        help='Horizontal deadzone in pixels (no motor movement within this range)')
    parser.add_argument('--ev3-deadzone-y', type=int, default=100,
                        help='Vertical deadzone in pixels (no motor movement within this range)')
    parser.add_argument('--ev3-speed-factor', type=float, default=1.0,
                        help='Speed multiplier for motor control (0.1-2.0)')
    parser.add_argument('--ev3-max-speed', type=int, default=5,
                        help='Maximum motor speed (1-100)')
    parser.add_argument('--ev3-invert-x', action='store_true', default=False,
                        help='Invert horizontal motor direction')
    parser.add_argument('--ev3-invert-y', action='store_true', default=False,
                        help='Invert vertical motor direction')

    args = parser.parse_args()

    tracker = MJPEGTracker(
        stream_url=args.url,
        output_dir=args.output_dir,
        detection_method=args.detection_method,
        confidence_threshold=args.confidence_threshold,
        detection_interval=args.detection_interval,
        process_scale=args.process_scale,
        ev3_deadzone_x=args.ev3_deadzone_x,
        ev3_deadzone_y=args.ev3_deadzone_y,
        ev3_speed_factor=args.ev3_speed_factor,
        ev3_max_speed=args.ev3_max_speed,
        ev3_invert_x=args.ev3_invert_x,
        ev3_invert_y=args.ev3_invert_y
    )

    tracker.run(
        display_video=not args.no_display,
        auto_record=not args.no_auto_record
    )

if __name__ == "__main__":
    main()
