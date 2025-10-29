import logging
import os
import time
from datetime import datetime

import cv2
import numpy as np
import yaml
from EV3Controller import EV3Controller

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
tracker_config = config['tracker']

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(tracker_config['log_file']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Tracker:
    def __init__(self):
        self.stream_url = tracker_config['stream_url']
        self.output_dir = tracker_config['output_dir']
        self.detection_type = tracker_config['detection']['type']
        self.confidence_threshold = tracker_config['detection']['confidence_threshold']
        self.detection_interval = tracker_config['detection']['interval']
        self.process_scale = tracker_config['detection']['process_scale']

        os.makedirs(self.output_dir, exist_ok=True)

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
        self.shift_log_file = os.path.join(self.output_dir, f"{tracker_config['shift_log_prefix']}_{timestamp}.txt")

        # Initialize shift log file
        with open(self.shift_log_file, 'w') as f:
            f.write("Human Position Shifts from Frame Center\n")
            f.write("=" * 50 + "\n")
            f.write("Format: timestamp | x=<shift> ; y=<shift>\n")
            f.write("=" * 50 + "\n\n")

        # Initialize EV3 controller
        self.ev3_controller = EV3Controller()

        self.initialize_detection()
        try:
            logger.debug(f"OpenCV version: {cv2.__version__}")
        except Exception:
            pass
        logger.info(f"Tracker initialized with URL: {self.stream_url}")
        logger.debug(f"Detection interval: every {self.detection_interval} frames")
        logger.debug(f"Process scale: {self.process_scale}")
        logger.debug(f"Shift log file: {self.shift_log_file}")
        if self.ev3_controller and self.ev3_controller.connected:
            logger.info(f"EV3 control enabled")

    def initialize_detection(self):
        try:
            # Use frontal face cascade for better performance
            if self.detection_type == "face":
                self.haarcascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            else:
                self.haarcascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
            logger.debug("Haar Cascade initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing detection: {e}")

    def connect_to_stream(self):
        try:
            self.cap = cv2.VideoCapture(self.stream_url)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                fps = self.cap.get(cv2.CAP_PROP_FPS) or 60
                self.ev3_controller.cam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.ev3_controller.cam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Calculate frame center
                self.frame_center_x = self.ev3_controller.cam_width // 2
                self.frame_center_y = self.ev3_controller.cam_height // 2

                logger.info(f"Connected to stream: {self.ev3_controller.cam_width}x{self.ev3_controller.cam_height} @ {fps} FPS")
                logger.debug(f"Frame center: ({self.frame_center_x}, {self.frame_center_y})")
                return True
            else:
                logger.error("Failed to connect to stream")
                return False
        except Exception as e:
            logger.error(f"Error connecting to stream: {e}")
            return False

    def detect_humans_haarcascade(self, frame):
        """Detect humans using Haar Cascade - optimized for single detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optimize parameters for speed
        bodies = None
        level_weights = None

        # Prefer APIs that return weights to estimate confidence
        try:
            if hasattr(self.haarcascade, 'detectMultiScale3'):
                bodies, rejectLevels, level_weights = self.haarcascade.detectMultiScale3(
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
                if hasattr(self.haarcascade, 'detectMultiScale2'):
                    bodies, level_weights = self.haarcascade.detectMultiScale2(
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
            bodies = self.haarcascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
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
            try:
                if hasattr(cv2, 'legacy'):
                    ctor = getattr(cv2.legacy, f'Tracker{name}_create', None)
                    if callable(ctor):
                        return ctor()
            except Exception:
                pass
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
                    'confidence': self.tracked_human['confidence'] 
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
        detected_human = self.detect_humans_haarcascade(small_frame)

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
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 60
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
                        logger.debug(f"Recording at {width}x{height} @ {fps} FPS")
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

    def process_key(self, key, frame):
        if key == ord('q'):
            return -1
        elif key == ord('r'):
            if self.is_recording:
                self.stop_recording()
            else:
                self.start_recording()
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(self.output_dir, f"screenshot_{timestamp}.jpg")
            cv2.imwrite(screenshot_path, frame)
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
        return 0


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

        logger.info("Starting human tracking...")
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

                    cv2.imshow('Human Tracker', annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if self.process_key(key, annotated_frame) == -1:
                        break
                else:
                    key = int(input()) & 0xFF
                    if self.process_key(key, annotated_frame) == -1:
                        break
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
        logger.debug(f"Shift log saved to: {self.shift_log_file}")
        logger.info("Cleanup completed")
