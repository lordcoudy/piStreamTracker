"""Object tracker, HumanTracker, shift log, CLI."""

import argparse
import logging
import time
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

import cv2
import numpy as np

from pistream.capture import VideoCapture
from pistream.config import apply_cli_overrides, configure_logging, load_config
from pistream.detect import AsyncDetector, PoseDetector
from pistream.horizon import tilt_degrees
from pistream.motors import MotorController
from pistream.camera_auth import auth_headers
from pistream.preview import accept_new_frame, capped_recording_fps
from pistream.record import _RecordingThread

logger = logging.getLogger(__name__)

REINIT_IOU = 0.3


def bbox_iou(a: tuple, b: tuple) -> float:
    """Intersection-over-union for (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def should_reinit_tracker(active: bool, iou: Optional[float], min_iou: float = REINIT_IOU) -> bool:
    if not active:
        return True
    if iou is None:
        return True
    return iou < min_iou


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
        self._last_keypoints = None
        self._last_confidence = 0.5
        self.frame_count = 0
        self._rec_thread: Optional[_RecordingThread] = None
        self._fps_count = 0
        self._fps_time = time.monotonic()
        self._fps = 0.0

        # Digital zoom (1.0 = no zoom, up to 4.0)
        self.zoom_level: float = 1.0

        # Horizon stabilization
        self.horizon_correction: bool = False
        self._horizon_angle_ema: float = 0.0  # exponential moving average of tilt

        # Recording mode: 'local' (Pi 5) or 'camera' (Pi 3B+)
        self.recording_mode = tracker_cfg.get('recording_mode', 'local')
        self._camera_base_url = f"http://{net['camera_ip']}:{cam['port']}"
        self._camera_token = (cam or {}).get('token') or ''

        # Async detector (created in connect() once we know frame size)
        self._async_detector: Optional[AsyncDetector] = None
        self._cleaned = False

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
            iou = None
            if self.tracker.active and self.current_detection:
                iou = bbox_iou(self.current_detection['bbox'], det_result['bbox'])
            if should_reinit_tracker(self.tracker.active, iou):
                self.tracker.init(frame, det_result['bbox'])
            self._last_keypoints = det_result.get('keypoints')
            self._last_confidence = det_result.get('confidence', 0.5)
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
                'confidence': self._last_confidence,
                'keypoints': self._last_keypoints,
            }
            self.current_detection = detection
        elif det_result:
            # Tracker init + update failed, but we have a fresh detection
            detection = det_result

        if detection:
            self._update_aim(detection)
        else:
            self.motors.stop()

        annotated = self._draw(frame.copy(), detection)
        annotated = self._apply_horizon_preview(annotated, detection)
        annotated = self._apply_zoom(annotated)
        return annotated, detection

    def _update_aim(self, detection: dict) -> None:
        """Drive motors from the raw (unrotated) bounding box."""
        x, y, w, h = detection['bbox']
        cx = self.capture.width // 2
        cy = self.capture.height // 2
        shift_x = x + w // 2 - cx
        shift_y = y + h // 4 - cy
        if self._shift_logger:
            self._shift_logger.log(shift_x, shift_y)
        self.motors.update(shift_x, shift_y)

    def _compute_horizon_correction(self, detection: Optional[dict]) -> float:
        """EMA-smoothed tilt in degrees from shoulder/hip keypoints."""
        if not self.horizon_correction or detection is None:
            return 0.0

        raw = tilt_degrees(detection.get('keypoints'), self.detector.keypoint_threshold)
        if raw is None:
            return self._horizon_angle_ema

        alpha = 0.15
        self._horizon_angle_ema = alpha * raw + (1 - alpha) * self._horizon_angle_ema
        return self._horizon_angle_ema

    def _apply_horizon_preview(self, frame: np.ndarray, detection: Optional[dict]) -> np.ndarray:
        """Rotate the already-annotated preview so overlay stays on the person."""
        angle = self._compute_horizon_correction(detection)
        if abs(angle) <= 0.5:
            return frame
        h_fr, w_fr = frame.shape[:2]
        matrix = cv2.getRotationMatrix2D((w_fr / 2, h_fr / 2), -angle, 1.0)
        frame = cv2.warpAffine(frame, matrix, (w_fr, h_fr),
                               borderMode=cv2.BORDER_REPLICATE)
        cv2.putText(frame, f"H:{angle:+.1f}", (w_fr // 2 - 30, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)
        return frame

    def _apply_zoom(self, frame: np.ndarray) -> np.ndarray:
        """Apply digital zoom by center-cropping and resizing."""
        if self.zoom_level <= 1.0:
            return frame

        h, w = frame.shape[:2]
        crop_w = int(w / self.zoom_level)
        crop_h = int(h / self.zoom_level)
        x1 = (w - crop_w) // 2
        y1 = (h - crop_h) // 2
        cropped = frame[y1:y1 + crop_h, x1:x1 + crop_w]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    def _draw(self, frame: np.ndarray, detection: Optional[dict]) -> np.ndarray:
        """Draw tracking overlay on an unrotated frame."""
        cx, cy = self.capture.width // 2, self.capture.height // 2

        cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (255, 0, 0), 1)
        cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (255, 0, 0), 1)

        if detection:
            x, y, w, h = detection['bbox']
            tx, ty = x + w // 2, y + h // 4
            shift_x, shift_y = tx - cx, ty - cy

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.line(frame, (x, y + h // 2), (x + w, y + h // 2), (0, 180, 0), 1)
            cv2.circle(frame, (tx, ty), 4, (0, 0, 255), -1)
            cv2.line(frame, (cx, cy), (tx, ty), (255, 255, 0), 1)

            kp = detection.get('keypoints')
            if kp is not None:
                for px, py, conf in kp:
                    if conf >= self.detector.keypoint_threshold:
                        cv2.circle(frame, (int(px), int(py)), 3, (255, 0, 255), -1)

            cv2.putText(frame, f"x={shift_x:+d} y={shift_y:+d}", (x, y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

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
                    headers=auth_headers(self._camera_token),
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

        source_fps = float(self.capture.fps or 0) if self.capture else 0.0
        if source_fps <= 1.0:
            source_fps = float((self.config.get('camera') or {}).get('framerate') or 30)
        fps = capped_recording_fps(self.recording_fps, source_fps)
        self._rec_thread = _RecordingThread(
            str(path), self.capture.width, self.capture.height,
            fps, encoder=encoder,
        )
        self._rec_thread.start()
        self.recording = True
        logger.info(f"Recording: {path}  ({fps:.0f} fps, source {source_fps:.0f})")

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
                    headers=auth_headers(self._camera_token),
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
            prev_frame = None
            while self.running:
                ret, frame = self.capture.read()
                prev_frame, is_new = accept_new_frame(prev_frame, ret, frame)
                if not is_new:
                    time.sleep(0.002)
                    continue

                rec_frame = (
                    frame.copy()
                    if self.recording and self.recording_mode != 'camera'
                    else None
                )
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
                        self._last_keypoints = None
                        self.motors.move_to_home()
                        logger.info("Detection reset + camera homed")
                    elif key == ord('e'):
                        self.motors.disconnect() if self.motors.connected else self.motors.connect()

        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            self.cleanup()

    def cleanup(self):
        """Release all resources. Safe to call more than once."""
        if self._cleaned:
            return
        self._cleaned = True
        logger.info("Cleaning up...")
        self.running = False
        if self._async_detector:
            self._async_detector.stop()
            self._async_detector = None
        self.motors.disconnect()
        self.stop_recording()
        if self.capture:
            self.capture.stop()
            self.capture = None
        if self._shift_logger:
            self._shift_logger.flush()
        cv2.destroyAllWindows()


# =============================================================================
# Utilities


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
    parser.add_argument('--preset', help='Performance preset from config.yaml (pi3, pi5, fast, quality)')

    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    apply_cli_overrides(config, args)
    configure_logging(config)

    tracker = HumanTracker(config)
    tracker.run(display=not args.no_display, auto_record=args.auto_record)


if __name__ == "__main__":
    main()
