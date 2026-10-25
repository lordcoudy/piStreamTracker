"""Adversarial motion boundaries: Stop, capture races and worker ownership."""

import time
from threading import Event, RLock, Thread
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from pistream.detect import AsyncDetector
from pistream.config import default_config
from pistream.track import HumanTracker


def test_stop_disables_motion_even_while_frame_processing_holds_lock():
    human = HumanTracker.__new__(HumanTracker)
    human._tracking_lock = RLock()
    human._async_detector = None
    human.running = True
    disconnected = Event()
    human.motors = Mock()
    human.motors.disconnect.side_effect = disconnected.set
    stopper = Thread(target=human.request_stop)
    try:
        with human._tracking_lock:
            stopper.start()
            disabled_without_waiting_for_frame = disconnected.wait(.25)
    finally:
        stopper.join(1)
    assert disabled_without_waiting_for_frame
    assert human._motion_stopped
    assert not human.running


def test_aim_uses_consumed_frame_dimensions_when_capture_already_advanced():
    human = HumanTracker.__new__(HumanTracker)
    human._tracking_lock = RLock()
    human.frame_count = 0
    human.detection_interval = 10
    human._last_keypoints = None
    human._last_confidence = .9
    human._shift_logger = None
    human._async_detector = None
    human.current_detection = {'bbox': (300, 220, 40, 80)}
    # Capture advanced to a new resolution after the 640x480 frame was consumed.
    human.capture = SimpleNamespace(width=1280, height=960)
    human.motors = Mock()
    human.tracker = SimpleNamespace(active=True, update=lambda _: (300, 220, 40, 80))
    human._update_horizon = lambda _: 0
    human._level_frame = lambda frame, _: frame
    human._draw = lambda frame, *args: frame
    human._apply_zoom = lambda frame: frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    human._capture_sample = (frame, 1, time.monotonic())
    human.process_frame(frame)
    human.motors.update.assert_called_once_with(0, 0)


def test_cleanup_retains_blocked_inference_until_worker_has_exited():
    entered, release = Event(), Event()

    class Detector:
        def detect(self, frame):
            entered.set()
            release.wait(5)
            return {'bbox': (1, 2, 10, 20), 'keypoints': None}

    worker = AsyncDetector(Detector(), 1, 64, 48)
    human = HumanTracker.__new__(HumanTracker)
    human._cleaned = False
    human._cleanup_ok = True
    human._tracking_lock = RLock()
    human._record_lock = RLock()
    human.remote_stop_pending = False
    human._rec_thread = None
    human.running = True
    human.motors = Mock()
    human._async_detector = worker
    human.capture = None
    human._shift_logger = None
    human.stop_recording = Mock(return_value=True)
    worker.start()
    try:
        worker.submit(np.zeros((48, 64, 3), dtype=np.uint8))
        assert entered.wait(1)
        human.cleanup()
        assert not worker.closed
        assert human.cleanup_pending
        assert human._async_detector is worker
    finally:
        release.set()
        worker.stop()
    assert worker.closed
    assert not human.cleanup_pending


def _human_with_result(tmp_path, now):
    config = default_config()
    config['motors']['backend'] = 'none'
    config['tracker']['output_dir'] = str(tmp_path)
    with patch('pistream.track.PoseDetector', return_value=SimpleNamespace(keypoint_threshold=.3)):
        human = HumanTracker(config)
    human.capture = SimpleNamespace(width=64, height=48, frame_metadata=(1, now), stop=lambda: None)
    human.motors = Mock()
    human._async_detector = AsyncDetector(None, 1, 64, 48)
    human._async_detector._new_result = True
    human._async_detector._result_at = now
    human._async_detector._result = {'bbox': (10, 10, 20, 30), 'keypoints': None}
    return human


def test_reset_and_reconnect_drop_old_detection_before_it_can_aim(tmp_path):
    for boundary in ('reset', 'reconnect'):
        now = time.monotonic()
        human = _human_with_result(tmp_path, now)
        try:
            if boundary == 'reset':
                human.reset_tracking(home=False)
            else:
                human._capture_generation = 0
                human._last_capture_at = now - .01
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            _, detection = human.process_frame(frame)
            assert detection is None
            human.motors.update.assert_not_called()
            # Fresh work in the new generation is still allowed to drive.
            worker = human._async_detector
            worker._new_result = True
            worker._result_at = now
            worker._result = {'bbox': (10, 10, 20, 30), 'keypoints': None}
            _, detection = human.process_frame(frame.copy())
            assert detection is not None
            human.motors.update.assert_called_once()
        finally:
            human.cleanup()


def test_old_buffered_frame_cannot_refresh_watchdog_or_issue_movement(tmp_path):
    now = time.monotonic()
    human = _human_with_result(tmp_path, now)
    human.capture.frame_metadata = (1, now - 10)
    try:
        _, detection = human.process_frame(np.zeros((48, 64, 3), dtype=np.uint8))
        assert detection is None
        human.motors.frame_seen.assert_not_called()
        human.motors.update.assert_not_called()
        human.motors.target_lost.assert_called()
    finally:
        human.cleanup()
