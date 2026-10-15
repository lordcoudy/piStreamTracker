"""Recording ownership when CLI tracking or an HTTP request is interrupted."""

import unittest
from threading import RLock
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2

from pistream.track import HumanTracker


def camera_tracker():
    tracker = HumanTracker.__new__(HumanTracker)
    tracker._record_lock = RLock()
    tracker._recording = False
    tracker._recording_backend = None
    tracker._rec_thread = None
    tracker._recording_error = None
    tracker.remote_stop_pending = False
    tracker._cleaned = False
    tracker._async_detector = None
    tracker._shift_logger = None
    tracker._camera_base_url = 'http://camera:8000'
    tracker._camera_token = 'secret'
    tracker.recording_mode = 'camera'
    tracker.running = False
    tracker.motors = Mock()
    tracker.capture = SimpleNamespace(read=Mock(), stop=Mock())
    tracker.connect = Mock(return_value=True)
    return tracker


class CameraResponse:
    def read(self):
        return b'{"recording": false}'

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class RecordingOwnershipIntegrationTests(unittest.TestCase):
    def test_headless_window_cleanup_failure_preserves_exit_status(self):
        tracker = camera_tracker()
        tracker.capture.read.side_effect = KeyboardInterrupt()
        with patch('pistream.track.cv2.destroyAllWindows', side_effect=cv2.error('No GUI')):
            self.assertEqual(tracker.run(display=False), 0)

    def test_shift_log_flush_failure_does_not_skip_camera_recording_stop(self):
        tracker = camera_tracker()
        tracker._recording = True
        tracker._recording_backend = 'camera'
        tracker._shift_logger = SimpleNamespace(flush=Mock(side_effect=OSError('disk full')))
        with patch('pistream.track.urllib.request.urlopen', return_value=CameraResponse()) as request:
            tracker.cleanup()
        request.assert_called_once()
        self.assertFalse(tracker.recording)
        self.assertFalse(tracker.remote_stop_pending)
        self.assertIsNone(tracker.capture)

    def test_tracking_exception_enters_pending_camera_recovery_before_exiting(self):
        tracker = camera_tracker()
        tracker._recording = True
        tracker._recording_backend = 'camera'
        tracker.capture.read.side_effect = RuntimeError('capture worker failed')

        def recover(**_kwargs):
            tracker.remote_stop_pending = False
            tracker.recording = False
            return 0

        tracker._recover_camera_stop = Mock(side_effect=recover)
        with (
            patch('pistream.track.urllib.request.urlopen', side_effect=OSError('camera offline')),
            patch('pistream.track.cv2.destroyAllWindows'),
        ):
            self.assertEqual(tracker.run(display=False), 1)
        tracker._recover_camera_stop.assert_called_once_with(display=False, frame=None)
        tracker.motors.disconnect.assert_called_once()
        self.assertIsNone(tracker.capture)

    def test_interrupted_camera_start_keeps_ownership_until_cleanup_confirms_stop(self):
        tracker = camera_tracker()
        camera_recording = []

        def request(req, **_kwargs):
            if req.full_url.endswith('/record/start'):
                camera_recording.append(True)
                raise KeyboardInterrupt()
            if req.full_url.endswith('/record/stop'):
                camera_recording.clear()
                return CameraResponse()
            self.fail('Unexpected camera request')

        with (
            patch('pistream.track.urllib.request.urlopen', side_effect=request),
            patch('pistream.track.cv2.destroyAllWindows'),
        ):
            self.assertEqual(tracker.run(display=False, auto_record=True), 0)
        self.assertFalse(camera_recording, 'Interrupted start left the camera recording without an owner')
        self.assertFalse(tracker.remote_stop_pending)
        self.assertFalse(tracker.recording)


if __name__ == '__main__':
    unittest.main()
