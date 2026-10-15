"""Recording backend selection and camera-to-local fallback."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, RLock
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from pistream.track import HumanTracker


class FakeRecordingThread:
    def __init__(self, path):
        self.output_path = path
        self.started = False
        self.stopped = False
        self.frame = None
        self.error = None
        self.finished = Event()
        self.closed = Event()
        self.ready = True
        self.can_close = True

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True
        if not self.can_close:
            return False
        self.finished.set()
        self.closed.set()
        return True

    def update_frame(self, frame):
        self.frame = frame


def tracker_shell(output_dir: str) -> HumanTracker:
    tracker = HumanTracker.__new__(HumanTracker)
    tracker.recording = False
    tracker.recording_mode = 'camera'
    tracker._recording_backend = None
    tracker._record_lock = RLock()
    tracker._rec_thread = None
    tracker._recording_error = None
    tracker.remote_stop_pending = False
    tracker._cleaned = False
    tracker._async_detector = None
    tracker._shift_logger = None
    tracker.motors = Mock()
    tracker.running = False
    tracker._camera_base_url = 'http://camera:8000'
    tracker._camera_token = 'secret'
    tracker.capture = SimpleNamespace(fps=30.0, width=64, height=48, stop=Mock())
    tracker.recording_fps = 30.0
    tracker.config = {
        'tracker': {'recording_encoder': 'mjpg'},
        'camera': {'framerate': 30},
    }
    tracker.output_dir = Path(output_dir)
    return tracker


class RecordingBackendTests(unittest.TestCase):
    def test_failed_writer_is_not_active_but_keeps_output_protected_until_closed(self):
        with TemporaryDirectory() as tmp:
            tracker = tracker_shell(tmp)
            recorder = FakeRecordingThread(str(Path(tmp) / 'lecture.avi'))
            tracker._rec_thread = recorder
            tracker._recording_backend = 'local'
            tracker.recording = True
            recorder.error = 'Encoder exited'
            recorder.finished.set()
            self.assertFalse(tracker.recording)
            self.assertEqual(tracker.recording_error, 'Encoder exited')
            self.assertEqual(tracker.local_recording_path, Path(recorder.output_path))
            self.assertTrue(tracker.stop_recording())
            self.assertIsNone(tracker.local_recording_path)
            self.assertEqual(tracker.recording_error, 'Encoder exited')

    def test_new_recording_closes_failed_writer_and_clears_error_only_on_success(self):
        with TemporaryDirectory() as tmp:
            tracker = tracker_shell(tmp)
            tracker.recording_mode = 'local'
            old = FakeRecordingThread(str(Path(tmp) / 'old.avi'))
            old.error = 'Broken pipe'
            old.finished.set()
            tracker._rec_thread = old
            tracker._recording_backend = 'local'
            tracker.recording = True
            new = FakeRecordingThread(str(Path(tmp) / 'new.avi'))
            with patch('pistream.track._RecordingThread', return_value=new):
                self.assertTrue(tracker.start_recording())
            self.assertTrue(old.closed.is_set())
            self.assertTrue(new.started)
            self.assertTrue(tracker.recording)
            self.assertIsNone(tracker.recording_error)

    def test_failed_close_retains_owner_and_prevents_replacement(self):
        with TemporaryDirectory() as tmp:
            tracker = tracker_shell(tmp)
            old = FakeRecordingThread(str(Path(tmp) / 'old.avi'))
            old.error = 'Blocked writer'
            old.finished.set()
            old.can_close = False
            tracker._rec_thread = old
            tracker._recording_backend = 'local'
            tracker.recording = True
            with patch('pistream.track._RecordingThread') as factory:
                self.assertFalse(tracker.start_recording())
            factory.assert_not_called()
            self.assertIs(tracker._rec_thread, old)
            self.assertTrue(tracker.cleanup_pending)

    def test_camera_stop_requires_explicit_false_and_remains_retryable(self):
        class Response:
            def __init__(self, data):
                self.data = data

            def read(self):
                return self.data

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        with TemporaryDirectory() as tmp:
            tracker = tracker_shell(tmp)
            tracker.recording = True
            tracker._recording_backend = 'camera'
            for body in (b'{}', b'<html>', b'{"recording": "false"}', b'{"recording": true}'):
                with self.subTest(body=body), patch(
                    'pistream.track.urllib.request.urlopen', return_value=Response(body)
                ):
                    self.assertFalse(tracker.stop_recording())
                    self.assertTrue(tracker.remote_stop_pending)
                    self.assertTrue(tracker.recording)
            with patch('pistream.track.urllib.request.urlopen', return_value=Response(b'{"recording": false}')):
                self.assertTrue(tracker.stop_recording())
            self.assertFalse(tracker.remote_stop_pending)
            self.assertFalse(tracker.recording)

    def test_cleanup_releases_local_resources_without_repeating_failed_camera_stop(self):
        with TemporaryDirectory() as tmp:
            tracker = tracker_shell(tmp)
            capture = tracker.capture
            tracker.recording = True
            tracker._recording_backend = 'camera'
            with patch('pistream.track.urllib.request.urlopen', side_effect=OSError('offline')) as request:
                tracker.cleanup()
                tracker.cleanup()
            self.assertEqual(request.call_count, 1)
            capture.stop.assert_called_once()
            tracker.motors.disconnect.assert_called_once()
            self.assertIsNone(tracker.capture)
            self.assertTrue(tracker.cleanup_pending)
            self.assertTrue(tracker.remote_stop_pending)

    def test_camera_failure_falls_back_to_live_local_backend(self):
        with TemporaryDirectory() as tmp:
            tracker = tracker_shell(tmp)
            recorder = FakeRecordingThread(str(Path(tmp) / 'fallback.avi'))
            with (
                patch('pistream.track.urllib.request.urlopen', side_effect=OSError('offline')) as open_url,
                patch('pistream.track._RecordingThread', return_value=recorder),
            ):
                self.assertTrue(tracker.start_recording())
                self.assertTrue(tracker.records_locally)
                self.assertEqual(tracker._recording_backend, 'local')
                frame = np.zeros((48, 64, 3), dtype=np.uint8)
                tracker.write_frame(frame)
                self.assertIs(recorder.frame, frame)
                tracker.stop_recording()

            self.assertTrue(recorder.started)
            self.assertTrue(recorder.stopped)
            self.assertEqual(open_url.call_count, 2)
            self.assertFalse(tracker.recording)

    def test_failed_camera_stop_keeps_truthful_retryable_state(self):
        with TemporaryDirectory() as tmp:
            tracker = tracker_shell(tmp)
            tracker.recording = True
            tracker._recording_backend = 'camera'
            with patch(
                'pistream.track.urllib.request.urlopen', side_effect=OSError('offline')
            ):
                self.assertFalse(tracker.stop_recording())

            self.assertTrue(tracker.recording)
            self.assertEqual(tracker._recording_backend, 'camera')

    def test_html_camera_start_body_falls_back_to_local(self):
        class HtmlResp:
            def read(self):
                return b'<html>nope</html>'

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        with TemporaryDirectory() as tmp:
            tracker = tracker_shell(tmp)
            recorder = FakeRecordingThread(str(Path(tmp) / 'fallback.avi'))
            with (
                patch('pistream.track.urllib.request.urlopen', return_value=HtmlResp()),
                patch('pistream.track._RecordingThread', return_value=recorder),
            ):
                self.assertTrue(tracker.start_recording())
                self.assertEqual(tracker._recording_backend, 'local')
