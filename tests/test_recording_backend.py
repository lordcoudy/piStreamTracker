"""Recording backend selection and camera-to-local fallback."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pistream.track import HumanTracker


class FakeRecordingThread:
    def __init__(self, path):
        self.output_path = path
        self.started = False
        self.stopped = False
        self.frame = None

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def update_frame(self, frame):
        self.frame = frame


def tracker_shell(output_dir: str) -> HumanTracker:
    tracker = HumanTracker.__new__(HumanTracker)
    tracker.recording = False
    tracker.recording_mode = 'camera'
    tracker._recording_backend = None
    tracker._record_lock = RLock()
    tracker._rec_thread = None
    tracker._camera_base_url = 'http://camera:8000'
    tracker._camera_token = 'secret'
    tracker.capture = SimpleNamespace(fps=30.0, width=64, height=48)
    tracker.recording_fps = 30.0
    tracker.config = {
        'tracker': {'recording_encoder': 'mjpg'},
        'camera': {'framerate': 30},
    }
    tracker.output_dir = Path(output_dir)
    return tracker


class RecordingBackendTests(unittest.TestCase):
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
