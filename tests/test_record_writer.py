"""Local recording writer initialization and fallback behavior."""

import unittest
from unittest.mock import patch

from pistream.record import _RecordingThread


class FakeVideoWriter:
    def __init__(self, opened=True):
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def release(self):
        self.released = True

    def write(self, _frame):
        return None


class RecordingWriterTests(unittest.TestCase):
    def test_mjpg_backend_uses_avi_fallback_without_ffmpeg(self):
        writer = FakeVideoWriter()
        with (
            patch('pistream.record.cv2.VideoWriter', return_value=writer),
            patch('pistream.record.subprocess.Popen') as popen,
        ):
            recorder = _RecordingThread('lecture.mp4', 64, 48, 30, encoder='mjpg')
            self.assertEqual(recorder.output_path, 'lecture.avi')
            self.assertTrue(recorder.ready)
            recorder.stop()

        popen.assert_not_called()
        self.assertTrue(writer.released)

    def test_unavailable_opencv_writer_fails_before_recording_state_changes(self):
        writer = FakeVideoWriter(opened=False)
        with patch('pistream.record.cv2.VideoWriter', return_value=writer):
            with self.assertRaisesRegex(RuntimeError, 'Could not open'):
                _RecordingThread('lecture.mp4', 64, 48, 30, encoder='mjpg')
        self.assertTrue(writer.released)
