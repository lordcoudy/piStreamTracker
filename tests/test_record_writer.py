"""Local recording writer initialization and fallback behavior."""

import io
import subprocess
import sys
import time
import unittest
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

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
    def recorder(self, writer=None):
        with patch('pistream.record.cv2.VideoWriter', return_value=writer or FakeVideoWriter()):
            return _RecordingThread('lecture.mp4', 64, 48, 30, encoder='mjpg')

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

    def test_partial_pipe_writes_deliver_the_complete_frame(self):
        class ShortPipe:
            def __init__(self):
                self.data = bytearray()

            def write(self, data):
                chunk = bytes(data[:3])
                self.data.extend(chunk)
                return len(chunk)

        recorder = self.recorder()
        pipe = ShortPipe()
        recorder._proc = SimpleNamespace(stdin=pipe, poll=lambda: None)
        frame = np.arange(12, dtype=np.uint8).reshape((2, 2, 3))
        recorder._write(frame)
        self.assertEqual(bytes(pipe.data), frame.tobytes())

    def test_encoder_death_reports_a_sticky_error(self):
        recorder = self.recorder()
        recorder._proc = SimpleNamespace(stdin=io.BytesIO(), poll=lambda: 17, returncode=17)
        recorder._write(np.zeros((48, 64, 3), dtype=np.uint8))
        self.assertIsNotNone(recorder.error)
        self.assertIn('17', recorder.error)
        first_error = recorder.error
        recorder._proc.returncode = 23
        recorder._write(np.zeros((48, 64, 3), dtype=np.uint8))
        self.assertEqual(recorder.error, first_error)

    def test_encoder_death_without_a_frame_finishes_the_worker(self):
        recorder = self.recorder()
        proc = SimpleNamespace(
            stdin=io.BytesIO(), poll=lambda: proc.returncode, returncode=None,
        )
        recorder._proc = proc
        try:
            recorder.start()
            proc.returncode = 19
            self.assertTrue(recorder.finished.wait(1))
            self.assertIn('19', recorder.error)
            self.assertFalse(recorder.ready)
        finally:
            recorder._stop.set()
            recorder._thread.join(1)
            proc.stdin.close()

    def test_encoder_finalization_timeout_is_killed_and_resources_are_closed(self):
        class Encoder:
            def __init__(self):
                self.stdin = io.BytesIO()
                self.stderr = io.BytesIO()
                self.returncode = None

            def poll(self):
                return self.returncode

            def wait(self, timeout):
                if self.returncode is None:
                    raise subprocess.TimeoutExpired('ffmpeg', timeout)
                return self.returncode

            def kill(self):
                self.returncode = -9

        proc = Encoder()
        with (
            patch('pistream.record.subprocess.Popen', return_value=proc),
            patch('pistream.record.shutil.which', return_value='ffmpeg'),
        ):
            recorder = _RecordingThread('lecture.mp4', 64, 48, 30, encoder='libx264')
        self.assertTrue(recorder.stop())
        self.assertIn('did not finish', recorder.error)
        self.assertTrue(recorder.closed.is_set())
        self.assertTrue(proc.stdin.closed)
        self.assertTrue(proc.stderr.closed)

    def test_writer_exception_marks_finished_but_preserves_resources_until_stop(self):
        class BrokenWriter(FakeVideoWriter):
            def write(self, _frame):
                raise OSError('disk full')

        writer = BrokenWriter()
        recorder = self.recorder(writer)
        recorder.update_frame(np.zeros((48, 64, 3), dtype=np.uint8))
        try:
            recorder.start()
            self.assertTrue(recorder.finished.wait(1))
            self.assertIn('disk full', recorder.error)
            self.assertFalse(recorder.closed.is_set())
            self.assertFalse(writer.released)
            self.assertTrue(recorder.stop())
            self.assertTrue(recorder.closed.is_set())
            self.assertIn('disk full', recorder.error)
        finally:
            recorder.stop()

    def test_live_opencv_writer_is_not_released_after_join_timeout(self):
        entered = Event()
        unblock = Event()

        class BlockedWriter(FakeVideoWriter):
            def write(self, _frame):
                entered.set()
                unblock.wait(8)

        writer = BlockedWriter()
        recorder = self.recorder(writer)
        recorder.update_frame(np.zeros((48, 64, 3), dtype=np.uint8))
        recorder.start()
        try:
            self.assertTrue(entered.wait(1))
            self.assertFalse(recorder.stop())
            self.assertFalse(writer.released)
            self.assertFalse(recorder.closed.is_set())
        finally:
            unblock.set()
            recorder.stop()
        self.assertTrue(writer.released)
        self.assertTrue(recorder.closed.is_set())

    def test_nonreading_encoder_is_killed_before_pipe_close_can_block(self):
        proc = subprocess.Popen(
            [sys.executable, '-c', 'import time; time.sleep(30)'],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
        )
        with (
            patch('pistream.record.subprocess.Popen', return_value=proc),
            patch('pistream.record.shutil.which', return_value='ffmpeg'),
        ):
            recorder = _RecordingThread('lecture.mp4', 1280, 960, 30, encoder='libx264')
        entered = Event()
        original_write = recorder._write

        def write(frame):
            entered.set()
            original_write(frame)

        recorder._write = write
        recorder.update_frame(np.zeros((960, 1280, 3), dtype=np.uint8))
        recorder.start()
        completed = Event()
        result = []

        def stop():
            result.append(recorder.stop())
            completed.set()

        stopper = Thread(target=stop, daemon=True)
        try:
            self.assertTrue(entered.wait(1))
            time.sleep(0.02)
            stopper.start()
            self.assertTrue(completed.wait(5), 'stop blocked while the encoder did not read stdin')
            self.assertEqual(result, [True])
            self.assertIsNotNone(proc.poll())
            self.assertTrue(recorder.finished.is_set())
            self.assertTrue(recorder.closed.is_set())
            self.assertIsNotNone(recorder.error)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=2)
            if stopper.ident is not None:
                stopper.join(2)
            recorder.stop()

    def test_normal_encoder_finalization_receives_the_complete_frame(self):
        proc = subprocess.Popen(
            [sys.executable, '-c',
             'import sys; data = sys.stdin.buffer.read(); print(len(data))'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
        )
        with (
            patch('pistream.record.subprocess.Popen', return_value=proc),
            patch('pistream.record.shutil.which', return_value='ffmpeg'),
        ):
            recorder = _RecordingThread('lecture.mp4', 64, 48, 1, encoder='libx264')
        wrote = Event()
        original_write = recorder._write

        def write(frame):
            original_write(frame)
            wrote.set()

        recorder._write = write
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        recorder.update_frame(frame)
        try:
            recorder.start()
            self.assertTrue(wrote.wait(1))
            self.assertTrue(recorder.stop())
            self.assertEqual(proc.stdout.read().strip(), str(frame.nbytes).encode())
            self.assertIsNone(recorder.error)
            self.assertTrue(recorder.finished.is_set())
            self.assertTrue(recorder.closed.is_set())
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=2)
            proc.stdout.close()
            recorder.stop()

    def test_unavailable_opencv_writer_fails_before_recording_state_changes(self):
        writer = FakeVideoWriter(opened=False)
        with patch('pistream.record.cv2.VideoWriter', return_value=writer):
            with self.assertRaisesRegex(RuntimeError, 'Could not open'):
                _RecordingThread('lecture.mp4', 64, 48, 30, encoder='mjpg')
        self.assertTrue(writer.released)
