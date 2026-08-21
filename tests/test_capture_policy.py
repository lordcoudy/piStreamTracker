"""Capture reconnect backoff never gives up."""

import unittest
from unittest.mock import patch

from pistream.capture import VideoCapture, reconnect_wait


class ReconnectWaitTests(unittest.TestCase):
    def test_first_attempts_use_short_delay(self):
        self.assertEqual(reconnect_wait(1), 1.0)
        self.assertEqual(reconnect_wait(9), 1.0)

    def test_every_tenth_failed_burst_backs_off(self):
        self.assertEqual(reconnect_wait(10), 5.0)
        self.assertEqual(reconnect_wait(20), 5.0)

    def test_never_signals_give_up(self):
        for n in range(1, 35):
            self.assertGreater(reconnect_wait(n), 0)


class CaptureCleanupTests(unittest.TestCase):
    def test_failed_initial_read_releases_capture(self):
        class FakeCapture:
            def __init__(self):
                self.released = False

            def isOpened(self):
                return True

            def set(self, *_args):
                return True

            def get(self, *_args):
                return 30

            def read(self):
                return False, None

            def release(self):
                self.released = True

        fake = FakeCapture()
        with patch('pistream.capture.cv2.VideoCapture', return_value=fake):
            capture = VideoCapture('http://camera/stream')
            self.assertFalse(capture.start())

        self.assertTrue(fake.released)
        self.assertFalse(capture.is_open)
