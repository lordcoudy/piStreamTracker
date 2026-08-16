"""Concurrent MJPEG client cap."""

import unittest

from pistream.stream_limit import StreamLimiter


class StreamLimiterTests(unittest.TestCase):
    def test_rejects_over_cap(self):
        lim = StreamLimiter(max_clients=2)
        self.assertTrue(lim.acquire())
        self.assertTrue(lim.acquire())
        self.assertFalse(lim.acquire())
        lim.release()
        self.assertTrue(lim.acquire())
