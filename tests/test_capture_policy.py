"""Capture reconnect backoff never gives up."""

import unittest

from pistream.capture import reconnect_wait


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
