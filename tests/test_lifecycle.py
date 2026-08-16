"""Web tracker start/stop lifecycle."""

import time
import unittest
from threading import Event

from pistream.lifecycle import TrackerLifecycle


class FakeTracker:
    def __init__(self, connect_ok=True, connect_delay=0.0):
        self.connect_ok = connect_ok
        self.connect_delay = connect_delay
        self.running = False
        self.cleaned = 0
        self.connected = False
        self._loop_entered = Event()

    def connect(self):
        if self.connect_delay:
            time.sleep(self.connect_delay)
        self.connected = self.connect_ok
        return self.connect_ok

    def cleanup(self):
        self.cleaned += 1
        self.running = False


def _loop_until_stopped(tracker):
    tracker._loop_entered.set()
    while tracker.running:
        time.sleep(0.01)


class TrackerLifecycleTests(unittest.TestCase):
    def test_start_fails_when_stream_does_not_connect(self):
        life = TrackerLifecycle(lambda: FakeTracker(connect_ok=False), _loop_until_stopped)
        result = life.start(connect_timeout=2.0)
        self.assertEqual(result['status'], 'error')
        self.assertIn('connect', result['message'].lower())
        self.assertIsNone(life.tracker)

    def test_start_returns_ok_only_after_connect(self):
        fake = FakeTracker(connect_ok=True)
        life = TrackerLifecycle(lambda: fake, _loop_until_stopped)
        result = life.start(connect_timeout=2.0)
        self.assertEqual(result['status'], 'ok')
        self.assertTrue(fake.running)
        life.stop()

    def test_double_start_is_rejected(self):
        fake = FakeTracker(connect_ok=True)
        life = TrackerLifecycle(lambda: fake, _loop_until_stopped)
        self.assertEqual(life.start(connect_timeout=2.0)['status'], 'ok')
        again = life.start(connect_timeout=2.0)
        self.assertEqual(again['status'], 'already_running')
        life.stop()

    def test_stop_is_idempotent_and_cleanup_runs_once(self):
        fake = FakeTracker(connect_ok=True)
        life = TrackerLifecycle(lambda: fake, _loop_until_stopped)
        life.start(connect_timeout=2.0)
        fake._loop_entered.wait(timeout=1.0)
        self.assertEqual(life.stop()['status'], 'ok')
        self.assertEqual(life.stop()['status'], 'ok')
        self.assertEqual(fake.cleaned, 1)
        self.assertIsNone(life.tracker)

    def test_connect_timeout_does_not_leave_a_running_tracker(self):
        fake = FakeTracker(connect_ok=True, connect_delay=0.4)
        life = TrackerLifecycle(lambda: fake, _loop_until_stopped, join_timeout=1.0)
        result = life.start(connect_timeout=0.05)
        self.assertEqual(result['status'], 'error')
        self.assertIn('timed out', result['message'].lower())
        time.sleep(0.5)
        self.assertFalse(fake.running)
        self.assertIsNone(life.tracker)

    def test_cleanup_is_safe_to_call_twice_on_tracker(self):
        # HumanTracker.cleanup must not break if invoked after lifecycle finally.
        fake = FakeTracker(connect_ok=True)
        fake.cleanup()
        fake.cleanup()
        self.assertEqual(fake.cleaned, 2)


if __name__ == '__main__':
    unittest.main()
