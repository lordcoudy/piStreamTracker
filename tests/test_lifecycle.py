"""Web tracker start/stop lifecycle."""

import time
import unittest
from threading import Event, Thread, Timer
from unittest.mock import Mock, patch

from pistream.lifecycle import TrackerLifecycle


class FakeTracker:
    def __init__(self, connect_ok=True, connect_delay=0.0):
        self.connect_ok = connect_ok
        self.connect_delay = connect_delay
        self.running = False
        self.cleaned = 0
        self.connected = False
        self.cleanup_pending = False
        self.remote_stop_pending = False
        self.local_recording_path = None
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
    def test_cancelled_guard_prevents_tracker_acquisition(self):
        cancelled = Event()
        cancelled.set()
        factory = Mock(side_effect=FakeTracker)
        life = TrackerLifecycle(factory, _loop_until_stopped)

        self.assertEqual(life.start(unless_cancelled=cancelled)['status'], 'cancelled')
        factory.assert_not_called()
        self.assertIsNone(life.tracker)
        self.assertIsNone(life.thread)

    def test_stop_during_guarded_connect_cancels_admitted_worker(self):
        connecting = Event()
        release_connect = Event()
        cancelled = Event()

        class GatedTracker(FakeTracker):
            def connect(self):
                connecting.set()
                release_connect.wait(2)
                return super().connect()

        fake = GatedTracker()
        factory = Mock(return_value=fake)
        life = TrackerLifecycle(factory, _loop_until_stopped, join_timeout=0.01)
        results = []
        starter = Thread(target=lambda: results.append(life.start(
            connect_timeout=1, unless_cancelled=cancelled)), daemon=True)
        starter.start()
        try:
            self.assertTrue(connecting.wait(1))
            cancelled.set()
            self.assertEqual(life.stop()['status'], 'stopping')
            release_connect.set()
            starter.join(1)
            self.assertFalse(starter.is_alive())
            self.assertEqual(results[0]['status'], 'error')
            self.assertFalse(fake.running)
            self.assertFalse(fake._loop_entered.is_set())
            self.assertEqual(fake.cleaned, 1)
            self.assertEqual(life.start(unless_cancelled=cancelled)['status'], 'cancelled')
            factory.assert_called_once_with()
        finally:
            release_connect.set()
            life.stop()
            starter.join(1)

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

    def test_timeout_does_not_forget_live_connect_worker(self):
        fake = FakeTracker(connect_ok=True, connect_delay=0.2)
        life = TrackerLifecycle(lambda: fake, _loop_until_stopped, join_timeout=0.01)
        result = life.start(connect_timeout=0.01)
        self.assertEqual(result['status'], 'error')
        self.assertIs(life.tracker, fake)
        self.assertEqual(life.start(connect_timeout=0.01)['status'], 'stopping')
        time.sleep(0.25)
        self.assertEqual(life.stop()['status'], 'ok')
        self.assertIsNone(life.tracker)

    def test_connect_exception_is_reported_as_start_error(self):
        class BrokenTracker(FakeTracker):
            def connect(self):
                raise RuntimeError('camera exploded')

        life = TrackerLifecycle(BrokenTracker, _loop_until_stopped)
        result = life.start(connect_timeout=1.0)
        self.assertEqual(result['status'], 'error')
        self.assertIn('camera exploded', result['message'])
        self.assertIsNone(life.tracker)

    def test_cleanup_is_safe_to_call_twice_on_tracker(self):
        # HumanTracker.cleanup must not break if invoked after lifecycle finally.
        fake = FakeTracker(connect_ok=True)
        fake.cleanup()
        fake.cleanup()
        self.assertEqual(fake.cleaned, 2)

    def test_failed_camera_stop_is_retained_and_only_explicit_stop_retries(self):
        class PendingTracker(FakeTracker):
            def __init__(self):
                super().__init__()
                self.stop_attempts = 0

            def stop_recording(self):
                self.stop_attempts += 1
                self.remote_stop_pending = self.stop_attempts < 3
                self.cleanup_pending = self.remote_stop_pending
                return not self.remote_stop_pending

            def cleanup(self):
                super().cleanup()
                self.stop_recording()

        fake = PendingTracker()
        factory_calls = []

        def factory():
            factory_calls.append(True)
            return fake

        life = TrackerLifecycle(factory, _loop_until_stopped)
        self.assertEqual(life.start()['status'], 'ok')
        self.assertEqual(life.stop()['status'], 'recording_stop_pending')
        self.assertIs(life.tracker, fake)
        self.assertFalse(life.stopping)
        self.assertEqual(fake.stop_attempts, 1)
        self.assertEqual(life.start()['status'], 'recording_stop_pending')
        self.assertEqual(len(factory_calls), 1)
        self.assertEqual(fake.stop_attempts, 1)
        self.assertEqual(life.stop()['status'], 'recording_stop_pending')
        self.assertEqual(fake.stop_attempts, 2)
        self.assertEqual(life.stop()['status'], 'ok')
        self.assertEqual(fake.stop_attempts, 3)
        self.assertEqual(fake.cleaned, 1)
        self.assertIsNone(life.tracker)

    def test_local_cleanup_pending_blocks_start_until_manual_stop_closes_it(self):
        class PendingTracker(FakeTracker):
            def cleanup(self):
                super().cleanup()
                self.cleanup_pending = True

            def stop_recording(self):
                self.cleanup_pending = False
                return True

        fake = PendingTracker()
        life = TrackerLifecycle(lambda: fake, _loop_until_stopped)
        self.assertEqual(life.start()['status'], 'ok')
        self.assertEqual(life.stop()['status'], 'stopping')
        self.assertIs(life.tracker, fake)
        self.assertTrue(life.stopping)
        self.assertEqual(life.start()['status'], 'stopping')
        self.assertEqual(life.stop()['status'], 'ok')
        self.assertIsNone(life.tracker)

    def test_unclosed_local_file_remains_stopping_even_with_remote_stop_pending(self):
        fake = FakeTracker()
        fake.cleanup_pending = True
        fake.remote_stop_pending = True
        fake.local_recording_path = 'unfinished.avi'
        life = TrackerLifecycle(lambda: fake, _loop_until_stopped)
        life.tracker = fake
        self.assertTrue(life.stopping)
        self.assertEqual(life.start()['status'], 'stopping')

    def test_stop_during_live_cleanup_does_not_start_a_second_recording_stop(self):
        cleanup_started = Event()
        allow_cleanup = Event()

        class SlowCleanup(FakeTracker):
            def __init__(self):
                super().__init__()
                self.stop_attempts = 0

            def cleanup(self):
                super().cleanup()
                self.remote_stop_pending = True
                self.cleanup_pending = True
                self.stop_attempts += 1
                cleanup_started.set()
                allow_cleanup.wait(3)

        fake = SlowCleanup()
        life = TrackerLifecycle(lambda: fake, _loop_until_stopped, join_timeout=0.01)
        life.start()
        try:
            self.assertEqual(life.stop()['status'], 'stopping')
            self.assertTrue(cleanup_started.wait(1))
            self.assertTrue(life.stopping)
            self.assertEqual(life.stop()['status'], 'stopping')
            self.assertEqual(fake.stop_attempts, 1)
            self.assertEqual(life.start()['status'], 'stopping')
        finally:
            allow_cleanup.set()
            life.thread.join(1)

    def test_unexpected_loop_exit_exposes_live_cleanup_as_stopping(self):
        cleanup_started = Event()
        allow_cleanup = Event()
        loop_exit = Event()

        class SlowCleanup(FakeTracker):
            def cleanup(self):
                super().cleanup()
                cleanup_started.set()
                allow_cleanup.wait(3)

        def loop(_tracker):
            loop_exit.wait(3)

        life = TrackerLifecycle(SlowCleanup, loop)
        self.assertEqual(life.start()['status'], 'ok')
        self.assertFalse(life.stopping)
        loop_exit.set()
        try:
            self.assertTrue(cleanup_started.wait(1))
            self.assertTrue(life.stopping)
            self.assertEqual(life.start()['status'], 'stopping')
        finally:
            allow_cleanup.set()
            life.stop()

    def test_stop_during_running_assignment_does_not_restart_cancelled_worker(self):
        assigning_running = Event()
        allow_assignment = Event()

        class GatedTracker(FakeTracker):
            def __setattr__(self, name, value):
                if name == 'running' and value is True:
                    assigning_running.set()
                    allow_assignment.wait(2)
                super().__setattr__(name, value)

        fake = GatedTracker()
        life = TrackerLifecycle(lambda: fake, _loop_until_stopped, join_timeout=0.01)
        results = []
        starter = Thread(target=lambda: results.append(life.start(connect_timeout=1)), daemon=True)
        starter.start()
        try:
            self.assertTrue(assigning_running.wait(1))
            self.assertEqual(life.stop()['status'], 'stopping')
            allow_assignment.set()
            starter.join(1)
            self.assertFalse(starter.is_alive())
            self.assertEqual(results[0]['status'], 'error')
            self.assertFalse(fake._loop_entered.is_set())
            self.assertFalse(fake.running)
        finally:
            allow_assignment.set()
            life.stop()
            starter.join(1)


class DelayedTrackerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.timers = []

        def timer_factory(delay, callback):
            timer = Timer(delay, callback)
            timer.start = Mock()
            self.timers.append(timer)
            return timer

        timer_patch = patch('pistream.lifecycle.Timer', side_effect=timer_factory)
        timer_patch.start()
        self.addCleanup(timer_patch.stop)
        self.factory = Mock(side_effect=FakeTracker)
        self.life = TrackerLifecycle(self.factory, _loop_until_stopped)
        self.addCleanup(self.life.stop)

    def test_scheduled_start_waits_two_minutes_before_acquiring_tracker(self):
        self.life.schedule_start(120)
        timer = self.timers[0]
        self.assertEqual(timer.interval, 120)
        self.assertTrue(timer.daemon)
        timer.start.assert_called_once_with()
        self.factory.assert_not_called()
        self.assertIsNone(self.life.tracker)

        timer.function()

        self.factory.assert_called_once_with()
        self.assertTrue(self.life.tracker.running)
        self.assertTrue(self.life.tracker.connected)

    def test_manual_start_consumes_pending_autostart(self):
        self.life.schedule_start(120)
        timer = self.timers[0]
        self.assertEqual(self.life.start()['status'], 'ok')
        tracker = self.life.tracker
        self.assertTrue(timer.finished.is_set())
        timer.function()  # A callback already entered before cancellation.
        self.factory.assert_called_once_with()
        self.assertIs(self.life.tracker, tracker)

    def test_manual_stop_cancels_even_an_already_entered_callback(self):
        self.life.schedule_start(120)
        timer = self.timers[0]
        self.assertEqual(self.life.stop()['status'], 'ok')
        self.assertTrue(timer.finished.is_set())
        timer.function()
        self.factory.assert_not_called()
        self.assertIsNone(self.life.tracker)

    def test_preview_cancellation_consumes_pending_autostart_without_a_tracker(self):
        self.life.schedule_start(120)
        timer = self.timers[0]
        self.life.cancel_scheduled_start()
        self.assertTrue(timer.finished.is_set())
        timer.function()
        self.factory.assert_not_called()
        self.assertIsNone(self.life.tracker)

    def test_preview_cancellation_preserves_a_running_tracker(self):
        self.assertEqual(self.life.start()['status'], 'ok')
        tracker = self.life.tracker
        self.life.schedule_start(120)
        timer = self.timers[0]
        self.life.cancel_scheduled_start()
        timer.function()
        self.assertIs(self.life.tracker, tracker)
        self.assertTrue(tracker.running)
        self.assertEqual(tracker.cleaned, 0)
        self.factory.assert_called_once_with()

    def test_preview_cancellation_blocks_callback_already_waiting_for_admission(self):
        self.life.schedule_start(120)
        entered = Event()
        release = Event()
        original_start = self.life._start

        def gated_start(*args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(2))
            return original_start(*args, **kwargs)

        with patch.object(self.life, '_start', side_effect=gated_start):
            callback = Thread(target=self.timers[0].function)
            callback.start()
            try:
                self.assertTrue(entered.wait(1))
                self.life.cancel_scheduled_start()
            finally:
                release.set()
                callback.join(2)
        self.assertFalse(callback.is_alive())
        self.factory.assert_not_called()
        self.assertIsNone(self.life.tracker)

    def test_preview_cancellation_does_not_cancel_an_admitted_manual_start(self):
        connecting = Event()
        release = Event()

        class ConnectingTracker(FakeTracker):
            def connect(self):
                connecting.set()
                release.wait(2)
                return super().connect()

        tracker = ConnectingTracker()
        self.factory.side_effect = lambda: tracker
        results = []
        starter = Thread(target=lambda: results.append(self.life.start(connect_timeout=2)))
        starter.start()
        try:
            self.assertTrue(connecting.wait(1))
            self.life.cancel_scheduled_start()
        finally:
            release.set()
            starter.join(2)
        self.assertFalse(starter.is_alive())
        self.assertEqual(results, [{'status': 'ok'}])
        self.assertIs(self.life.tracker, tracker)
        self.assertTrue(tracker.running)
        self.assertEqual(tracker.cleaned, 0)

    def test_old_callback_cannot_consume_a_replacement_schedule(self):
        self.life.schedule_start(120)
        old_timer = self.timers[0]
        self.life.schedule_start(30)
        self.assertTrue(old_timer.finished.is_set())
        old_timer.function()
        self.factory.assert_not_called()
        self.timers[1].function()
        self.assertTrue(self.life.tracker.running)
        self.factory.assert_called_once_with()

    def test_callback_is_one_shot_even_after_tracker_is_stopped(self):
        self.life.schedule_start(120)
        timer = self.timers[0]
        timer.function()
        self.life.stop()
        timer.function()
        self.factory.assert_called_once_with()
        self.assertIsNone(self.life.tracker)

    def test_autostart_does_not_replace_an_existing_session(self):
        self.assertEqual(self.life.start()['status'], 'ok')
        tracker = self.life.tracker
        self.life.schedule_start(120)
        self.timers[0].function()
        self.assertIs(self.life.tracker, tracker)
        self.factory.assert_called_once_with()

    def test_failed_autostart_is_logged_without_retry(self):
        self.factory.side_effect = lambda: FakeTracker(connect_ok=False)
        self.life.schedule_start(120)
        with self.assertLogs('pistream.lifecycle', level='WARNING') as logs:
            self.timers[0].function()
        self.assertIn('Failed to connect', '\n'.join(logs.output))
        self.factory.assert_called_once_with()
        self.assertIsNone(self.life.tracker)
        self.assertEqual(len(self.timers), 1)


if __name__ == '__main__':
    unittest.main()
