"""A failed EV3 USB release retains its web tracker until an explicit Stop retry."""

from threading import Event, RLock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pistream.config import default_config
from pistream.lifecycle import TrackerLifecycle
from pistream.motors import MotorController
from pistream.track import HumanTracker


class FakeAxis:
    def __init__(self):
        self.stops = 0

    def stop(self):
        self.stops += 1


class FailingReleaseBrick:
    """A resource owner whose release fails until the operator resolves it."""

    def __init__(self):
        self.release_fails = True
        self.close_calls = 0
        self.closed = False

    def Led(self, *_args):
        pass

    def close(self):
        self.close_calls += 1
        if self.release_fails:
            raise OSError('USB dispose failed')
        self.closed = True


class ClosingResource:
    def __init__(self):
        self.closed = False
        self.stop_calls = 0

    def invalidate(self):
        pass

    def stop(self):
        self.stop_calls += 1
        self.closed = True


def tracker_with_motors(motors):
    """Use the real cleanup path without opening capture, model, USB or files."""
    tracker = HumanTracker.__new__(HumanTracker)
    tracker._record_lock = RLock()
    tracker._motor_control_lock = RLock()
    tracker._recording = False
    tracker._recording_backend = None
    tracker._recording_error = None
    tracker._rec_thread = None
    tracker.remote_stop_pending = False
    tracker._cleaned = False
    tracker._cleanup_ok = True
    tracker._motion_stopped = False
    tracker._async_detector = ClosingResource()
    tracker.capture = ClosingResource()
    tracker._shift_logger = None
    tracker.motors = motors
    tracker.running = False
    tracker.connect = Mock(return_value=True)
    return tracker


def lifecycle_for(tracker):
    entered = Event()
    wake = Event()

    def loop(owner):
        entered.set()
        while owner.running:
            wake.wait(0.005)

    factory = Mock(return_value=tracker)
    lifecycle = TrackerLifecycle(factory, loop, join_timeout=1)
    return lifecycle, factory, entered, wake


def test_failed_motor_release_retains_tracker_until_explicit_stop_retry():
    config = default_config()['ev3']
    config.update(enabled=False, keep_alive_interval=None)
    motors = MotorController(config)
    brick = FailingReleaseBrick()
    motors._ev3 = brick
    motors._pan, motors._tilt = FakeAxis(), FakeAxis()
    motors.connected = True
    tracker = tracker_with_motors(motors)
    capture, detector = tracker.capture, tracker._async_detector
    lifecycle, factory, entered, wake = lifecycle_for(tracker)
    replacement_motors = None
    try:
        assert lifecycle.start()['status'] == 'ok'
        assert entered.wait(1)
        result = lifecycle.stop()
        assert lifecycle.thread is None or not lifecycle.thread.is_alive()
        assert capture.closed and detector.closed
        assert tracker.capture is None and tracker._async_detector is None
        assert tracker._cleaned is True, 'Explicit retry must work after cleanup has short-circuited'
        assert result['status'] == 'stopping'
        assert lifecycle.tracker is tracker, 'Unreleased motors must retain their tracker owner'
        assert motors.cleanup_pending is True
        assert motors._ev3 is brick
        assert tracker.cleanup_pending
        assert not brick.closed

        calls_after_cleanup = brick.close_calls
        assert calls_after_cleanup > 0
        assert lifecycle.start()['status'] == 'stopping'
        assert lifecycle.tracker is tracker
        assert factory.call_count == 1
        assert brick.close_calls == calls_after_cleanup, 'Start must not silently retry USB release'

        brick.release_fails = False
        for _ in range(3):
            assert tracker.cleanup_pending
            assert lifecycle.stopping
        assert brick.close_calls == calls_after_cleanup, 'Status must not silently retry USB release'
        assert lifecycle.start()['status'] == 'stopping'
        assert factory.call_count == 1
        assert not brick.closed

        assert lifecycle.stop()['status'] == 'ok'
        assert lifecycle.tracker is None
        assert brick.closed
        assert motors.cleanup_pending is False
        assert motors._ev3 is None
        assert not tracker.cleanup_pending
        assert capture.stop_calls == detector.stop_calls == 1
        assert factory.call_count == 1

        replacement_motors = MotorController(config)
        replacement_brick = FailingReleaseBrick()
        replacement_brick.release_fails = False
        replacement_motors._ev3 = replacement_brick
        replacement_motors._pan, replacement_motors._tilt = FakeAxis(), FakeAxis()
        replacement_motors.connected = True
        replacement = tracker_with_motors(replacement_motors)
        factory.return_value = replacement
        assert lifecycle.start()['status'] == 'ok'
        assert lifecycle.tracker is replacement and replacement.running
        assert replacement_motors._ev3 is replacement_brick
        assert factory.call_count == 2, 'A replacement session is admitted only after release succeeds'
    finally:
        brick.release_fails = False
        tracker.running = False
        wake.set()
        lifecycle.stop()
        motors.disconnect()
        if replacement_motors is not None:
            replacement_motors.disconnect()


def test_disabling_motors_reports_unconfirmed_usb_release():
    config = default_config()['ev3']
    config.update(enabled=False, keep_alive_interval=None)
    motors = MotorController(config)
    brick = FailingReleaseBrick()
    motors._ev3 = brick
    motors._pan, motors._tilt = FakeAxis(), FakeAxis()
    motors.connected = True
    tracker = tracker_with_motors(motors)
    tracker.running = True
    try:
        assert tracker.set_motors_enabled(False) is False
        assert motors.cleanup_pending is True
        assert motors._ev3 is brick
        assert not brick.closed
        brick.release_fails = False
        assert tracker.set_motors_enabled(False) is True
        assert motors.cleanup_pending is False
        assert brick.closed
    finally:
        brick.release_fails = False
        tracker.cleanup()


@pytest.mark.parametrize('motors', [
    Mock(),
    SimpleNamespace(disconnect=Mock()),
], ids=['generic-mock', 'no-cleanup-property'])
def test_motor_without_explicit_cleanup_pending_does_not_retain_tracker(motors):
    tracker = tracker_with_motors(motors)
    lifecycle, factory, entered, wake = lifecycle_for(tracker)
    try:
        assert lifecycle.start()['status'] == 'ok'
        assert entered.wait(1)
        assert lifecycle.stop()['status'] == 'ok'
        assert not tracker.cleanup_pending
        assert lifecycle.tracker is None
        assert factory.call_count == 1
    finally:
        tracker.running = False
        wake.set()
        lifecycle.stop()
