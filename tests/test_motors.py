"""EV3 motor command issuance and home-hold."""

import sys
import unittest
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

# ev3_usb imports ev3_dc at module load; provide a stub so tests run without the brick lib.
_fake_ev3 = MagicMock()
_fake_ev3.PORT_A = 1
_fake_ev3.PORT_B = 2
_fake_ev3.PORT_C = 4
_fake_ev3.PORT_D = 8
_fake_ev3.PORT_1 = 16
_fake_ev3.PORT_2 = 32
_fake_ev3.PORT_3 = 64
_fake_ev3.PORT_4 = 128
_fake_ev3.LED_OFF = 0
_fake_ev3.USB = 'usb'
sys.modules.setdefault('ev3_dc', _fake_ev3)

from pistream.ev3_usb import Motor  # noqa: E402
from pistream.motors_math import compute_axis_command, motors_held  # noqa: E402


class FakeInner:
    def __init__(self, busy=False, position=0):
        self.busy = busy
        self.position = position
        self.moves = []
        self.stops = 0

    def start_move_by(self, degrees, speed=80):
        self.moves.append((degrees, speed))
        self.busy = True

    def start_move(self, direction=1, speed=80):
        self.moves.append(('run', direction, speed))
        self.busy = True

    def stop(self):
        self.stops += 1
        self.busy = False


class MotorRunToTests(unittest.TestCase):
    def test_run_to_issues_move_when_idle(self):
        inner = FakeInner(busy=False)
        Motor(inner).run_to(degrees=15, speed=40)
        self.assertEqual(inner.moves, [(15, 40)])
        self.assertEqual(inner.stops, 0)

    def test_run_to_stops_then_reissues_when_busy(self):
        inner = FakeInner(busy=True)
        Motor(inner).run_to(degrees=-8, speed=20)
        self.assertEqual(inner.stops, 1)
        self.assertEqual(inner.moves, [(-8, 20)])


class AxisCommandTests(unittest.TestCase):
    def test_inside_deadzone_means_stop(self):
        self.assertIsNone(compute_axis_command(40, deadzone=90, invert=False,
                                               frame_dim=1280, scale=128.0,
                                               speed_factor=1.0, max_speed=50))

    def test_outside_deadzone_scales_degrees(self):
        cmd = compute_axis_command(200, deadzone=90, invert=False,
                                   frame_dim=1280, scale=128.0,
                                   speed_factor=1.0, max_speed=50)
        self.assertIsNotNone(cmd)
        degrees, speed = cmd
        self.assertEqual(degrees, 20)  # 200 / (1280/128)
        # (|200|-90) / (1280/2 - 90) * 50 = 110/550 * 50 = 10
        self.assertEqual(speed, 10)
        self.assertGreater(speed, 2)


class HomeHoldTests(unittest.TestCase):
    def test_held_before_deadline(self):
        self.assertTrue(motors_held(now=10.0, hold_until=12.0))

    def test_released_after_deadline(self):
        self.assertFalse(motors_held(now=12.0, hold_until=12.0))


def _held_controller(home_hold=3.0):
    from pistream.motors import MotorController

    ctrl = MotorController({
        'deadzone': {'x': 90, 'y': 90},
        'speed_factor': 1.0,
        'max_speed': 50,
        'invert': {'x': False, 'y': False},
        'cooldown': 0.0,
        'home_hold': home_hold,
        'ports': {'pan': 'a', 'tilt': 'b'},
        'enabled': False,
    })
    pan_inner = FakeInner(position=40)
    tilt_inner = FakeInner(position=20)
    ctrl._pan = Motor(pan_inner)
    ctrl._tilt = Motor(tilt_inner)
    ctrl._pan_home = 0
    ctrl._tilt_home = 0
    ctrl.connected = True
    return ctrl, pan_inner, tilt_inner


class MotorControllerHoldTests(unittest.TestCase):
    def test_stop_and_update_do_not_command_during_home_hold(self):
        ctrl, pan, tilt = _held_controller(home_hold=3.0)
        ctrl.move_to_home()
        pan_home_moves = list(pan.moves)
        tilt_home_moves = list(tilt.moves)
        self.assertTrue(pan_home_moves)

        ctrl.stop()
        ctrl.update(200, 200)

        self.assertEqual(pan.stops, 0)
        self.assertEqual(tilt.stops, 0)
        self.assertEqual(pan.moves, pan_home_moves)
        self.assertEqual(tilt.moves, tilt_home_moves)

    def test_update_commands_after_hold_expires(self):
        ctrl, pan, tilt = _held_controller(home_hold=3.0)
        ctrl.move_to_home()
        after_home = len(pan.moves)
        ctrl._hold_until = 0.0
        ctrl.update(200, 200)
        self.assertGreater(len(pan.moves), after_home)


class ControlledKeepAliveEvent:
    """Trigger periodic ticks without waiting a real sixty-second interval."""

    def __init__(self):
        self.stopped = Event()
        self.tick = Event()
        self.waiting = Event()
        self.intervals = []

    def set(self):
        self.stopped.set()
        self.tick.set()

    def is_set(self):
        return self.stopped.is_set()

    def wait(self, interval):
        self.intervals.append(interval)
        self.waiting.set()
        self.tick.wait()
        self.tick.clear()
        return self.stopped.is_set()


@pytest.fixture
def keep_alive_controller(monkeypatch):
    from pistream import motors
    from pistream.config import default_config

    events = []
    def make_event():
        event = ControlledKeepAliveEvent()
        events.append(event)
        return event
    monkeypatch.setattr(motors, 'Event', make_event)
    config = default_config()['ev3']
    config['enabled'] = False
    controller = motors.MotorController(config)
    brick = Mock()
    brick.pan, brick.tilt = Mock(position=0), Mock(position=0)
    brick.Motor.side_effect = [brick.pan, brick.tilt]
    with patch('pistream.ev3_usb.EV3_USB', return_value=brick) as factory:
        try:
            yield controller, brick, factory, events
        finally:
            controller.disconnect()


def test_keep_alive_wrapper_sends_only_official_reset_timer_opcode():
    from pistream import ev3_usb

    brick = ev3_usb.EV3_USB.__new__(ev3_usb.EV3_USB)
    brick._ev3 = Mock()
    brick._ev3.send_direct_cmd.return_value = b'\x0a'
    library = SimpleNamespace(opKeep_Alive=b'\x90', GVX=Mock(return_value=b'\x60'))
    with patch.object(ev3_usb, 'ev3', library):
        assert brick.keep_alive() == b'\x0a'
    library.GVX.assert_called_once_with(0)
    brick._ev3.send_direct_cmd.assert_called_once_with(b'\x90\x60', global_mem=1)


def test_keep_alive_never_reopens_a_closed_usb_wrapper():
    from pistream import ev3_usb

    brick = ev3_usb.EV3_USB.__new__(ev3_usb.EV3_USB)
    brick._ev3 = None
    with pytest.raises(RuntimeError, match='closed'):
        brick.keep_alive()


def test_connected_keep_alive_resets_immediately_and_periodically_without_motor_moves(keep_alive_controller):
    controller, brick, factory, events = keep_alive_controller
    assert controller.connect()
    worker = controller._keep_alive_thread
    event = events[-1]
    assert event.waiting.wait(1)
    assert brick.keep_alive.call_count == 1
    assert event.intervals == [60]
    called = Event()
    def keep_alive():
        assert not controller._io_lock.acquire(blocking=False), 'USB commands need the shared lock'
        called.set()
    brick.keep_alive.side_effect = keep_alive
    event.tick.set()
    assert called.wait(1)
    controller.disconnect()
    assert not worker.is_alive()
    factory.assert_called_once_with()
    brick.pan.run_to.assert_not_called()
    brick.tilt.run_to.assert_not_called()


def test_disabled_keep_alive_opens_no_worker_or_extra_usb_command(keep_alive_controller):
    controller, brick, factory, events = keep_alive_controller
    controller.keep_alive_interval = None
    assert controller.connect()
    assert controller._keep_alive_thread is None
    brick.keep_alive.assert_not_called()


def test_repeated_connect_retains_one_usb_owner_and_one_keep_alive_worker(keep_alive_controller):
    controller, brick, factory, events = keep_alive_controller
    assert controller.connect()
    worker = controller._keep_alive_thread
    assert controller.connect()
    assert controller._keep_alive_thread is worker
    factory.assert_called_once_with()
    brick.keep_alive.assert_called_once_with()


def test_disconnect_signals_waiting_worker_before_io_lock_and_joins_outside_it(keep_alive_controller):
    controller, brick, factory, events = keep_alive_controller
    assert controller.connect()
    worker = controller._keep_alive_thread
    event = events[-1]
    assert event.waiting.wait(1)
    with controller._io_lock:
        event.tick.set()  # The keep-alive worker now waits for the USB lock.
        stopper = Thread(target=controller.disconnect)
        stopper.start()
        assert event.stopped.wait(1), 'Disconnect must signal before waiting for the USB lock'
    stopper.join(1)
    assert not stopper.is_alive(), 'Joining while holding USB lock would deadlock'
    assert not worker.is_alive()
    assert not controller.connected
    brick.keep_alive.assert_called_once_with()
    brick.close.assert_called_once_with()
    factory.assert_called_once_with()


def test_disconnect_waits_for_inflight_keep_alive_then_closes_usb_once(keep_alive_controller):
    controller, brick, factory, events = keep_alive_controller
    assert controller.connect()
    event = events[-1]
    assert event.waiting.wait(1)
    entered, release = Event(), Event()
    def keep_alive():
        entered.set()
        assert release.wait(1)
    brick.keep_alive.side_effect = keep_alive
    event.tick.set()
    assert entered.wait(1)
    stopper = Thread(target=controller.disconnect)
    stopper.start()
    try:
        assert event.stopped.wait(1)
        brick.close.assert_not_called()
    finally:
        release.set()
        stopper.join(1)
    assert not stopper.is_alive()
    brick.close.assert_called_once_with()
    factory.assert_called_once_with()
    assert not controller.connected


def test_disconnect_during_connect_also_cancels_replacement_worker_event(keep_alive_controller):
    controller, brick, factory, events = keep_alive_controller
    entered, release = Event(), Event()
    original_event = events[-1]
    def opening_usb():
        entered.set()
        assert release.wait(1)
        return brick
    factory.side_effect = opening_usb
    starter = Thread(target=controller.connect)
    starter.start()
    assert entered.wait(1)
    stopper = Thread(target=controller.disconnect)
    stopper.start()
    try:
        assert original_event.stopped.wait(1)
    finally:
        release.set()
        starter.join(1)
        stopper.join(1)
    assert not starter.is_alive() and not stopper.is_alive()
    assert events[-1] is not original_event
    assert events[-1].is_set()
    assert controller._keep_alive_thread is None
    assert not controller.connected
    brick.close.assert_called_once_with()
    factory.assert_called_once_with()


def test_keep_alive_failure_retains_usb_ownership_and_reports_error_until_success(keep_alive_controller):
    controller, brick, factory, events = keep_alive_controller
    brick.keep_alive.side_effect = OSError('USB is unavailable')
    assert controller.connect()
    assert controller.connected
    assert 'keep-alive' in controller.status()['error']
    assert 'USB is unavailable' in controller.status()['error']
    event = events[-1]
    assert event.waiting.wait(1)
    event.waiting.clear()
    brick.keep_alive.side_effect = None
    event.tick.set()
    assert event.waiting.wait(1)
    assert controller.status()['error'] is None
    factory.assert_called_once_with()


def test_reconnect_starts_new_worker_without_reviving_previous_usb_owner(keep_alive_controller):
    controller, brick, factory, events = keep_alive_controller
    assert controller.connect()
    old_worker, old_event = controller._keep_alive_thread, events[-1]
    controller.disconnect()
    replacement = Mock()
    replacement.Motor.side_effect = [Mock(position=0), Mock(position=0)]
    factory.return_value = replacement
    assert controller.connect()
    assert controller._keep_alive_thread is not old_worker
    assert not old_worker.is_alive()
    assert old_event.is_set()
    replacement.keep_alive.assert_called_once_with()
    brick.keep_alive.assert_called_once_with()


@pytest.mark.parametrize('failure_stage', ['second_motor', 'worker_start'])
def test_partial_connect_failure_releases_usb_before_retry(keep_alive_controller, monkeypatch, failure_stage):
    from pistream import motors

    controller, brick, factory, events = keep_alive_controller
    if failure_stage == 'second_motor':
        brick.Motor.side_effect = [brick.pan, OSError('second motor unavailable')]
    else:
        monkeypatch.setattr(motors.Thread, 'start', Mock(side_effect=RuntimeError('thread unavailable')))
    assert not controller.connect()
    assert not controller.connected
    assert controller._ev3 is None
    assert controller._pan is None and controller._tilt is None
    assert controller._keep_alive_thread is None
    brick.close.assert_called_once_with()
    controller.disconnect()
    brick.close.assert_called_once_with()


def test_disconnect_releases_partial_owner_and_stops_both_axes_after_one_stop_fails(keep_alive_controller):
    controller, brick, factory, events = keep_alive_controller
    controller._ev3 = brick
    controller._pan, controller._tilt = brick.pan, brick.tilt
    brick.pan.stop.side_effect = OSError('pan stop failed')
    controller.disconnect()
    brick.tilt.stop.assert_called_once_with()
    brick.close.assert_called_once_with()
    assert controller._ev3 is None and not controller.connected


def test_failed_usb_close_retains_owner_blocks_connect_and_retries_explicitly(keep_alive_controller):
    controller, brick, factory, events = keep_alive_controller
    assert controller.connect()
    worker = controller._keep_alive_thread
    brick.close.side_effect = OSError('USB disposal failed')
    controller.disconnect()
    assert not controller.connected
    assert not worker.is_alive()
    assert controller.cleanup_pending is True
    assert controller._ev3 is brick
    assert controller._pan is brick.pan and controller._tilt is brick.tilt
    assert 'USB disposal failed' in controller.status()['error']
    assert not controller.connect()
    factory.assert_called_once_with()
    assert brick.close.call_count == 1, 'Connect must not retry or replace a pending USB owner'
    brick.close.side_effect = None
    controller.disconnect()
    assert controller.cleanup_pending is False
    assert controller._ev3 is None
    assert controller._pan is None and controller._tilt is None
    assert controller.status()['error'] is None
    assert brick.close.call_count == 2
    controller.disconnect()
    assert brick.close.call_count == 2


def test_failed_partial_initialization_and_disposal_retains_cleanup_error(keep_alive_controller):
    controller, brick, factory, events = keep_alive_controller
    brick.Motor.side_effect = [brick.pan, OSError('tilt unavailable')]
    brick.close.side_effect = OSError('USB disposal failed')
    assert not controller.connect()
    assert controller.cleanup_pending is True
    assert controller._ev3 is brick
    assert 'USB disposal failed' in controller.status()['error']
    assert not controller.connect()
    factory.assert_called_once_with()
    brick.close.side_effect = None
    controller.disconnect()
    assert controller.cleanup_pending is False


if __name__ == '__main__':
    unittest.main()
