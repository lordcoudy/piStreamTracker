"""SG90 command safety without GPIO or a PWM daemon."""

import copy
import math
import time
from unittest.mock import patch

import pytest

from pistream.servo import (
    ServoController, angle_to_pulse, default_servo_config, validate_servo_config,
)


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class Pigpio:
    connected = True

    def __init__(self):
        self.commands = []
        self.closed = False

    def set_servo_pulsewidth(self, pin, pulse):
        self.commands.append((pin, pulse))
        return 0

    def stop(self):
        self.closed = True


@pytest.fixture
def rig(tmp_path):
    config = default_servo_config()
    config['calibrated'] = True
    client, clock = Pigpio(), Clock()
    controller = ServoController(
        config, client_factory=lambda: client, clock=clock,
        start_worker=False, lock_dir=tmp_path,
    )
    yield controller, client, clock
    controller.disconnect()


def test_working_limits_do_not_rescale_calibration_and_inversion():
    axis = default_servo_config()['pan']
    axis.update(min_angle=-10, max_angle=20)
    assert angle_to_pulse(axis, 20) == pytest.approx(1611, abs=1)
    assert angle_to_pulse(axis, 100) == angle_to_pulse(axis, 20)
    axis['invert'] = True
    assert angle_to_pulse(axis, 20) == pytest.approx(1389, abs=1)
    assert angle_to_pulse(axis, -100) == pytest.approx(1556, abs=1)


def test_different_users_cannot_acquire_the_same_host_gpio(tmp_path):
    config = default_servo_config()
    with patch('os.getuid', return_value=1001):
        first = ServoController(config, client_factory=Pigpio,
                                start_worker=False, lock_dir=tmp_path)
    second_client = Pigpio()
    try:
        with patch('os.getuid', return_value=1002):
            second = ServoController(config, client_factory=lambda: second_client,
                                     start_worker=False, lock_dir=tmp_path)
        assert not second.connected
        assert 'already owned' in second.error
        assert second_client.commands == []
        second.disconnect()
    finally:
        first.disconnect()


def test_reconnect_confirms_pwm_off_after_previous_shutdown_error(rig):
    controller, client, _ = rig
    client.set_servo_pulsewidth = lambda *_: -1
    controller.disconnect()
    assert controller.status()['pulse_enabled'] is None
    replacement = Pigpio()
    controller._client_factory = lambda: replacement
    assert controller.connect()
    assert controller.status()['pulse_enabled'] is False
    assert controller.status()['error'] is None
    assert replacement.commands == [(18, 0), (19, 0)]


@pytest.mark.parametrize('path,value', [
    (('pan', 'gpio'), True), (('pan', 'gpio'), 18.5),
    (('pan', 'max_speed'), 0), (('pan', 'home'), 90),
    (('pan', 'min_angle'), float('nan')), (('max_dt',), math.inf),
    (('frame_timeout',), -1), (('calibrated',), 'yes'),
    (('home_hold',), 0), (('manual_hold',), 0),
    (('tilt', 'gpio'), 18),
    (('pan', 'calibration', 'min_pulse_us'), 300),
    (('pan', 'calibration', 'max_pulse_us'), 1000),
    (('pan', 'calibration', 'min_angle'), 0),
])
def test_validation_names_unsafe_parameter(path, value):
    config = default_servo_config()
    parent = config
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(ValueError, match='gpio_servo'):
        validate_servo_config(config)


def test_invalid_config_does_not_open_daemon():
    config = default_servo_config()
    config['tilt']['max_speed'] = False
    clients = []
    with pytest.raises(ValueError, match='tilt.max_speed'):
        ServoController(config, client_factory=lambda: clients.append(True))
    assert clients == []


def test_speed_limit_clamps_long_delays_and_axes_independently(rig):
    ctrl, client, clock = rig
    assert all(pulse == 0 for _, pulse in client.commands)
    ctrl.manual_move(1000, -1000)
    clock.advance(10)
    ctrl._tick()
    state = ctrl.status()
    assert state['pan']['target_angle'] == 30
    assert state['tilt']['target_angle'] == -15
    assert state['pan']['commanded_angle'] == pytest.approx(3)
    assert state['tilt']['commanded_angle'] == pytest.approx(-2)
    assert all(pulse == 0 or 1000 <= pulse <= 2000 for _, pulse in client.commands)


def test_auto_is_blocked_before_calibration(rig):
    ctrl, client, clock = rig
    ctrl.config['calibrated'] = False
    ctrl.frame_seen()
    ctrl.update(300, 200)
    clock.advance(.1)
    ctrl._tick()
    assert all(pulse == 0 for _, pulse in client.commands)
    ctrl.manual_move(1, 0)
    clock.advance(.1)
    ctrl._tick()
    assert client.commands


def test_deadzone_and_frame_size_normalize_auto_error(rig):
    ctrl, _, _ = rig
    ctrl.frame_seen()
    ctrl.set_frame_size(640, 480)
    ctrl.update(10, 5)
    assert ctrl.status()['pan']['target_angle'] == 0
    ctrl.update(160, 120)
    first = ctrl.status()
    ctrl.set_frame_size(1280, 960)
    ctrl.update(320, 240)
    assert ctrl.status()['pan']['target_angle'] == first['pan']['target_angle']
    assert ctrl.status()['tilt']['target_angle'] == first['tilt']['target_angle']


def test_loss_and_frame_timeout_cancel_auto_without_new_commands(rig):
    ctrl, client, clock = rig
    ctrl.set_frame_size(640, 480)
    ctrl.frame_seen()
    ctrl.update(300, 200)
    clock.advance(.1)
    ctrl._tick()
    issued = ctrl.status()['pan']['commanded_angle']
    ctrl.target_lost()
    before = list(client.commands)
    clock.advance(.1)
    ctrl._tick()
    assert client.commands == before
    assert ctrl.status()['pan']['target_angle'] == issued
    ctrl.frame_seen()
    ctrl.update(300, 200)
    clock.advance(ctrl.config['frame_timeout'] + .1)
    ctrl._tick()
    assert client.commands == before
    assert ctrl.status()['mode'] == 'idle'


@pytest.mark.parametrize('command', ['manual', 'home'])
def test_stop_cancels_home_manual_hold_and_releases_pulses(rig, command):
    ctrl, client, clock = rig
    ctrl.manual_move(10, 10)
    clock.advance(.1)
    ctrl._tick()
    if command == 'home':
        ctrl.move_to_home()
    ctrl.stop()
    assert not ctrl.connected
    assert client.closed
    assert client.commands[-2:] == [(18, 0), (19, 0)]
    stopped = list(client.commands)
    clock.advance(5)
    ctrl._tick()
    ctrl.update(300, 200)
    ctrl.manual_move(2, 2)
    assert client.commands == stopped


def test_home_and_manual_pause_auto_and_use_limits(rig):
    ctrl, _, clock = rig
    ctrl.manual_move(100, -100)
    ctrl.frame_seen()
    ctrl.update(-300, 200)
    assert ctrl.status()['pan']['target_angle'] == 30
    clock.advance(.1)
    ctrl._tick()
    ctrl.move_to_home()
    assert ctrl.status()['pan']['target_angle'] == 0
    ctrl.update(300, 200)
    assert ctrl.status()['pan']['target_angle'] == 0


def test_second_owner_is_rejected_and_can_connect_after_release(rig, tmp_path):
    ctrl, _, _ = rig
    other_client = Pigpio()
    other = ServoController(default_servo_config(), client_factory=lambda: other_client,
                            start_worker=False, lock_dir=tmp_path)
    assert not other.connected
    assert 'owned' in other.status()['error']
    ctrl.disconnect()
    assert other.connect()
    other.disconnect()


def test_missing_daemon_is_explicit_error_and_no_worker(tmp_path):
    client = Pigpio()
    client.connected = False
    ctrl = ServoController(default_servo_config(), client_factory=lambda: client,
                           lock_dir=tmp_path)
    assert not ctrl.connected
    assert 'pigpiod' in ctrl.status()['error']
    assert client.closed
    assert ctrl._thread is None


def test_worker_watchdog_operates_without_process_frames(tmp_path):
    config = default_servo_config()
    config.update(calibrated=True, frame_timeout=.06, update_interval=.01, max_dt=.02)
    client = Pigpio()
    ctrl = ServoController(config, client_factory=lambda: client, lock_dir=tmp_path)
    try:
        ctrl.frame_seen()
        ctrl.update(500, 300)
        deadline = time.monotonic() + 1
        while ctrl.status()['mode'] != 'idle' and time.monotonic() < deadline:
            time.sleep(.01)
        assert ctrl.status()['mode'] == 'idle'
        commands = list(client.commands)
        time.sleep(.04)
        assert client.commands == commands
    finally:
        ctrl.disconnect()
    assert not ctrl._thread.is_alive()


def test_repeated_connect_does_not_spawn_second_worker(rig):
    ctrl, client, _ = rig
    assert ctrl.connect()
    assert ctrl.connect()
    assert not client.closed


def test_daemon_write_failure_turns_off_other_axis_and_reports_error(rig):
    ctrl, client, clock = rig
    original = client.set_servo_pulsewidth

    def fail(pin, pulse):
        return -1 if pulse and pin == 18 else original(pin, pulse)

    client.set_servo_pulsewidth = fail
    ctrl.manual_move(10, 10)
    clock.advance(.1)
    ctrl._tick()
    assert not ctrl.connected
    assert 'pigpio' in ctrl.status()['error']
    assert (19, 0) in client.commands


def test_validation_does_not_mutate_input():
    config = default_servo_config()
    original = copy.deepcopy(config)
    validated = validate_servo_config(config)
    validated['pan']['home'] = 2
    assert config == original


def test_failed_pwm_disable_remains_visible_after_repeated_cleanup(rig):
    ctrl, client, clock = rig
    ctrl.manual_move(10, 10)
    clock.advance(.1)
    ctrl._tick()
    client.set_servo_pulsewidth = lambda pin, pulse: -1
    ctrl.disconnect()
    state = ctrl.status()
    assert 'disable' in state['error']
    assert state['pulse_enabled'] is None
    ctrl.disconnect()
    assert ctrl.status()['pulse_enabled'] is None


def test_stale_capture_timestamp_does_not_refresh_watchdog(rig):
    ctrl, client, clock = rig
    ctrl.frame_seen(clock() - 10)
    ctrl.update(500, 300)
    clock.advance(.1)
    ctrl._tick()
    assert all(pulse == 0 for _, pulse in client.commands)
