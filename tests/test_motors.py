"""EV3 motor command issuance and home-hold."""

import sys
import unittest
from unittest.mock import MagicMock

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
        self.assertEqual(speed, 2)     # min(200/100 * 1.0, 50)


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


if __name__ == '__main__':
    unittest.main()
