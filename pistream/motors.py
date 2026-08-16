"""EV3 pan/tilt controller."""

import logging
import time
from threading import Lock
from typing import Optional

from pistream.motors_math import compute_axis_command, motors_held

logger = logging.getLogger(__name__)


class MotorController:
    """EV3 motor controller for pan/tilt tracking."""

    def __init__(self, config: dict):
        self.deadzone_x = config['deadzone']['x']
        self.deadzone_y = config['deadzone']['y']
        self.speed_factor = min(config['speed_factor'], 2.0)
        self.max_speed = min(config['max_speed'], 100)
        self.invert_x = config['invert']['x']
        self.invert_y = config['invert']['y']
        self.cooldown = config['cooldown']
        self.home_hold = float(config.get('home_hold', 3.0))
        self.pan_port = config['ports']['pan']
        self.tilt_port = config['ports']['tilt']

        self._ev3 = None
        self._pan = None
        self._tilt = None
        self._pan_home = 0
        self._tilt_home = 0
        self._last_cmd = 0.0
        self._hold_until = 0.0
        self._io_lock = Lock()
        self._cam_w = 1280
        self._cam_h = 960
        self.connected = False

        if config['enabled']:
            self.connect()

    def connect(self) -> bool:
        """Connect to EV3 via USB."""
        with self._io_lock:
            try:
                from pistream.ev3_usb import EV3_USB
                logger.info("Connecting to EV3...")
                self._ev3 = EV3_USB()
                self._pan = self._ev3.Motor(self.pan_port)
                self._tilt = self._ev3.Motor(self.tilt_port)
                self.connected = True

                try:
                    self._pan_home = self._pan.position
                    self._tilt_home = self._tilt.position
                except Exception:
                    self._pan_home = 0
                    self._tilt_home = 0

                try:
                    self._ev3.Led('green', 'pulse')
                except Exception:
                    pass

                self._stop_unlocked()
                logger.info(f"EV3 connected (home: pan={self._pan_home}, tilt={self._tilt_home})")
                return True

            except Exception as e:
                logger.warning(f"EV3 connection failed: {e}")
                self.connected = False
                return False

    def set_frame_size(self, width: int, height: int):
        """Update frame dimensions for motor calculations."""
        self._cam_w = width
        self._cam_h = height

    def _drive_axis(self, motor, shift: int, deadzone: int, invert: bool,
                    frame_dim: int, scale: float):
        """Drive a single motor axis based on shift from center."""
        cmd = compute_axis_command(
            shift, deadzone, invert, frame_dim, scale,
            self.speed_factor, self.max_speed,
        )
        if cmd is None:
            motor.stop()
            return
        degrees, speed = cmd
        motor.run_to(degrees=degrees, speed=speed)

    def update(self, shift_x: int, shift_y: int):
        """Update motors based on target offset from center."""
        if not self.connected:
            return

        with self._io_lock:
            if not self.connected:
                return
            now = time.monotonic()
            if motors_held(now, self._hold_until):
                return
            if now - self._last_cmd < self.cooldown:
                return
            try:
                self._drive_axis(self._pan, shift_x, self.deadzone_x,
                                 self.invert_x, self._cam_w, 128.0)
                self._drive_axis(self._tilt, shift_y, self.deadzone_y,
                                 self.invert_y, self._cam_h, 96.0)
                self._last_cmd = now
            except Exception as e:
                logger.debug(f"Motor error: {e}")

    def _stop_unlocked(self):
        try:
            if self._pan:
                self._pan.stop()
            if self._tilt:
                self._tilt.stop()
        except Exception:
            pass

    def stop(self):
        """Stop all motors. No-op while a home/manual hold is active."""
        if not self.connected:
            return
        with self._io_lock:
            if not self.connected:
                return
            if motors_held(time.monotonic(), self._hold_until):
                return
            self._stop_unlocked()

    def move_to_home(self):
        """Move motors back to their initial (home) position and hold auto-track."""
        if not self.connected:
            return
        with self._io_lock:
            if not self.connected:
                return
            try:
                pan_pos = self._pan.position
                tilt_pos = self._tilt.position
                pan_delta = self._pan_home - pan_pos
                tilt_delta = self._tilt_home - tilt_pos
                speed = max(int(self.max_speed * 0.6), 10)
                if abs(pan_delta) > 2:
                    self._pan.run_to(degrees=pan_delta, speed=speed)
                if abs(tilt_delta) > 2:
                    self._tilt.run_to(degrees=tilt_delta, speed=speed)
                self._hold_until = time.monotonic() + self.home_hold
                logger.info(
                    f"Moving to home (pan={pan_delta:+d}, tilt={tilt_delta:+d}); "
                    f"auto-track held {self.home_hold:.1f}s"
                )
            except Exception as e:
                logger.warning(f"Move to home failed: {e}")

    def manual_move(self, pan_degrees: int = 0, tilt_degrees: int = 0,
                    speed: Optional[int] = None):
        """Manually move camera by given degrees. Pauses auto-track briefly."""
        if not self.connected:
            return
        if speed is None:
            speed = max(int(self.max_speed * 0.5), 10)
        with self._io_lock:
            if not self.connected:
                return
            try:
                if pan_degrees != 0 and self._pan:
                    d = -pan_degrees if self.invert_x else pan_degrees
                    self._pan.run_to(degrees=d, speed=speed)
                if tilt_degrees != 0 and self._tilt:
                    d = -tilt_degrees if self.invert_y else tilt_degrees
                    self._tilt.run_to(degrees=d, speed=speed)
                self._hold_until = time.monotonic() + max(self.cooldown, 1.0)
            except Exception as e:
                logger.debug(f"Manual move error: {e}")

    def disconnect(self):
        """Disconnect from EV3."""
        with self._io_lock:
            if not self.connected:
                return
            self._stop_unlocked()
            try:
                if self._ev3:
                    self._ev3.Led('orange', 'static')
            except Exception:
                pass
            try:
                if self._ev3:
                    self._ev3.close()
            except Exception:
                pass
            self._pan = self._tilt = None
            self._ev3 = None
            self.connected = False
            logger.info("EV3 disconnected")
