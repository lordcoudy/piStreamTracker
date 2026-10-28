#!/usr/bin/env python3
"""
EV3 USB Communication Wrapper
Simplified interface for controlling EV3 motors via USB
"""

import math
import time
from threading import RLock, local

import ev3_dc as ev3


def _dispose_usb_device(device):
    """Release the PyUSB handle and claimed interfaces on Raspberry Pi/Linux."""
    import usb.util

    usb.util.dispose_resources(device)


class _BoundedUSBTransport:
    """Bound this EV3's library transactions, including nested reply waits.

    ev3-dc 0.9.10.2 passes an infinite USB read timeout and can strand its
    lock when read/write raises. The adapter preserves its protocol handling
    while supplying an absolute deadline and restoring lock depth on exit.
    Motor/Jukebox children share this same physical instance.
    """

    def __init__(self, physical, timeout=1.0):
        self._physical = physical
        self._raw_device = physical._device
        self._timeout = timeout
        self._lock = RLock()
        self._thread = local()
        self._closed = False
        methods = {name: getattr(physical, name) for name in (
            'send_direct_cmd', 'wait_for_reply', 'send_system_cmd', '_wait_for_system_reply'
        )}
        physical._device = self
        physical._lock = self
        for name, method in methods.items():
            def bounded(*args, _method=method, **kwargs):
                return self._call(_method, *args, **kwargs)
            setattr(physical, name, bounded)

    def _depth(self):
        return getattr(self._thread, 'depth', 0)

    def acquire(self, *args, **kwargs):
        acquired = self._lock.acquire(*args, **kwargs)
        if acquired:
            self._thread.depth = self._depth() + 1
        return acquired

    def release(self):
        self._lock.release()
        self._thread.depth = self._depth() - 1

    def _call(self, method, *args, **kwargs):
        if self._closed:
            raise RuntimeError('EV3 USB connection is closed')
        previous = getattr(self._thread, 'deadline', None)
        deadline = time.monotonic() + self._timeout
        if previous is not None:
            deadline = min(deadline, previous)
        baseline = self._depth()
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not self.acquire(timeout=remaining):
            raise TimeoutError('EV3 USB transaction lock timed out')
        self._thread.deadline = deadline
        try:
            if self._closed:
                raise RuntimeError('EV3 USB connection is closed')
            return method(*args, **kwargs)
        finally:
            # Nested library methods may release a lock acquired by their
            # caller, or raise before releasing it. Release only this thread's
            # remaining acquisitions, never another transaction's lock.
            while self._depth() > baseline:
                self.release()
            self._thread.deadline = previous

    def _milliseconds(self, requested):
        if self._closed:
            raise RuntimeError('EV3 USB connection is closed')
        deadline = getattr(self._thread, 'deadline', None)
        remaining = self._timeout if deadline is None else deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('EV3 USB transaction timed out')
        timeout = max(1, math.ceil(remaining * 1000))
        return min(timeout, requested) if requested and requested > 0 else timeout

    def read(self, endpoint, size, timeout=None):
        return self._raw_device.read(endpoint, size, self._milliseconds(timeout))

    def write(self, endpoint, data, timeout=None):
        return self._raw_device.write(endpoint, data, self._milliseconds(timeout))

    def close(self):
        if self._closed:
            return

        def dispose():
            _dispose_usb_device(self._raw_device)
            self._physical._devices.discard(self._physical._host)
            self._closed = True
            self._raw_device = None

        self._call(dispose)

# Port mappings
MOTOR_PORTS = {'a': ev3.PORT_A, 'b': ev3.PORT_B, 'c': ev3.PORT_C, 'd': ev3.PORT_D}

# LED colors and modes
LED_MODES = {
    ('off', None): ev3.LED_OFF,
    ('red', 'static'): ev3.LED_RED,
    ('red', 'flash'): ev3.LED_RED_FLASH,
    ('red', 'pulse'): ev3.LED_RED_PULSE,
    ('orange', 'static'): ev3.LED_ORANGE,
    ('orange', 'flash'): ev3.LED_ORANGE_FLASH,
    ('orange', 'pulse'): ev3.LED_ORANGE_PULSE,
    ('green', 'static'): ev3.LED_GREEN,
    ('green', 'flash'): ev3.LED_GREEN_FLASH,
    ('green', 'pulse'): ev3.LED_GREEN_PULSE,
}


class Motor:
    """Simple motor wrapper with intuitive interface."""

    def __init__(self, motor):
        self._motor = motor

    def run_to(self, degrees: int, speed: int = 80):
        """Rotate motor by degrees (latest-wins: stop a prior move first)."""
        if self._motor.busy:
            self._motor.stop()
        self._motor.start_move_by(degrees, speed=speed)

    def stop(self):
        """Stop the motor."""
        self._motor.stop()

    @property
    def position(self) -> int:
        """Get current motor position in degrees."""
        return self._motor.position


class EV3_USB:
    """EV3 USB connection manager."""

    def __init__(self):
        self._ev3 = ev3.EV3(protocol=ev3.USB)
        try:
            self._transport = _BoundedUSBTransport(self._ev3._physical_ev3)
        except Exception:
            physical = self._ev3._physical_ev3
            device = physical._device
            _dispose_usb_device(getattr(device, '_raw_device', device))
            physical._devices.discard(physical._host)
            self._ev3._physical_ev3 = None
            self._ev3 = None
            raise

    def close(self):
        """Close the USB connection to the EV3 brick."""
        if self._ev3 is None:
            return
        # Keep ownership and host registration if resource disposal fails so
        # explicit Stop can retry rather than opening a second USB owner.
        self._transport.close()
        self._ev3._physical_ev3 = None
        self._ev3 = None

    def keep_alive(self):
        """Reset the brick's idle timer without changing timeout or moving motors."""
        if self._ev3 is None:
            raise RuntimeError('EV3 USB connection is closed')
        return self._ev3.send_direct_cmd(ev3.opKeep_Alive + ev3.GVX(0), global_mem=1)

    def Motor(self, port: str) -> Motor:
        """
        Get motor on specified port.

        Args:
            port: Motor port ('a', 'b', 'c', or 'd')

        Returns:
            Motor wrapper object
        """
        port = port.lower()
        if port not in MOTOR_PORTS:
            raise ValueError(f"Invalid motor port: {port}. Use 'a', 'b', 'c', or 'd'")

        motor = ev3.Motor(MOTOR_PORTS[port], protocol=ev3.USB, ev3_obj=self._ev3)
        return Motor(motor)

    def Led(self, color: str, mode: str = 'static'):
        """
        Set EV3 brick LED color and mode.

        Args:
            color: LED color ('red', 'orange', 'green', or 'off')
            mode: LED mode ('static', 'flash', 'pulse')
        """
        if color == 'off':
            led = LED_MODES[('off', None)]
        else:
            key = (color.lower(), mode.lower())
            if key not in LED_MODES:
                raise ValueError(f"Invalid LED color/mode: {color}/{mode}")
            led = LED_MODES[key]

        jukebox = ev3.Jukebox(protocol=ev3.USB, ev3_obj=self._ev3)
        jukebox.change_color(led)
