#!/usr/bin/env python3
"""
EV3 USB Communication Wrapper
Simplified interface for controlling EV3 motors via USB
"""

import ev3_dc as ev3

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
        if self._motor.busy:
            self._motor.stop()

    @property
    def position(self) -> int:
        """Get current motor position in degrees."""
        return self._motor.position


class EV3_USB:
    """EV3 USB connection manager."""

    def __init__(self):
        self._ev3 = ev3.EV3(protocol=ev3.USB)

    def close(self):
        """Close the USB connection to the EV3 brick."""
        if self._ev3 is None:
            return
        ev3 = self._ev3
        self._ev3 = None
        for name in ('close', 'disconnect'):
            fn = getattr(ev3, name, None)
            if callable(fn):
                try:
                    fn()
                    return
                except Exception:
                    pass

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
