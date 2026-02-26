#!/usr/bin/env python3
"""
EV3 USB Communication Wrapper
Simplified interface for controlling EV3 motors and sensors via USB
"""

import ev3_dc as ev3

# Port mappings
MOTOR_PORTS = {'a': ev3.PORT_A, 'b': ev3.PORT_B, 'c': ev3.PORT_C, 'd': ev3.PORT_D}
SENSOR_PORTS = {1: ev3.PORT_1, 2: ev3.PORT_2, 3: ev3.PORT_3, 4: ev3.PORT_4}

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

    def run(self, direction: int = 1, speed: int = 80):
        """Run motor continuously at given speed."""
        if not self._motor.busy:
            self._motor.start_move(direction=direction, speed=speed)

    def run_to(self, degrees: int, speed: int = 80):
        """Rotate motor to specified degrees."""
        if not self._motor.busy:
            self._motor.start_move_by(degrees, speed=speed)

    def stop(self):
        """Stop the motor."""
        if self._motor.busy:
            self._motor.stop()

    @property
    def position(self) -> int:
        """Get current motor position in degrees."""
        return self._motor.position

    @property
    def busy(self) -> bool:
        """Check if motor is currently moving."""
        return self._motor.busy


class EV3_USB:
    """EV3 USB connection manager."""

    def __init__(self):
        self._ev3 = ev3.EV3(protocol=ev3.USB)

    def close(self):
        """Properly close the USB connection to the EV3 brick."""
        if self._ev3 is None:
            return
        try:
            self._ev3.__del__()
        except Exception:
            pass
        self._ev3 = None

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

    def Sensor(self, port: int, sensor_type: str):
        """
        Get sensor on specified port.

        Args:
            port: Sensor port (1, 2, 3, or 4)
            sensor_type: Type of sensor ('color', 'touch', 'ultrasonic', 'infrared', 'gyro')

        Returns:
            Sensor object
        """
        if port not in SENSOR_PORTS:
            raise ValueError(f"Invalid sensor port: {port}. Use 1, 2, 3, or 4")

        ev_port = SENSOR_PORTS[port]
        sensor_classes = {
            'color': ev3.Color,
            'touch': ev3.Touch,
            'ultrasonic': ev3.Ultrasonic,
            'infrared': ev3.Infrared,
            'gyro': ev3.Gyro
        }

        sensor_type = sensor_type.lower()
        if sensor_type not in sensor_classes:
            raise ValueError(f"Invalid sensor type: {sensor_type}. Use: {', '.join(sensor_classes.keys())}")

        return sensor_classes[sensor_type](port=ev_port, protocol=ev3.USB, ev3_obj=self._ev3)

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

    def Sound(self, volume: int = 80):
        """
        Get sound interface.

        Args:
            volume: Sound volume (0-100)

        Returns:
            Jukebox object for playing sounds
        """
        return ev3.Jukebox(protocol=ev3.USB, ev3_obj=self._ev3, volume=volume)

    def Voice(self):
        """
        Get text-to-speech interface.

        Note: Requires WiFi connection on host PC.

        Returns:
            Voice object for text-to-speech
        """
        return ev3.Voice(protocol=ev3.USB, ev3_obj=self._ev3)

    def get_status(self) -> dict:
        """
        Get status of all connected sensors and motors.

        Returns:
            Dictionary with sensor/motor information
        """
        self._ev3._physical_ev3.introspection(self._ev3._verbosity)
        sensors = self._ev3._physical_ev3._introspection["sensors"]

        type_names = {
            7: 'large_motor', 8: 'medium_motor',
            16: 'touch', 29: 'color', 30: 'ultrasonic', 32: 'gyro', 33: 'infrared',
            124: 'none', 125: 'none'
        }

        def get_type(type_id):
            return type_names.get(type_id, f'unknown({type_id})' if type_id else None)

        return {
            'sensors': {
                'port_1': get_type(sensors[ev3.PORT_1]["type"]),
                'port_2': get_type(sensors[ev3.PORT_2]["type"]),
                'port_3': get_type(sensors[ev3.PORT_3]["type"]),
                'port_4': get_type(sensors[ev3.PORT_4]["type"]),
            },
            'motors': {
                'port_a': get_type(sensors[ev3.PORT_A_SENSOR]["type"]),
                'port_b': get_type(sensors[ev3.PORT_B_SENSOR]["type"]),
                'port_c': get_type(sensors[ev3.PORT_C_SENSOR]["type"]),
                'port_d': get_type(sensors[ev3.PORT_D_SENSOR]["type"]),
            }
        }
