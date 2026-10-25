"""Small pan/tilt boundary; hardware imports stay inside the selected backend."""

from typing import Protocol

from pistream.config import motor_backend


class PanTiltController(Protocol):
    connected: bool

    def connect(self) -> bool: ...
    def set_frame_size(self, width: int, height: int): ...
    def update(self, shift_x: int, shift_y: int): ...
    def target_lost(self): ...
    def frame_seen(self, timestamp: float | None = None): ...
    def stop(self): ...
    def disconnect(self): ...
    def move_to_home(self): ...
    def manual_move(self, pan_degrees=0, tilt_degrees=0, speed=None): ...
    def status(self) -> dict: ...


class NoMotors:
    connected = False
    backend = 'none'

    def connect(self):
        return False

    def set_frame_size(self, width, height):
        pass

    def update(self, shift_x, shift_y):
        pass

    def frame_seen(self, timestamp=None):
        pass

    def stop(self):
        pass

    target_lost = disconnect = move_to_home = stop

    def manual_move(self, pan_degrees=0, tilt_degrees=0, speed=None):
        pass

    def status(self):
        return {'backend': self.backend, 'connected': False, 'error': None}


def create_motor_controller(config: dict) -> PanTiltController:
    backend = motor_backend(config)
    if backend == 'none':
        return NoMotors()
    if backend == 'gpio_servo':
        from pistream.servo import ServoController
        return ServoController(config['gpio_servo'])
    if backend != 'ev3':
        raise ValueError(f'Unknown motors.backend: {backend}')
    from pistream.motors import MotorController
    return MotorController({**config['ev3'], 'enabled': True})
