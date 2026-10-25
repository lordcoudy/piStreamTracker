"""Position commands for two SG90s using a local, hardware-timed pigpio daemon.

Angles here are commands, never position measurements. Calibration maps angles
to pulses; narrower working limits protect the assembled mechanism and cable.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from pathlib import Path
from threading import Event, RLock, Thread, current_thread

logger = logging.getLogger(__name__)
AXES = ('pan', 'tilt')


def default_servo_config() -> dict:
    def axis(gpio, low, high, speed):
        return {
            'gpio': gpio,
            'calibration': {
                'min_angle': -90.0, 'max_angle': 90.0,
                'min_pulse_us': 1000.0, 'max_pulse_us': 2000.0,
            },
            'min_angle': low, 'max_angle': high, 'home': 0.0,
            'invert': False, 'max_speed': speed,
        }

    return {
        'calibrated': False, 'frame_timeout': 0.75, 'update_interval': 0.02,
        'max_dt': 0.1, 'home_hold': 3.0, 'manual_hold': 1.0,
        'deadzone': 0.08, 'gain': 30.0, 'min_step': 0.2,
        'pan': axis(18, -30.0, 30.0, 30.0),
        'tilt': axis(19, -15.0, 15.0, 20.0),
    }


def validate_servo_config(config: dict) -> dict:
    """Return a normalized copy; reject unsafe settings before opening GPIO."""
    config = copy.deepcopy(config)

    def mapping(value, name):
        if not isinstance(value, dict):
            raise ValueError(f'{name} must be a mapping')
        return value

    def number(parent, key, name, low=-math.inf, high=math.inf, positive=False):
        value = parent.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{name} must be a number')
        if not math.isfinite(value) or not low <= value <= high or (positive and value <= 0):
            raise ValueError(f'{name} must be finite and within permitted bounds')
        parent[key] = float(value)
        return float(value)

    def boolean(parent, key, name):
        if not isinstance(parent.get(key), bool):
            raise ValueError(f'{name} must be true or false')

    mapping(config, 'gpio_servo')
    boolean(config, 'calibrated', 'gpio_servo.calibrated')
    for key in ('frame_timeout', 'update_interval', 'max_dt', 'gain', 'min_step',
                'home_hold', 'manual_hold'):
        number(config, key, f'gpio_servo.{key}', positive=True)
    number(config, 'deadzone', 'gpio_servo.deadzone', low=0, high=0.95)
    if not config['update_interval'] <= config['max_dt'] <= 1:
        raise ValueError('gpio_servo.max_dt must be between update_interval and 1 second')
    if config['frame_timeout'] < config['update_interval']:
        raise ValueError('gpio_servo.frame_timeout must be at least update_interval')

    for name in AXES:
        axis = mapping(config.get(name), f'gpio_servo.{name}')
        prefix = f'gpio_servo.{name}'
        gpio = axis.get('gpio')
        if isinstance(gpio, bool) or not isinstance(gpio, int) or not 0 <= gpio <= 27:
            raise ValueError(f'{prefix}.gpio must be an integer BCM GPIO between 0 and 27')
        boolean(axis, 'invert', f'{prefix}.invert')
        number(axis, 'max_speed', f'{prefix}.max_speed', positive=True)
        cal = mapping(axis.get('calibration'), f'{prefix}.calibration')
        for key in ('min_angle', 'max_angle'):
            number(cal, key, f'{prefix}.calibration.{key}', low=-180, high=180)
            number(axis, key, f'{prefix}.{key}', low=-180, high=180)
        if not cal['min_angle'] < cal['max_angle']:
            raise ValueError(f'{prefix}.calibration.min_angle must be below max_angle')
        for key in ('min_pulse_us', 'max_pulse_us'):
            number(cal, key, f'{prefix}.calibration.{key}', low=500, high=2500)
        if not cal['min_pulse_us'] < cal['max_pulse_us']:
            raise ValueError(f'{prefix}.calibration.min_pulse_us must be below max_pulse_us')
        if not cal['min_angle'] <= axis['min_angle'] < axis['max_angle'] <= cal['max_angle']:
            raise ValueError(f'{prefix}.min_angle/max_angle must be ordered within calibration')
        number(axis, 'home', f'{prefix}.home', low=axis['min_angle'], high=axis['max_angle'])
    if config['pan']['gpio'] == config['tilt']['gpio']:
        raise ValueError('gpio_servo.tilt.gpio must differ from gpio_servo.pan.gpio')
    return config


def angle_to_pulse(axis: dict, angle: float) -> int:
    """Clamp to mechanical limits, then use the independent calibration map."""
    if isinstance(angle, bool) or not isinstance(angle, (float, int)) or not math.isfinite(angle):
        raise ValueError('servo angle must be finite')
    angle = min(axis['max_angle'], max(axis['min_angle'], angle))
    cal = axis['calibration']
    fraction = (angle - cal['min_angle']) / (cal['max_angle'] - cal['min_angle'])
    if axis['invert']:
        fraction = 1.0 - fraction
    pulse = cal['min_pulse_us'] + fraction * (cal['max_pulse_us'] - cal['min_pulse_us'])
    return int(round(min(cal['max_pulse_us'], max(cal['min_pulse_us'], pulse))))


class ServoController:
    """One owner, one latest target per axis, one interruptible motion worker."""

    backend = 'gpio_servo'

    def __init__(self, config: dict, *, client_factory=None, clock=time.monotonic,
                 start_worker=True, lock_dir=None):
        self.config = validate_servo_config(config)
        self._client_factory = client_factory
        self._clock = clock
        self._start_worker = start_worker
        self._lock_dir = Path(lock_dir or '/tmp')
        self._lock = RLock()
        self._pin_locks = []
        self._client = None
        self._thread = None
        self._stop_event = Event()
        self.connected = False
        self.error = None
        self.pulse_enabled = False
        self._ever_issued = False
        self._issued = {name: self.config[name]['home'] for name in AXES}
        self._targets = dict(self._issued)
        self._mode = 'idle'
        self._frame_at = None
        self._hold_until = 0.0
        self._last_tick = clock()
        self._frame_size = (640, 480)
        self.connect()

    def _acquire_pins(self):
        import fcntl

        for gpio in sorted(self.config[name]['gpio'] for name in AXES):
            path = self._lock_dir / f'pistream-gpio-{gpio}.lock'
            handle = path.open('a+')
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                raise RuntimeError(f'GPIO {gpio} already owned by tracker or calibration') from None
            self._pin_locks.append(handle)

    def _release_pins(self):
        for handle in self._pin_locks:
            handle.close()
        self._pin_locks.clear()

    def connect(self) -> bool:
        with self._lock:
            if self.connected:
                return True
            try:
                self._acquire_pins()
                if self._client_factory is None:
                    import pigpio
                    self._client = pigpio.pi('127.0.0.1', 8888)
                else:
                    self._client = self._client_factory()
                if not self._client.connected:
                    raise RuntimeError('Cannot connect to local pigpiod at 127.0.0.1:8888')
                # A previous process may have died while daemon pulses remained active.
                for name in AXES:
                    self._write(name, 0)
                self.connected = True
                self.error = None
                self.pulse_enabled = False
                self._mode = 'idle'
                self._targets = dict(self._issued)
                self._frame_at = None
                self._last_tick = self._clock()
                self._stop_event = Event()
                if self._start_worker:
                    self._thread = Thread(target=self._run, args=(self._stop_event,),
                                          name='GPIOServo', daemon=True)
                    self._thread.start()
                return True
            except Exception as exc:
                self._fail_locked(exc)
                return False

    def _write(self, name, pulse):
        result = self._client.set_servo_pulsewidth(self.config[name]['gpio'], pulse)
        if result is not None and result < 0:
            raise RuntimeError(f'pigpio rejected {name} pulse ({result})')

    def _close_locked(self):
        self._stop_event.set()
        self.connected = False
        self._targets = dict(self._issued)
        self._mode = 'stopped'
        if self._client is not None:
            disable_errors = []
            for name in AXES:
                try:
                    self._write(name, 0)
                except Exception as exc:
                    disable_errors.append(f'{name}: {exc}')
                    logger.warning('Failed to disable %s servo pulses', name, exc_info=True)
            self.pulse_enabled = None if disable_errors else False
            if disable_errors:
                detail = 'Unable to disable PWM: ' + '; '.join(disable_errors)
                self.error = f'{self.error}; {detail}' if self.error else detail
            try:
                self._client.stop()
            except Exception as exc:
                detail = f'Unable to close pigpio: {exc}'
                self.error = f'{self.error}; {detail}' if self.error else detail
                logger.warning('Failed to close pigpio connection', exc_info=True)
            self._client = None
        self._release_pins()

    def _fail_locked(self, exc):
        self.error = str(exc)
        logger.error('GPIO servo controller: %s', exc)
        self._close_locked()

    def disconnect(self):
        with self._lock:
            self._close_locked()
            worker = self._thread
        if worker and worker is not current_thread():
            worker.join(timeout=2.0)
            if worker.is_alive():
                logger.error('GPIO servo worker did not stop')

    def stop(self):
        """Immediately cancel every mode and release pulses, worker and GPIO."""
        self.disconnect()

    def set_frame_size(self, width, height):
        if width <= 0 or height <= 0:
            raise ValueError('servo frame dimensions must be positive')
        with self._lock:
            self._frame_size = (width, height)

    def frame_seen(self, timestamp=None):
        with self._lock:
            self._frame_at = self._clock() if timestamp is None else timestamp

    def _fresh(self, now):
        return self._frame_at is not None and 0 <= now - self._frame_at <= self.config['frame_timeout']

    def target_lost(self):
        """Hold the last issued position; never continue an automatic target."""
        with self._lock:
            if self._mode == 'auto':
                self._targets = dict(self._issued)
                self._mode = 'idle'

    def update(self, shift_x, shift_y):
        with self._lock:
            now = self._clock()
            if not self.connected or not self.config['calibrated'] or not self._fresh(now):
                return
            moving = any(abs(self._targets[n] - self._issued[n]) > self.config['min_step'] for n in AXES)
            if self._mode in ('home', 'manual') and (moving or now < self._hold_until):
                return
            for name, shift, size in zip(AXES, (shift_x, shift_y), self._frame_size):
                if not math.isfinite(shift):
                    raise ValueError('servo target offset must be finite')
                error = max(-1.0, min(1.0, shift / (size / 2.0)))
                delta = 0.0 if abs(error) <= self.config['deadzone'] else error * self.config['gain']
                self._targets[name] = self._clamp(name, self._issued[name] + delta)
            self._mode = 'auto'

    def _clamp(self, name, angle):
        axis = self.config[name]
        return min(axis['max_angle'], max(axis['min_angle'], angle))

    def _manual_target(self, targets, mode, hold):
        if not self.connected:
            return
        self._targets = {n: self._clamp(n, targets[n]) for n in AXES}
        self._mode = mode
        self._hold_until = self._clock() + hold

    def move_to_home(self):
        with self._lock:
            self._manual_target({n: self.config[n]['home'] for n in AXES},
                                'home', self.config['home_hold'])

    def manual_move(self, pan_degrees=0, tilt_degrees=0, speed=None):
        """Relative logical angles in degrees; each axis retains its own °/s cap.

        ``speed`` is accepted for API compatibility; EV3's 1–100 scale is unused.
        """
        del speed
        with self._lock:
            for value in (pan_degrees, tilt_degrees):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError('servo manual angles must be finite numbers')
            self._manual_target({
                n: self._issued[n] + delta
                for n, delta in zip(AXES, (pan_degrees, tilt_degrees))
            }, 'manual', self.config['manual_hold'])

    def _tick(self):
        with self._lock:
            if not self.connected:
                return
            now = self._clock()
            elapsed = max(0.0, min(now - self._last_tick, self.config['max_dt']))
            self._last_tick = now
            if self._mode == 'auto' and not self._fresh(now):
                self.target_lost()
                return
            if self._mode not in ('auto', 'manual', 'home') or elapsed <= 0:
                return
            try:
                for name in AXES:
                    delta = self._targets[name] - self._issued[name]
                    if self._ever_issued and abs(delta) < self.config['min_step']:
                        continue
                    limit = self.config[name]['max_speed'] * elapsed
                    angle = self._clamp(name, self._issued[name] + max(-limit, min(limit, delta)))
                    self._write(name, angle_to_pulse(self.config[name], angle))
                    self._issued[name] = angle
                self.pulse_enabled = True
                self._ever_issued = True
            except Exception as exc:
                self._fail_locked(exc)

    def _run(self, stop_event):
        while not stop_event.wait(self.config['update_interval']):
            self._tick()

    def status(self):
        with self._lock:
            state = {
                'backend': self.backend, 'connected': self.connected, 'error': self.error,
                'calibrated': self.config['calibrated'], 'pulse_enabled': self.pulse_enabled,
                'mode': self._mode,
            }
            for name in AXES:
                state[name] = {
                    'commanded_angle': self._issued[name] if self._ever_issued else None,
                    'target_angle': self._targets[name],
                    **{k: self.config[name][k] for k in ('min_angle', 'max_angle', 'home')},
                }
            return state
