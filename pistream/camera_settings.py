"""Validated Picamera2 controls and transactional, camera-host settings persistence."""

from __future__ import annotations

import copy
import math
import os
import tempfile
from pathlib import Path
from threading import RLock

import yaml


CONTROL_DEFAULTS = {
    'awb_mode': 'auto',
    'colour_gains': [1.5, 1.5],
    'exposure_mode': 'auto',
    'exposure_time_us': 20000,
    'analogue_gain': 1.0,
    'brightness': 0.0,
    'contrast': 1.0,
    'saturation': 1.0,
    'sharpness': 1.0,
    'exposure_value': 0.0,
    'noise_reduction': 'fast',
}
SETTING_KEYS = frozenset({'resolution', 'framerate', *CONTROL_DEFAULTS})
RESOLUTIONS = {(1280, 720), (1280, 960), (1920, 1080)}
AWB_MODES = {'auto': 0, 'indoor': 4, 'daylight': 5, 'cloudy': 6}
NOISE_REDUCTION_MODES = {'off': 0, 'fast': 1, 'high_quality': 2, 'minimal': 3}
IMAGE_RANGES = {
    'brightness': (-1, 1), 'contrast': (0, 32), 'saturation': (0, 32),
    'sharpness': (0, 16), 'exposure_value': (-8, 8),
}
IMAGE_CONTROLS = {
    'brightness': 'Brightness', 'contrast': 'Contrast', 'saturation': 'Saturation',
    'sharpness': 'Sharpness', 'exposure_value': 'ExposureValue',
    'noise_reduction': 'NoiseReductionMode',
}
METADATA_KEYS = {
    'ExposureTime', 'AnalogueGain', 'DigitalGain', 'ColourGains',
    'ColourTemperature', 'FrameDuration', 'Lux', 'SensorTimestamp',
    *IMAGE_CONTROLS.values(),
}
LIMIT_KEYS = ('ExposureTime', 'AnalogueGain', 'ColourGains', 'FrameDurationLimits',
              *IMAGE_CONTROLS.values())


def _number(value, name, minimum, maximum):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a number')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a number') from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be finite and between {minimum:g} and {maximum:g}')
    return value


def _integer(value, name, minimum, maximum):
    value = _number(value, name, minimum, maximum)
    if not value.is_integer():
        raise ValueError(f'{name} must be an integer')
    return int(value)


def _mode(value, name, allowed):
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f'{name} must be one of {", ".join(allowed)}')
    return value


def validate_camera_controls(camera: dict) -> dict:
    """Fill/validate new YAML controls without restricting older capture geometry."""
    for key, value in CONTROL_DEFAULTS.items():
        camera.setdefault(key, copy.deepcopy(value))
    camera['awb_mode'] = _mode(camera['awb_mode'], 'camera.awb_mode', (*AWB_MODES, 'manual'))
    gains = camera['colour_gains']
    if not isinstance(gains, (list, tuple)) or len(gains) != 2:
        raise ValueError('camera.colour_gains must contain red and blue gains')
    camera['colour_gains'] = [
        _number(value, 'camera.colour_gains', 0.1, 8) for value in gains
    ]
    camera['exposure_mode'] = _mode(
        camera['exposure_mode'], 'camera.exposure_mode', ('auto', 'manual')
    )
    camera['exposure_time_us'] = _integer(
        camera['exposure_time_us'], 'camera.exposure_time_us', 1, 1000000
    )
    camera['analogue_gain'] = _number(camera['analogue_gain'], 'camera.analogue_gain', 1, 16)
    for key, (minimum, maximum) in IMAGE_RANGES.items():
        camera[key] = _number(camera[key], f'camera.{key}', minimum, maximum)
    camera['noise_reduction'] = _mode(
        camera['noise_reduction'], 'camera.noise_reduction', NOISE_REDUCTION_MODES
    )
    if camera['exposure_mode'] == 'manual':
        _validate_exposure_budget(camera)
    return camera


def _validate_exposure_budget(settings):
    if settings['exposure_time_us'] > int(1000000 / settings['framerate']):
        raise ValueError('camera.exposure_time_us must not exceed the frame duration')


def controls_for_settings(settings: dict, available_controls: dict | None = None) -> dict:
    """Build native libcamera controls; optional controls require advertised support."""
    controls = {
        'AwbEnable': settings['awb_mode'] != 'manual',
        'AeEnable': settings['exposure_mode'] != 'manual',
        'FrameDurationLimits': (round(1000000 / settings['framerate']),) * 2,
    }
    if settings['awb_mode'] == 'manual':
        controls['ColourGains'] = tuple(settings['colour_gains'])
    else:
        controls['AwbMode'] = AWB_MODES[settings['awb_mode']]
    if settings['exposure_mode'] == 'manual':
        controls['ExposureTime'] = settings['exposure_time_us']
        controls['AnalogueGain'] = settings['analogue_gain']
    for key, name in IMAGE_CONTROLS.items():
        value = settings.get(key, CONTROL_DEFAULTS[key])
        if available_controls is not None and name not in available_controls:
            if value != CONTROL_DEFAULTS[key]:
                raise ValueError(f'camera.{key} ({name}) is not supported by this camera')
            continue
        if key == 'exposure_value' and settings['exposure_mode'] == 'manual':
            continue  # Retain the saved draft until automatic exposure is enabled.
        controls[name] = NOISE_REDUCTION_MODES[value] if key == 'noise_reduction' else value
    if available_controls and {'AeFlickerMode', 'AeFlickerPeriod'} <= available_controls.keys():
        controls.update(AeFlickerMode=1, AeFlickerPeriod=10000)
    return controls


def _json_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    number = float(value)
    return number if math.isfinite(number) else None


class CameraSettings:
    """Serialize camera changes and save only after a successful runtime application.

    ``reconfigure(settings)`` rebuilds the camera's running video output for a
    changed resolution/FPS; this manager then applies the complete controls.
    A shared reentrant recorder lock also lets HTTP handlers reject recording
    changes without a race. It must cover the recording check and ``update``.
    """

    def __init__(self, config_path, config, camera, reconfigure, lock=None):
        self.path = Path(config_path).resolve()
        self.config = config
        self.camera = camera
        self.reconfigure = reconfigure
        self._lock = lock if lock is not None else RLock()
        camera_settings = validate_camera_controls(copy.deepcopy(config['camera']))
        self._settings = {key: camera_settings[key] for key in SETTING_KEYS}

    def snapshot(self):
        with self._lock:
            return self._snapshot()

    def _snapshot(self):
        # Camera restart/disconnect must not turn a successful save into an error.
        try:
            metadata = self.camera.capture_metadata(wait=1.0)
            metadata = {key: _json_value(value) for key, value in metadata.items()
                        if key in METADATA_KEYS}
        except (RuntimeError, OSError, TypeError, ValueError, AttributeError):
            metadata = {}
        limits = {}
        available = getattr(self.camera, 'camera_controls', {})
        for name in LIMIT_KEYS:
            if name in available:
                limits[name] = dict(zip(('min', 'max', 'default'),
                                        (_json_value(value) for value in available[name])))
        return {'settings': copy.deepcopy(self._settings), 'metadata': metadata, 'limits': limits}

    def _candidate(self, patch):
        if not isinstance(patch, dict):
            raise ValueError('Camera settings must be a JSON object')
        unknown = patch.keys() - SETTING_KEYS
        if unknown:
            raise ValueError(f'Unknown camera settings: {", ".join(sorted(unknown))}')
        candidate = copy.deepcopy(self._settings)
        for name, value in patch.items():
            if name == 'resolution':
                if not isinstance(value, dict) or value.keys() - {'width', 'height'}:
                    raise ValueError('camera.resolution must contain only width and height')
                candidate['resolution'].update(value)
            else:
                candidate[name] = copy.deepcopy(value)
        resolution = candidate['resolution']
        for name in ('width', 'height'):
            resolution[name] = _integer(resolution[name], f'camera.resolution.{name}', 1, 16384)
        if (resolution['width'], resolution['height']) not in RESOLUTIONS:
            raise ValueError('camera.resolution must be 1280x720, 1280x960 or 1920x1080')
        candidate['framerate'] = _number(candidate['framerate'], 'camera.framerate', 25, 40)
        validate_camera_controls(candidate)
        _validate_exposure_budget(candidate)
        return candidate

    def _validate_hardware(self, settings, requested):
        available = getattr(self.camera, 'camera_controls', {})
        for key in requested:
            if key in IMAGE_CONTROLS and IMAGE_CONTROLS[key] not in available:
                raise ValueError(f'camera.{key} ({IMAGE_CONTROLS[key]}) is not supported by this camera')
        controls = controls_for_settings(settings, available)
        # EV is stored in manual exposure mode, but must still fit this camera
        # so it can be applied when autoexposure is restored.
        if 'ExposureValue' in available:
            controls['ExposureValue'] = settings['exposure_value']
        for name in LIMIT_KEYS:
            if name not in controls or name not in available:
                continue
            lower, upper = available[name][:2]
            values = controls[name]
            if not isinstance(values, tuple):
                values = (values,)
            for index, value in enumerate(values):
                minimum = lower[index] if isinstance(lower, (tuple, list)) else lower
                maximum = upper[index] if isinstance(upper, (tuple, list)) else upper
                if not minimum <= value <= maximum:
                    raise ValueError(f'{name} is outside camera limits ({minimum:g}–{maximum:g})')

    def _apply_controls(self, settings):
        self.camera.set_controls(controls_for_settings(
            settings, getattr(self.camera, 'camera_controls', {})
        ))

    def update(self, patch):
        with self._lock:
            candidate = self._candidate(patch)
            self._validate_hardware(candidate, patch)
            previous = self._settings
            geometry_changed = any(candidate[key] != previous[key]
                                   for key in ('resolution', 'framerate'))
            try:
                if geometry_changed:
                    self.reconfigure(copy.deepcopy(candidate))
                    self._validate_hardware(candidate, patch)
                self._apply_controls(candidate)
                self._save(candidate)
            except Exception as error:
                try:
                    if geometry_changed:
                        self.reconfigure(copy.deepcopy(previous))
                    self._apply_controls(previous)
                except Exception as rollback_error:
                    raise RuntimeError(f'{error}; camera rollback failed: {rollback_error}') from error
                raise
            self._settings = candidate
            self.config['camera'].update(copy.deepcopy(candidate))
            return self._snapshot()

    def _save(self, settings):
        from pistream.config_store import config_file_lock

        with config_file_lock(self.path):
            self._save_locked(settings)

    def _save_locked(self, settings):
        # Re-read here so an unrelated edit since service startup is preserved.
        if self.path.exists():
            with self.path.open(encoding='utf-8') as handle:
                document = yaml.safe_load(handle) or {}
        else:
            document = {}
        if not isinstance(document, dict):
            raise ValueError(f'Configuration root must be a mapping: {self.path}')
        camera = document.setdefault('camera', {})
        if not isinstance(camera, dict):
            raise ValueError('camera must be a mapping')
        camera.update(copy.deepcopy(settings))
        from pistream.config import deep_merge, default_config, validate_config

        validate_config(deep_merge(default_config(), copy.deepcopy(document)))
        temporary = None
        try:
            with tempfile.NamedTemporaryFile('w', dir=self.path.parent, prefix=f'.{self.path.name}.',
                                             suffix='.tmp', encoding='utf-8', delete=False) as handle:
                temporary = Path(handle.name)
                yaml.safe_dump(document, handle, sort_keys=False, allow_unicode=True)
                handle.flush()
                os.fsync(handle.fileno())
            if self.path.exists():
                os.chmod(temporary, self.path.stat().st_mode & 0o777)
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
