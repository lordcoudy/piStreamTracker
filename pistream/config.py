"""Config load, preset, and CLI merge (no OpenCV / EV3)."""

from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path
from typing import Any


def project_root() -> Path:
    """Repo root (parent of the pistream package, or cwd if that has config.yaml)."""
    here = Path(__file__).resolve().parent
    for candidate in (here.parent, Path.cwd(), here):
        if (candidate / 'config.yaml').exists() or (candidate / 'tracker.py').exists():
            return candidate
    return here.parent


def default_config() -> dict:
    return {
        'network': {
            'camera_ip': '192.168.100.1',
            'tracker_ip': '192.168.100.2',
            'interface': 'eth0',
            'subnet': '24',
        },
        'camera': {
            'host': None,
            'port': 8000,
            'framerate': 30,
            'jpeg_quality': 80,
            'token': None,
            'max_stream_clients': 4,
            'recording_dir': 'recordings',
            'resolution': {'width': 1280, 'height': 960},
        },
        'tracker': {
            'stream_url': None,
            'output_dir': 'recordings',
            'recording_fps': 30,
            'recording_encoder': 'auto',
            'recording_mode': 'local',
            'horizon': {
                'enabled': False,
                'max_angle': 20,
                'ema_alpha': 0.15,
                'min_apply': 0.5,
                'fill_crop': True,
            },
            'detection': {'interval': 10, 'scale': 0.4, 'confidence': 0.5, 'keypoint_threshold': 0.3},
            'movenet': {'model_path': None, 'threads': None}
        },
        'ev3': {
            'enabled': True, 'deadzone': {'x': 90, 'y': 90},
            'speed_factor': 1.0, 'max_speed': 50, 'cooldown': 0.5,
            'home_hold': 3.0,
            'invert': {'x': False, 'y': False}, 'ports': {'pan': 'a', 'tilt': 'b'}
        },
        'web': {
            'enabled': True, 'host': None, 'port': 5000,
            'preview_quality': 70, 'preview_max_edge': 640,
            'preview_max_fps': 15, 'overlay': True,
        },
        'logging': {'level': 'INFO', 'file': None, 'verbose_shifts': False},
        'presets': {}
    }


def deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file with defaults."""
    config = default_config()
    config_file = Path(config_path)
    if config_file.exists():
        import yaml
        with open(config_file) as handle:
            user_config = yaml.safe_load(handle) or {}
        if not isinstance(user_config, dict):
            raise ValueError(f"Configuration root must be a mapping: {config_file}")
        deep_merge(config, user_config)
    env_token = os.environ.get('PISTREAM_CAMERA_TOKEN')
    if env_token is not None and isinstance(config.get('camera'), dict):
        config['camera']['token'] = env_token or None
    return validate_config(config)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _bounded_number(value: Any, name: str, minimum: float, maximum: float) -> float:
    number = _finite_number(value, name)
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return number


def _positive_int(value: Any, name: str, maximum: int | None = None) -> int:
    number = _finite_number(value, name)
    if not number.is_integer() or number < 1 or (maximum is not None and number > maximum):
        suffix = f" and at most {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be a positive integer{suffix}")
    return int(number)


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    number = _finite_number(value, name)
    if not number.is_integer() or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return int(number)


def _mapping(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _optional_string(value: Any, name: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string or null")
    return value


def validate_config(config: dict) -> dict:
    """Validate safety-critical settings and normalize their numeric types."""
    camera = _mapping(config.get('camera'), 'camera')
    tracker = _mapping(config.get('tracker'), 'tracker')
    detection = _mapping(tracker.get('detection'), 'tracker.detection')
    movenet = _mapping(tracker.get('movenet'), 'tracker.movenet')
    ev3 = _mapping(config.get('ev3'), 'ev3')
    web = _mapping(config.get('web'), 'web')
    network = _mapping(config.get('network'), 'network')
    log_config = _mapping(config.get('logging'), 'logging')

    camera['port'] = _positive_int(camera.get('port'), 'camera.port', 65535)
    camera['framerate'] = _bounded_number(camera.get('framerate'), 'camera.framerate', 1, 240)
    camera['max_stream_clients'] = _positive_int(
        camera.get('max_stream_clients'), 'camera.max_stream_clients', 100
    )
    camera['jpeg_quality'] = _positive_int(
        camera.get('jpeg_quality', 80), 'camera.jpeg_quality', 100
    )
    resolution = _mapping(
        camera.setdefault('resolution', {'width': 1280, 'height': 960}), 'camera.resolution'
    )
    resolution['width'] = _positive_int(resolution.get('width'), 'camera.resolution.width', 16384)
    resolution['height'] = _positive_int(resolution.get('height'), 'camera.resolution.height', 16384)

    tracker['recording_fps'] = _bounded_number(
        tracker.get('recording_fps', camera['framerate']), 'tracker.recording_fps', 1, 240
    )
    mode = tracker.setdefault('recording_mode', 'local')
    if mode not in {'local', 'camera'}:
        raise ValueError("tracker.recording_mode must be 'local' or 'camera'")
    encoder = tracker.setdefault('recording_encoder', 'auto')
    if encoder not in {'auto', 'h264_v4l2m2m', 'libx264', 'mjpg'}:
        raise ValueError(
            "tracker.recording_encoder must be auto, h264_v4l2m2m, libx264, or mjpg"
        )

    horizon = _mapping(tracker.setdefault('horizon', {}), 'tracker.horizon')
    horizon['enabled'] = _boolean(horizon.get('enabled', False), 'tracker.horizon.enabled')
    horizon['max_angle'] = _bounded_number(
        horizon.get('max_angle', 20), 'tracker.horizon.max_angle', 0, 90
    )
    horizon['ema_alpha'] = _bounded_number(
        horizon.get('ema_alpha', 0.15), 'tracker.horizon.ema_alpha', 0, 1
    )
    horizon['min_apply'] = _bounded_number(
        horizon.get('min_apply', 0.5), 'tracker.horizon.min_apply', 0, 90
    )
    horizon['fill_crop'] = _boolean(
        horizon.get('fill_crop', True), 'tracker.horizon.fill_crop'
    )

    detection['interval'] = _positive_int(detection.get('interval'), 'tracker.detection.interval')
    detection['scale'] = _bounded_number(detection.get('scale'), 'tracker.detection.scale', 0.05, 1)
    detection['confidence'] = _bounded_number(
        detection.get('confidence'), 'tracker.detection.confidence', 0, 1
    )
    detection['keypoint_threshold'] = _bounded_number(
        detection.get('keypoint_threshold'), 'tracker.detection.keypoint_threshold', 0, 1
    )
    if movenet.get('threads') is not None:
        movenet['threads'] = _positive_int(movenet['threads'], 'tracker.movenet.threads', 256)
    movenet['model_path'] = _optional_string(movenet.get('model_path'), 'tracker.movenet.model_path')

    tracker['output_dir'] = _optional_string(tracker.get('output_dir'), 'tracker.output_dir')
    if not tracker['output_dir']:
        raise ValueError('tracker.output_dir must not be empty')
    camera['recording_dir'] = _optional_string(
        camera.setdefault('recording_dir', 'recordings'), 'camera.recording_dir'
    )
    if not camera['recording_dir']:
        raise ValueError('camera.recording_dir must not be empty')
    camera['token'] = _optional_string(camera.get('token'), 'camera.token')
    camera['host'] = _optional_string(camera.get('host'), 'camera.host')
    tracker['stream_url'] = _optional_string(tracker.get('stream_url'), 'tracker.stream_url')

    ev3['speed_factor'] = _bounded_number(ev3.get('speed_factor'), 'ev3.speed_factor', 0.1, 2)
    ev3['max_speed'] = _positive_int(ev3.get('max_speed'), 'ev3.max_speed', 100)
    ev3['cooldown'] = _bounded_number(ev3.get('cooldown'), 'ev3.cooldown', 0, 60)
    ev3['home_hold'] = _bounded_number(ev3.get('home_hold'), 'ev3.home_hold', 0, 60)
    ev3['enabled'] = _boolean(ev3.get('enabled'), 'ev3.enabled')
    deadzone = _mapping(ev3.get('deadzone'), 'ev3.deadzone')
    for axis in ('x', 'y'):
        deadzone[axis] = _bounded_int(deadzone.get(axis), f'ev3.deadzone.{axis}', 0, 10000)
    invert = _mapping(ev3.get('invert'), 'ev3.invert')
    for axis in ('x', 'y'):
        invert[axis] = _boolean(invert.get(axis), f'ev3.invert.{axis}')
    ports = _mapping(ev3.get('ports'), 'ev3.ports')
    for axis in ('pan', 'tilt'):
        port = ports.get(axis)
        if not isinstance(port, str) or not port.strip():
            raise ValueError(f'ev3.ports.{axis} must be a non-empty string')

    web['port'] = _positive_int(web.get('port'), 'web.port', 65535)
    web['preview_quality'] = _positive_int(web.get('preview_quality'), 'web.preview_quality', 100)
    web['preview_max_edge'] = _bounded_int(
        web.get('preview_max_edge'), 'web.preview_max_edge', 0, 16384
    )
    web['preview_max_fps'] = _bounded_number(
        web.get('preview_max_fps'), 'web.preview_max_fps', 1, 240
    )
    web['overlay'] = _boolean(web.get('overlay'), 'web.overlay')
    web['enabled'] = _boolean(web.get('enabled'), 'web.enabled')
    web['host'] = _optional_string(web.get('host'), 'web.host')
    for key in ('camera_ip', 'tracker_ip'):
        value = network.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'network.{key} must be a non-empty string')
    iface = network.get('interface', 'eth0')
    if (
        not isinstance(iface, str)
        or not iface.strip()
        or not re.fullmatch(r'[A-Za-z0-9._-]+', iface.strip())
    ):
        raise ValueError('network.interface must be a non-empty interface name')
    network['interface'] = iface.strip()
    raw_subnet = network.get('subnet', 24)
    try:
        prefix = int(str(raw_subnet).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError('network.subnet must be an integer prefix length') from exc
    if not 1 <= prefix <= 32:
        raise ValueError('network.subnet must be between 1 and 32')
    network['subnet'] = prefix
    level = log_config.get('level')
    if not isinstance(level, str) or level.upper() not in {
        'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    }:
        raise ValueError('logging.level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL')
    log_config['level'] = level.upper()
    log_config['file'] = _optional_string(log_config.get('file'), 'logging.file')
    log_config['verbose_shifts'] = _boolean(
        log_config.get('verbose_shifts'), 'logging.verbose_shifts'
    )
    _mapping(config.get('presets'), 'presets')
    return config


def apply_preset(config: dict, name: str) -> dict:
    """Overlay a named performance preset onto tracker detection settings."""
    preset = (config.get('presets') or {}).get(name)
    if not preset:
        raise ValueError(f"Unknown preset: {name}")
    det = config['tracker']['detection']
    if 'detection_interval' in preset:
        det['interval'] = _positive_int(
            preset['detection_interval'], f'presets.{name}.detection_interval'
        )
    if 'process_scale' in preset:
        det['scale'] = _bounded_number(
            preset['process_scale'], f'presets.{name}.process_scale', 0.05, 1
        )
    if 'movenet_threads' in preset:
        config['tracker']['movenet']['threads'] = _positive_int(
            preset['movenet_threads'], f'presets.{name}.movenet_threads', 256
        )
    return config


def apply_cli_overrides(config: dict, args: Any) -> dict:
    """Merge argparse namespace fields into config (None means unset).

    Named preset is applied first so explicit flags win.
    """
    if getattr(args, 'preset', None):
        apply_preset(config, args.preset)
    if getattr(args, 'url', None):
        config['tracker']['stream_url'] = args.url
    if getattr(args, 'output_dir', None):
        config['tracker']['output_dir'] = args.output_dir
    if getattr(args, 'detection_interval', None) is not None:
        config['tracker']['detection']['interval'] = _positive_int(
            args.detection_interval, '--detection-interval'
        )
    if getattr(args, 'process_scale', None) is not None:
        config['tracker']['detection']['scale'] = _bounded_number(
            args.process_scale, '--process-scale', 0.05, 1
        )
    if getattr(args, 'confidence', None) is not None:
        config['tracker']['detection']['confidence'] = _bounded_number(
            args.confidence, '--confidence', 0, 1
        )
    if getattr(args, 'movenet_threads', None) is not None:
        config['tracker']['movenet']['threads'] = _positive_int(
            args.movenet_threads, '--movenet-threads', 256
        )
    if getattr(args, 'no_ev3', False):
        config['ev3']['enabled'] = False
    return config


def configure_logging(config: dict) -> None:
    """Apply logging.level / logging.file from config."""
    log_cfg = config.get('logging') or {}
    level_name = str(log_cfg.get('level') or 'INFO').upper()
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    for handler in root.handlers:
        handler.setLevel(level)
    log_file = log_cfg.get('file')
    if not log_file:
        return
    abs_path = str(Path(log_file).resolve())
    already = any(
        isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == abs_path
        for h in root.handlers
    )
    if already:
        return
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root.addHandler(fh)


def web_bind_host(config: dict) -> str:
    """Host for the Flask app: explicit web.host, else tracker_ip."""
    host = (config.get('web') or {}).get('host')
    if host:
        return host
    return (config.get('network') or {}).get('tracker_ip') or '0.0.0.0'


def camera_bind_host(config: dict) -> str:
    """Host for the camera HTTP server: explicit camera.host, else camera_ip."""
    host = (config.get('camera') or {}).get('host')
    if host:
        return host
    return (config.get('network') or {}).get('camera_ip') or '0.0.0.0'
