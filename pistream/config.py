"""Config load, preset, and CLI merge (no OpenCV / EV3)."""

from __future__ import annotations

import logging
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
        'network': {'camera_ip': '192.168.100.1', 'tracker_ip': '192.168.100.2'},
        'camera': {'port': 8000, 'framerate': 30},
        'tracker': {
            'output_dir': 'recordings',
            'recording_fps': 30,
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
        deep_merge(config, user_config)
    return config


def apply_preset(config: dict, name: str) -> dict:
    """Overlay a named performance preset onto tracker detection settings."""
    preset = (config.get('presets') or {}).get(name)
    if not preset:
        raise ValueError(f"Unknown preset: {name}")
    det = config['tracker']['detection']
    if 'detection_interval' in preset:
        det['interval'] = preset['detection_interval']
    if 'process_scale' in preset:
        det['scale'] = preset['process_scale']
    if 'movenet_threads' in preset:
        config['tracker']['movenet']['threads'] = preset['movenet_threads']
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
    if getattr(args, 'detection_interval', None):
        config['tracker']['detection']['interval'] = args.detection_interval
    if getattr(args, 'process_scale', None):
        config['tracker']['detection']['scale'] = args.process_scale
    if getattr(args, 'confidence', None):
        config['tracker']['detection']['confidence'] = args.confidence
    if getattr(args, 'movenet_threads', None):
        config['tracker']['movenet']['threads'] = args.movenet_threads
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
