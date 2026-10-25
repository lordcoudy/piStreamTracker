"""Interactive SG90 commissioning; stop the tracker service before using it."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import signal
import tempfile
from pathlib import Path

import yaml

from pistream.config import load_config
from pistream.servo import ServoController, validate_servo_config


def change_setting(config: dict, path: str, value) -> dict:
    """Validate a proposed change, leaving the current config untouched on error."""
    if path == 'calibrated':
        raise ValueError('Use confirm after checking both axes to enable automatic tracking')
    updated = copy.deepcopy(config)
    parent = updated
    parts = path.split('.')
    for part in parts[:-1]:
        if not isinstance(parent, dict) or part not in parent:
            raise ValueError(f'Unknown gpio_servo setting: {path}')
        parent = parent[part]
    if not isinstance(parent, dict) or parts[-1] not in parent:
        raise ValueError(f'Unknown gpio_servo setting: {path}')
    parent[parts[-1]] = value
    updated['calibrated'] = False
    return validate_servo_config(updated)


def save_calibration(path: Path, config: dict):
    """Atomically save only the gpio_servo section, preserving other settings."""
    config = validate_servo_config(config)
    path = Path(path)
    with path.open(encoding='utf-8') as handle:
        document = yaml.safe_load(handle) or {}
    if not isinstance(document, dict):
        raise ValueError(f'Configuration root must be a mapping: {path}')
    document['gpio_servo'] = config
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix=f'.{path.name}.',
                                         suffix='.tmp', encoding='utf-8', delete=False) as handle:
            temporary = Path(handle.name)
            yaml.safe_dump(document, handle, sort_keys=False, allow_unicode=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


HELP = '''Commands (angles are commanded degrees, never measured positions):
  status                         show commands, limits, calibration and errors
  home                           command the saved Home, gradually
  pan 1 / tilt -1                small relative moves, within working limits
  set pan.min_angle -10          edit any gpio_servo setting by dotted path
  set pan.invert true            reverse that axis's pulse mapping
  set pan.calibration.min_pulse_us 1100
  save                           save settings; automatic tracking stays gated
  confirm                        mark both axes checked and save calibration
  connect                        reopen controller after correcting an error
  stop / quit                    immediately disable pulses and exit
Every setting change disables pulses and requires a new manual/Home command.
Start with unmounted horns or narrow limits; keep the CSI cable loose.
Do not command the full servo travel in the assembled platform.'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.pi4.yaml')
    args = parser.parse_args(argv)
    if os.geteuid() == 0:
        parser.error('Run calibration as the ordinary tracker user, without sudo')
    path = Path(args.config).resolve()
    if not path.is_file():
        parser.error(f'Configuration file does not exist: {path}')
    try:
        settings = load_config(str(path))['gpio_servo']
    except (ValueError, KeyError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    controller = None

    def terminate(signum, frame):
        raise KeyboardInterrupt

    previous_term = signal.signal(signal.SIGTERM, terminate)
    try:
        controller = ServoController(settings)
        if not controller.connected:
            parser.exit(1, f"Controller error: {controller.status()['error']}\n")
        print(HELP)
        while True:
            try:
                parts = shlex.split(input('calibrate> '))
                if not parts:
                    continue
                command = parts[0].lower()
                if command in ('stop', 'quit', 'exit'):
                    break
                if command in ('help', '?'):
                    print(HELP)
                elif command == 'status':
                    print(json.dumps({'controller': controller.status(), 'settings': settings}, indent=2))
                elif command == 'home' and len(parts) == 1:
                    controller.move_to_home()
                elif command == 'connect' and len(parts) == 1:
                    if not controller.connect():
                        raise ValueError(f"Controller error: {controller.status()['error']}")
                elif command in ('pan', 'tilt') and len(parts) == 2:
                    value = float(parts[1])
                    controller.manual_move(**{f'{command}_degrees': value})
                elif command == 'set' and len(parts) >= 3:
                    proposed = change_setting(settings, parts[1], yaml.safe_load(' '.join(parts[2:])))
                    # Release the old pins before a GPIO or calibration change.
                    controller.disconnect()
                    settings = proposed
                    controller = ServoController(settings)
                    if not controller.connected:
                        raise ValueError(f"Controller error: {controller.status()['error']}")
                    print('Updated; PWM disabled, automatic tracking requires confirm.')
                elif command in ('save', 'confirm') and len(parts) == 1:
                    if command == 'confirm':
                        settings['calibrated'] = True
                    save_calibration(path, settings)
                    print(f'Saved {path}; calibrated={settings["calibrated"]}')
                else:
                    print('Unknown command; type help.')
            except (ValueError, OSError, yaml.YAMLError) as exc:
                controller.disconnect()
                print(f'Error: {exc}; PWM disabled. Correct settings, then use connect.')
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        if controller is not None:
            controller.disconnect()
        signal.signal(signal.SIGTERM, previous_term)


if __name__ == '__main__':
    main()
