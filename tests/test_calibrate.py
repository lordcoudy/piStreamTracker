"""Calibration editing preserves the rest of the selected YAML file."""

import yaml
import pytest

from pistream.calibrate import change_setting, save_calibration
from pistream.servo import default_servo_config


def test_edit_rejects_unknown_and_unsafe_values_without_changing_original():
    config = default_servo_config()
    with pytest.raises(ValueError, match='Unknown'):
        change_setting(config, 'pan.gpio_number', 20)
    with pytest.raises(ValueError, match='pan.home'):
        change_setting(config, 'pan.home', 100)
    assert config['pan']['home'] == 0
    assert config['pan']['gpio'] == 18


def test_any_edit_invalidates_previous_automatic_calibration():
    config = default_servo_config()
    config['calibrated'] = True
    result = change_setting(config, 'pan.max_angle', 20)
    assert result['pan']['max_angle'] == 20
    assert result['calibrated'] is False
    assert config['calibrated'] is True


def test_calibration_flag_needs_explicit_confirmation_command():
    with pytest.raises(ValueError, match='confirm'):
        change_setting(default_servo_config(), 'calibrated', True)


def test_save_preserves_other_config_and_calibration_gate(tmp_path):
    path = tmp_path / 'config.pi4.yaml'
    path.write_text('web:\n  port: 5002\ntracker:\n  output_dir: keep-me\n', encoding='utf-8')
    config = default_servo_config()
    config['pan']['max_angle'] = 25
    save_calibration(path, config)
    saved = yaml.safe_load(path.read_text())
    assert saved['web']['port'] == 5002
    assert saved['tracker']['output_dir'] == 'keep-me'
    assert saved['gpio_servo']['calibrated'] is False
    assert saved['gpio_servo']['pan']['max_angle'] == 25


def test_invalid_save_keeps_original_file(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text('web: {port: 5555}\n', encoding='utf-8')
    config = default_servo_config()
    config['pan']['gpio'] = 50
    with pytest.raises(ValueError, match='pan.gpio'):
        save_calibration(path, config)
    assert path.read_text() == 'web: {port: 5555}\n'


def test_sigterm_and_bad_input_disable_pwm_and_restore_signal_handler(tmp_path, monkeypatch):
    from pistream import calibrate
    from unittest.mock import Mock

    path = tmp_path / 'config.yaml'
    path.write_text('{}\n')
    ctrl = Mock(connected=True)
    monkeypatch.setattr(calibrate, 'load_config', lambda _: {'gpio_servo': default_servo_config()})
    monkeypatch.setattr(calibrate, 'ServoController', lambda _: ctrl)
    handlers = {}
    previous = object()

    def set_handler(signum, handler):
        old = handlers.get(signum, previous)
        handlers[signum] = handler
        return old

    monkeypatch.setattr(calibrate.signal, 'signal', set_handler)
    commands = iter(['set pan.home 100', 'signal'])

    def get_input(_):
        command = next(commands)
        if command == 'signal':
            handlers[calibrate.signal.SIGTERM](calibrate.signal.SIGTERM, None)
        return command

    monkeypatch.setattr('builtins.input', get_input)
    calibrate.main(['--config', str(path)])
    assert ctrl.disconnect.call_count >= 2
    assert handlers[calibrate.signal.SIGTERM] is previous
