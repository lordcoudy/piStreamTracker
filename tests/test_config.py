"""Config preset and CLI merge helpers."""

import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pistream.config import (
    apply_cli_overrides,
    apply_preset,
    camera_bind_host,
    load_config,
    web_bind_host,
)


def _base():
    return {
        'network': {'tracker_ip': '192.168.100.2'},
        'tracker': {
            'stream_url': None,
            'output_dir': 'recordings',
            'detection': {'interval': 10, 'scale': 0.4, 'confidence': 0.5},
            'movenet': {'threads': 4},
        },
        'ev3': {'enabled': True},
        'web': {'host': None, 'port': 5000},
        'presets': {
            'fast': {'detection_interval': 10, 'process_scale': 0.35, 'movenet_threads': 4},
            'quality': {'detection_interval': 4, 'process_scale': 0.6, 'movenet_threads': 4},
        },
    }


class ApplyPresetTests(unittest.TestCase):
    def test_fast_preset_overrides_scale(self):
        cfg = apply_preset(_base(), 'fast')
        self.assertEqual(cfg['tracker']['detection']['scale'], 0.35)
        self.assertEqual(cfg['tracker']['detection']['interval'], 10)

    def test_unknown_preset_raises(self):
        with self.assertRaises(ValueError):
            apply_preset(_base(), 'turbo')


class CliOverrideTests(unittest.TestCase):
    def test_explicit_interval_wins_over_preset(self):
        args = Namespace(
            url=None, output_dir=None, detection_interval=7,
            process_scale=None, confidence=None, movenet_threads=None,
            no_ev3=False, preset='fast',
        )
        cfg = apply_cli_overrides(_base(), args)
        self.assertEqual(cfg['tracker']['detection']['interval'], 7)
        self.assertEqual(cfg['tracker']['detection']['scale'], 0.35)

    def test_no_ev3_disables_motors(self):
        args = Namespace(
            url=None, output_dir=None, detection_interval=None,
            process_scale=None, confidence=None, movenet_threads=None,
            no_ev3=True, preset=None,
        )
        cfg = apply_cli_overrides(_base(), args)
        self.assertFalse(cfg['ev3']['enabled'])

    def test_zero_detection_interval_is_rejected(self):
        args = Namespace(
            url=None, output_dir=None, detection_interval=0,
            process_scale=None, confidence=None, movenet_threads=None,
            no_ev3=False, preset=None,
        )
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            apply_cli_overrides(_base(), args)


class MotorBackendTests(unittest.TestCase):
    def test_legacy_enabled_flag_selects_ev3_or_none(self):
        from pistream.config import motor_backend

        cfg = _base()
        self.assertEqual(motor_backend(cfg), 'ev3')
        cfg['ev3']['enabled'] = False
        self.assertEqual(motor_backend(cfg), 'none')

    def test_explicit_backend_overrides_legacy_enabled_flag(self):
        from pistream.config import motor_backend

        cfg = _base()
        cfg['ev3']['enabled'] = False
        cfg['motors'] = {'backend': 'gpio_servo'}
        self.assertEqual(motor_backend(cfg), 'gpio_servo')
        cfg['motors']['backend'] = 'ev3'
        self.assertEqual(motor_backend(cfg), 'ev3')

    def test_disable_flag_precedence_is_unambiguous(self):
        from pistream.config import motor_backend

        for backend, no_ev3, no_motors, expected in (
            ('gpio_servo', True, False, 'gpio_servo'),
            ('ev3', True, False, 'none'),
            ('gpio_servo', False, True, 'none'),
            ('gpio_servo', True, True, 'none'),
        ):
            with self.subTest(backend=backend, no_ev3=no_ev3, no_motors=no_motors):
                args = Namespace(motor_backend=backend, no_ev3=no_ev3, no_motors=no_motors)
                cfg = apply_cli_overrides(_base(), args)
                self.assertEqual(motor_backend(cfg), expected)

    def test_unknown_backend_is_rejected_at_config_load(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.yaml'
            path.write_text('motors:\n  backend: software_pwm\n')
            with self.assertRaisesRegex(ValueError, 'motors.backend'):
                load_config(str(path))

    def test_pi4_config_uses_local_camera_and_distinct_artifact_directories(self):
        from pistream.config import motor_backend
        from pistream.preview import camera_stream_url

        cfg = load_config(str(Path(__file__).resolve().parents[1] / 'config.pi4.yaml'))
        self.assertEqual(motor_backend(cfg), 'gpio_servo')
        self.assertEqual(camera_bind_host(cfg), '127.0.0.1')
        self.assertEqual(web_bind_host(cfg), '0.0.0.0')
        self.assertEqual(camera_stream_url(cfg), 'http://127.0.0.1:8000/stream')
        self.assertEqual(cfg['tracker']['stream_url'], camera_stream_url(cfg))
        self.assertEqual(cfg['camera']['resolution'], {'width': 1280, 'height': 720})
        self.assertEqual(cfg['camera']['framerate'], 25)
        self.assertEqual(cfg['camera']['jpeg_quality'], 80)
        self.assertEqual(cfg['camera']['tuning_file'], 'tuning/imx219-160-colour.json')
        self.assertEqual(cfg['tracker']['recording_fps'], 25)
        self.assertEqual(cfg['tracker']['movenet']['threads'], 1)
        self.assertLessEqual(cfg['web']['preview_max_fps'], 15)
        self.assertNotEqual(cfg['camera']['recording_dir'], cfg['tracker']['output_dir'])
        self.assertTrue(cfg['gpio_servo']['calibrated'])
        self.assertEqual(cfg['web']['autostart_delay'], 120)

    def test_pi4_config_preserves_limits_and_uses_live_tested_motion_settings(self):
        cfg = load_config(str(Path(__file__).resolve().parents[1] / 'config.pi4.yaml'))
        self.assertEqual(cfg['gpio_servo']['gain'], 6)
        for axis, minimum, maximum, speed in (
            ('pan', -90, 90, 25), ('tilt', -45, 45, 15),
        ):
            with self.subTest(axis=axis):
                settings = cfg['gpio_servo'][axis]
                self.assertEqual(settings['min_angle'], minimum)
                self.assertEqual(settings['max_angle'], maximum)
                self.assertEqual(settings['max_speed'], speed)
                self.assertTrue(settings['invert'])

    def test_builtin_servo_defaults_still_require_calibration(self):
        from pistream.config import default_config

        self.assertFalse(default_config()['gpio_servo']['calibrated'])

    def test_pi4_preset_sets_camera_and_inference_defaults(self):
        from pistream.config import default_config

        cfg = apply_preset(default_config(), 'pi4')
        self.assertEqual(cfg['camera']['resolution'], {'width': 640, 'height': 480})
        self.assertEqual(cfg['camera']['framerate'], 15)
        self.assertEqual(cfg['tracker']['movenet']['threads'], 2)

    def test_servo_validation_runs_before_hardware_use(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.yaml'
            path.write_text('motors:\n  backend: gpio_servo\ngpio_servo:\n  pan:\n    max_speed: .nan\n')
            with self.assertRaisesRegex(ValueError, 'pan.max_speed'):
                load_config(str(path))


class ConfigValidationTests(unittest.TestCase):
    def test_non_mapping_yaml_root_has_clear_error(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.yaml'
            path.write_text('- not\n- a\n- mapping\n')
            with self.assertRaisesRegex(ValueError, 'root must be a mapping'):
                load_config(str(path))

    def test_invalid_recording_mode_is_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.yaml'
            path.write_text('tracker:\n  recording_mode: nowhere\n')
            with self.assertRaisesRegex(ValueError, 'recording_mode'):
                load_config(str(path))

    def test_null_section_has_clear_error(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.yaml'
            path.write_text('camera: null\n')
            with self.assertRaisesRegex(ValueError, 'camera must be a mapping'):
                load_config(str(path))

    def test_horizon_defaults_are_present(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.yaml'
            path.write_text('tracker:\n  output_dir: recordings\n')
            config = load_config(str(path))
        h = config['tracker']['horizon']
        self.assertFalse(h['enabled'])
        self.assertEqual(h['max_angle'], 20)
        self.assertEqual(h['ema_alpha'], 0.15)
        self.assertEqual(h['min_apply'], 0.5)
        self.assertTrue(h['fill_crop'])

    def test_invalid_horizon_max_angle_is_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.yaml'
            path.write_text('tracker:\n  horizon:\n    max_angle: 180\n')
            with self.assertRaisesRegex(ValueError, 'horizon.max_angle'):
                load_config(str(path))

    def test_environment_camera_token_overrides_file(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.yaml'
            path.write_text('camera:\n  token: file-secret\n')
            with patch.dict('os.environ', {'PISTREAM_CAMERA_TOKEN': 'env-secret'}):
                config = load_config(str(path))
        self.assertEqual(config['camera']['token'], 'env-secret')


class CameraTuningConfigTests(unittest.TestCase):
    def test_default_camera_uses_automatic_tuning(self):
        from pistream.config import default_config

        self.assertIsNone(default_config()['camera']['tuning_file'])
        with TemporaryDirectory() as tmp:
            config = load_config(str(Path(tmp) / 'missing.yaml'))
        self.assertIsNone(config['camera']['tuning_file'])

    def test_custom_tuning_path_or_null_is_preserved(self):
        for value in ('tuning/imx219-160-no-alsc.json', '/opt/camera/profile.json', None):
            with self.subTest(value=value), TemporaryDirectory() as tmp:
                import yaml

                path = Path(tmp) / 'config.yaml'
                path.write_text(yaml.safe_dump({'camera': {'tuning_file': value}}))
                config = load_config(str(path))
                self.assertEqual(config['camera']['tuning_file'], value)

    def test_non_string_tuning_file_is_rejected(self):
        for value in ('true', '123', '[]', '{}'):
            with self.subTest(value=value), TemporaryDirectory() as tmp:
                path = Path(tmp) / 'config.yaml'
                path.write_text(f'camera:\n  tuning_file: {value}\n')
                with self.assertRaisesRegex(ValueError, 'camera.tuning_file'):
                    load_config(str(path))


class WebAutostartConfigTests(unittest.TestCase):
    def test_default_delay_is_two_minutes(self):
        from pistream.config import default_config

        self.assertEqual(default_config()['web']['autostart_delay'], 120)

    def test_existing_config_without_autostart_key_gets_default(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.yaml'
            path.write_text('web:\n  overlay: false\n')
            config = load_config(str(path))
        self.assertEqual(config['web']['autostart_delay'], 120)
        self.assertFalse(config['web']['overlay'])

    def test_missing_config_gets_default_delay(self):
        with TemporaryDirectory() as tmp:
            config = load_config(str(Path(tmp) / 'missing.yaml'))
        self.assertEqual(config['web']['autostart_delay'], 120)

    def test_configured_delay_accepts_seconds_and_boundary_values(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.yaml'
            for delay in (0, 2.5, 86400):
                with self.subTest(delay=delay):
                    path.write_text(f'web:\n  autostart_delay: {delay}\n')
                    config = load_config(str(path))
                    self.assertEqual(config['web']['autostart_delay'], delay)

    def test_null_disables_autostart(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.yaml'
            path.write_text('web:\n  autostart_delay: null\n')
            config = load_config(str(path))
        self.assertIsNone(config['web']['autostart_delay'])

    def test_invalid_autostart_delays_are_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.yaml'
            for delay in ('true', 'false', '-1', '86401', '.nan', '.inf', '-.inf', 'later'):
                with self.subTest(delay=delay):
                    path.write_text(f'web:\n  autostart_delay: {delay}\n')
                    with self.assertRaisesRegex(ValueError, 'web.autostart_delay'):
                        load_config(str(path))


class NetworkFieldTests(unittest.TestCase):
    def test_rejects_interface_with_spaces(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.yaml'
            path.write_text(
                'network:\n  camera_ip: "192.168.100.1"\n'
                '  tracker_ip: "192.168.100.2"\n  interface: "eth 0"\n'
            )
            with self.assertRaisesRegex(ValueError, 'interface'):
                load_config(str(path))

    def test_subnet_string_coerced(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.yaml'
            path.write_text(
                'network:\n  camera_ip: "192.168.100.1"\n'
                '  tracker_ip: "192.168.100.2"\n  subnet: "24"\n'
            )
            cfg = load_config(str(path))
            self.assertEqual(cfg['network']['subnet'], 24)


class WebBindHostTests(unittest.TestCase):
    def test_falls_back_to_tracker_ip(self):
        self.assertEqual(web_bind_host(_base()), '192.168.100.2')

    def test_explicit_host_wins(self):
        cfg = _base()
        cfg['web']['host'] = '0.0.0.0'
        self.assertEqual(web_bind_host(cfg), '0.0.0.0')


class CameraBindHostTests(unittest.TestCase):
    def test_falls_back_to_camera_ip(self):
        cfg = _base()
        cfg['network']['camera_ip'] = '192.168.100.1'
        cfg['camera'] = {'host': None}
        self.assertEqual(camera_bind_host(cfg), '192.168.100.1')


if __name__ == '__main__':
    unittest.main()
