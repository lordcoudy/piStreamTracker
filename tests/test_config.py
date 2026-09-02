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
