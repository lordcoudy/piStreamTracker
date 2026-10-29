"""Camera controls, persistent settings and serialized rollback without CSI hardware."""

import copy
import threading
from unittest.mock import patch

import pytest
import yaml

from pistream.camera_settings import CameraSettings, controls_for_settings
from pistream.config import load_config


class FakeCamera:
    camera_controls = {
        'ExposureTime': (100, 100000, 20000),
        'AnalogueGain': (1.0, 10.0, 1.0),
        'ColourGains': (0.0, 32.0, None),
        'FrameDurationLimits': (20000, 100000, None),
        'Brightness': (-1.0, 1.0, 0.0),
        'Contrast': (0.0, 32.0, 1.0),
        'Saturation': (0.0, 32.0, 1.0),
        'Sharpness': (0.0, 16.0, 1.0),
        'ExposureValue': (-8.0, 8.0, 0.0),
        'NoiseReductionMode': (0, 4, 0),
    }

    def __init__(self):
        self.controls = []
        self.fail_once = False

    def set_controls(self, controls):
        self.controls.append(copy.deepcopy(controls))
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError('camera rejected controls')

    def capture_metadata(self, wait=None):
        return {
            'ExposureTime': 19000, 'AnalogueGain': 1.25, 'ColourGains': (1.6, 1.4),
            'ColourTemperature': 4400, 'FrameDuration': 40000, 'Unrelated': object(),
        }


@pytest.fixture
def setup_manager(tmp_path):
    path = tmp_path / 'config.yaml'
    document = {
        'camera': {
            'framerate': 25, 'resolution': {'width': 1280, 'height': 720},
            'token': 'preserved-token', 'jpeg_quality': 80,
        },
        'motors': {'backend': 'none'},
        'presets': {'custom': {'detection_interval': 7}},
    }
    path.write_text(yaml.safe_dump(document), encoding='utf-8')
    config = load_config(str(path))
    camera = FakeCamera()
    reconfigurations = []
    manager = CameraSettings(path, config, camera, reconfigurations.append)
    return manager, camera, path, config, reconfigurations


def test_old_config_gets_automatic_controls_without_changing_geometry(tmp_path):
    path = tmp_path / 'old.yaml'
    path.write_text('camera:\n  framerate: 120\n  resolution: {width: 640, height: 480}\n')
    camera = load_config(str(path))['camera']
    assert camera['framerate'] == 120
    assert camera['resolution'] == {'width': 640, 'height': 480}
    assert camera['awb_mode'] == 'auto'
    assert camera['colour_gains'] == [1.5, 1.5]
    assert camera['exposure_mode'] == 'auto'
    assert camera['exposure_time_us'] == 20000
    assert camera['analogue_gain'] == 1.0
    assert {key: camera[key] for key in IMAGE_SETTINGS} == IMAGE_SETTINGS


IMAGE_SETTINGS = {
    'brightness': 0.0, 'contrast': 1.0, 'saturation': 1.0,
    'sharpness': 1.0, 'exposure_value': 0.0, 'noise_reduction': 'fast',
}


def test_image_settings_apply_live_and_persist_without_changing_existing_controls(setup_manager):
    manager, camera, path, config, changes = setup_manager
    original = manager.snapshot()['settings']
    desired = {'brightness': -0.2, 'contrast': 1.4, 'saturation': 0.8,
               'sharpness': 2.5, 'exposure_value': 0.5, 'noise_reduction': 'high_quality'}
    result = manager.update(desired)
    assert all(result['settings'][key] == value for key, value in desired.items())
    assert all(result['settings'][key] == value for key, value in original.items()
               if key not in desired)
    assert camera.controls[-1] == {
        **controls_for_settings(original), 'Brightness': -0.2, 'Contrast': 1.4,
        'Saturation': 0.8, 'Sharpness': 2.5, 'ExposureValue': 0.5, 'NoiseReductionMode': 2,
    }
    assert changes == []
    saved = yaml.safe_load(path.read_text())
    assert saved['camera']['token'] == 'preserved-token'
    reloaded = CameraSettings(path, load_config(str(path)), camera, changes.append)
    assert reloaded.snapshot()['settings'] == result['settings']
    assert config['camera']['sharpness'] == 2.5


@pytest.mark.parametrize('name,value', [
    ('brightness', -1.1), ('brightness', 1.1), ('brightness', True),
    ('contrast', -0.1), ('contrast', 33), ('contrast', float('nan')),
    ('contrast', 10 ** 400),
    ('saturation', -1), ('saturation', 33), ('saturation', float('inf')),
    ('sharpness', -1), ('sharpness', 17), ('sharpness', False),
    ('exposure_value', -9), ('exposure_value', 9), ('exposure_value', float('-inf')),
    ('noise_reduction', 'zsl'), ('noise_reduction', 1), ('noise_reduction', None),
])
def test_invalid_image_setting_is_rejected_on_update_and_startup(setup_manager, name, value):
    manager, camera, path, config, changes = setup_manager
    original = path.read_bytes()
    with pytest.raises(ValueError, match=f'camera.{name}'):
        manager.update({name: value})
    assert path.read_bytes() == original
    assert camera.controls == []
    document = yaml.safe_load(path.read_text())
    document['camera'][name] = value
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match=f'camera.{name}'):
        load_config(str(path))


@pytest.mark.parametrize('mode,number', [
    ('off', 0), ('fast', 1), ('high_quality', 2), ('minimal', 3),
])
def test_noise_reduction_uses_installed_libcamera_enum(setup_manager, mode, number):
    manager, camera, path, config, changes = setup_manager
    manager.update({'noise_reduction': mode})
    assert camera.controls[-1]['NoiseReductionMode'] == number


def test_noise_reduction_and_image_controls_report_hardware_limits(setup_manager):
    manager, camera, path, config, changes = setup_manager
    limits = manager.snapshot()['limits']
    assert limits['Brightness'] == {'min': -1.0, 'max': 1.0, 'default': 0.0}
    assert limits['Sharpness'] == {'min': 0.0, 'max': 16.0, 'default': 1.0}
    assert limits['NoiseReductionMode'] == {'min': 0, 'max': 4, 'default': 0}


@pytest.mark.parametrize('name,native', [
    ('brightness', 'Brightness'), ('contrast', 'Contrast'), ('saturation', 'Saturation'),
    ('sharpness', 'Sharpness'), ('exposure_value', 'ExposureValue'),
    ('noise_reduction', 'NoiseReductionMode'),
])
def test_explicit_unsupported_image_control_is_rejected_even_at_default(setup_manager, name, native):
    manager, camera, path, config, changes = setup_manager
    camera.camera_controls = {key: value for key, value in camera.camera_controls.items()
                              if key != native}
    original = path.read_bytes()
    with pytest.raises(ValueError, match=f'camera.{name}.*not supported'):
        manager.update({name: IMAGE_SETTINGS[name]})
    assert path.read_bytes() == original
    assert camera.controls == []
    # Other controls still work; their default application skips the absent one.
    manager.update({'awb_mode': 'indoor'})
    assert native not in camera.controls[-1]
    assert native not in manager.snapshot()['limits']


@pytest.mark.parametrize('name,value,native,bounds', [
    ('contrast', 2.5, 'Contrast', (0.0, 2.0, 1.0)),
    ('sharpness', 6, 'Sharpness', (0.0, 4.0, 1.0)),
    ('exposure_value', 6, 'ExposureValue', (-4.0, 4.0, 0.0)),
    ('noise_reduction', 'minimal', 'NoiseReductionMode', (0, 2, 0)),
])
def test_image_settings_obey_narrower_hardware_limits(setup_manager, name, value, native, bounds):
    manager, camera, path, config, changes = setup_manager
    camera.camera_controls = {**camera.camera_controls, native: bounds}
    with pytest.raises(ValueError, match='camera limits'):
        manager.update({name: value})
    assert camera.controls == []


def test_exposure_compensation_is_saved_in_manual_mode_and_applied_with_auto_ae(setup_manager):
    manager, camera, path, config, changes = setup_manager
    result = manager.update({'exposure_mode': 'manual', 'exposure_time_us': 10000,
                             'exposure_value': 1.25})
    assert result['settings']['exposure_value'] == 1.25
    assert 'ExposureValue' not in camera.controls[-1]
    assert yaml.safe_load(path.read_text())['camera']['exposure_value'] == 1.25
    manager.update({'exposure_mode': 'auto'})
    assert camera.controls[-1]['ExposureValue'] == 1.25


def test_image_settings_roll_back_on_persistence_failure(setup_manager):
    manager, camera, path, config, changes = setup_manager
    original = path.read_bytes()
    previous = manager.snapshot()['settings']
    with patch('pistream.camera_settings.os.replace', side_effect=OSError('disk failed')):
        with pytest.raises(OSError, match='disk failed'):
            manager.update({'brightness': 0.4, 'noise_reduction': 'off'})
    assert path.read_bytes() == original
    assert camera.controls[-1] == controls_for_settings(previous)
    assert manager.snapshot()['settings'] == previous


def test_legacy_helper_input_retains_native_image_defaults(setup_manager):
    manager, camera, path, config, changes = setup_manager
    settings = {key: value for key, value in manager.snapshot()['settings'].items()
                if key not in IMAGE_SETTINGS}
    controls = controls_for_settings(settings, camera.camera_controls)
    assert controls['Brightness'] == 0
    assert controls['Contrast'] == controls['Saturation'] == controls['Sharpness'] == 1
    assert controls['NoiseReductionMode'] == 1
    assert controls['ExposureValue'] == 0


@pytest.mark.parametrize('name,value', [
    ('awb_mode', 'kelvin'), ('awb_mode', True), ('colour_gains', [1.0]),
    ('colour_gains', [float('nan'), 1]), ('colour_gains', [True, 1]),
    ('exposure_mode', 'night'), ('exposure_time_us', True),
    ('exposure_time_us', 10.2), ('analogue_gain', float('inf')),
])
def test_startup_rejects_invalid_camera_control_fields(tmp_path, name, value):
    path = tmp_path / 'invalid.yaml'
    path.write_text(yaml.safe_dump({'camera': {name: value}}))
    with pytest.raises(ValueError, match=f'camera.{name}'):
        load_config(str(path))


def test_snapshot_reports_actual_metadata_and_hardware_limits(setup_manager):
    manager, camera, path, config, changes = setup_manager
    result = manager.snapshot()
    assert result['metadata']['ColourGains'] == [1.6, 1.4]
    assert result['metadata']['ExposureTime'] == 19000
    assert 'Unrelated' not in result['metadata']
    assert result['limits']['AnalogueGain'] == {'min': 1.0, 'max': 10.0, 'default': 1.0}
    result['settings']['resolution']['width'] = 100
    assert manager.snapshot()['settings']['resolution']['width'] == 1280


def test_successful_save_is_reported_when_metadata_capture_temporarily_fails(setup_manager):
    manager, camera, path, config, changes = setup_manager
    with patch.object(camera, 'capture_metadata', side_effect=OSError('sensor is reconnecting')):
        result = manager.update({'awb_mode': 'indoor'})
    assert result['metadata'] == {}
    assert load_config(str(path))['camera']['awb_mode'] == 'indoor'


def test_stalled_metadata_cannot_hold_a_successful_save_open(setup_manager):
    manager, camera, path, config, changes = setup_manager
    frame_ready = threading.Event()
    results = []
    failures = []

    def stalled_capture(wait=None):
        if not frame_ready.wait(timeout=wait):
            raise TimeoutError('camera stopped producing frames')
        return {}

    def update():
        try:
            results.append(manager.update({'awb_mode': 'indoor'}))
        except Exception as exc:
            failures.append(exc)

    with patch.object(camera, 'capture_metadata', side_effect=stalled_capture):
        worker = threading.Thread(target=update)
        worker.start()
        try:
            worker.join(2)
            assert not worker.is_alive(), 'A metadata stall must not retain the settings/recorder lock'
            assert failures == []
            assert results[0]['metadata'] == {}
            assert load_config(str(path))['camera']['awb_mode'] == 'indoor'
        finally:
            frame_ready.set()
            worker.join(2)

    # A later request remains usable once the sensor is producing frames again.
    assert manager.update({'awb_mode': 'daylight'})['settings']['awb_mode'] == 'daylight'


def test_manual_controls_and_fixed_framerate_are_applied_and_saved(setup_manager):
    manager, camera, path, config, changes = setup_manager
    result = manager.update({
        'awb_mode': 'manual', 'colour_gains': [1.7, 1.3],
        'exposure_mode': 'manual', 'exposure_time_us': 10000, 'analogue_gain': 2,
    })
    assert camera.controls[-1] == {
        'AwbEnable': False, 'ColourGains': (1.7, 1.3), 'AeEnable': False,
        'ExposureTime': 10000, 'AnalogueGain': 2.0, 'FrameDurationLimits': (40000, 40000),
        'Brightness': 0.0, 'Contrast': 1.0, 'Saturation': 1.0, 'Sharpness': 1.0,
        'NoiseReductionMode': 1,
    }
    assert changes == []
    assert result['settings']['exposure_time_us'] == 10000
    document = yaml.safe_load(path.read_text())
    assert document['camera']['awb_mode'] == 'manual'
    assert document['camera']['token'] == 'preserved-token'
    assert document['camera']['jpeg_quality'] == 80
    assert document['motors'] == {'backend': 'none'}
    assert document['presets'] == {'custom': {'detection_interval': 7}}
    assert config['camera']['colour_gains'] == [1.7, 1.3]


@pytest.mark.parametrize('mode,number', [('auto', 0), ('indoor', 4), ('daylight', 5), ('cloudy', 6)])
def test_automatic_white_balance_selects_libcamera_mode(setup_manager, mode, number):
    manager, camera, path, config, changes = setup_manager
    settings = manager.snapshot()['settings']
    settings['awb_mode'] = mode
    controls = controls_for_settings(settings)
    assert controls['AwbEnable'] is True
    assert controls['AwbMode'] == number
    assert controls['AeEnable'] is True
    assert 'ColourGains' not in controls
    assert 'ExposureTime' not in controls


def test_mains_flicker_controls_require_both_advertised_controls(setup_manager):
    manager, camera, path, config, changes = setup_manager
    settings = manager.snapshot()['settings']
    supported = {'AeFlickerMode': (0, 1, 0), 'AeFlickerPeriod': (100, 20000, 10000)}
    controls = controls_for_settings(settings, supported)
    assert controls['AeFlickerMode'] == 1
    assert controls['AeFlickerPeriod'] == 10000
    assert 'AeFlickerMode' not in controls_for_settings(settings, {'AeFlickerMode': (0, 1, 0)})


def test_geometry_change_reconfigures_before_controls(setup_manager):
    manager, camera, path, config, changes = setup_manager
    result = manager.update({'resolution': {'height': 960}, 'framerate': 30})
    assert changes == [result['settings']]
    assert camera.controls[-1]['FrameDurationLimits'] == (33333, 33333)
    assert result['settings']['resolution'] == {'width': 1280, 'height': 960}
    assert load_config(str(path))['camera']['framerate'] == 30


@pytest.mark.parametrize('update', [
    {'token': 'changed'}, {'resolution': {'width': 640, 'height': 480}},
    {'resolution': {'width': 1280, 'height': 720, 'other': 2}},
    {'resolution': {'width': True}}, {'resolution': []},
    {'framerate': 24}, {'framerate': True}, {'framerate': float('nan')},
    {'framerate': 41}, {'awb_mode': 'unknown'}, {'colour_gains': [1, 9]},
    {'colour_gains': [False, 1]}, {'colour_gains': '1,1'},
    {'exposure_mode': 'night'}, {'exposure_time_us': 50000},
    {'exposure_time_us': False}, {'analogue_gain': 0.5},
    {'analogue_gain': float('inf')},
])
def test_invalid_update_does_not_touch_camera_or_file(setup_manager, update):
    manager, camera, path, config, changes = setup_manager
    original = path.read_bytes()
    with pytest.raises(ValueError):
        manager.update(update)
    assert path.read_bytes() == original
    assert camera.controls == []
    assert changes == []


@pytest.mark.parametrize('update', [
    {'exposure_mode': 'manual', 'exposure_time_us': 50},
    {'exposure_mode': 'manual', 'analogue_gain': 11},
])
def test_manual_values_outside_hardware_limits_are_rejected(setup_manager, update):
    manager, camera, path, config, changes = setup_manager
    with pytest.raises(ValueError):
        manager.update(update)
    assert camera.controls == []


def test_runtime_control_failure_restores_old_controls_and_keeps_file(setup_manager):
    manager, camera, path, config, changes = setup_manager
    original = path.read_bytes()
    old_settings = manager.snapshot()['settings']
    camera.fail_once = True
    with pytest.raises(RuntimeError, match='camera rejected'):
        manager.update({'awb_mode': 'indoor'})
    assert path.read_bytes() == original
    assert camera.controls[-1] == controls_for_settings(old_settings)
    assert manager.snapshot()['settings'] == old_settings


def test_write_failure_rolls_back_reconfiguration_and_controls(setup_manager):
    manager, camera, path, config, changes = setup_manager
    old_settings = manager.snapshot()['settings']
    original = path.read_bytes()
    with patch('pistream.camera_settings.os.replace', side_effect=OSError('disk write failed')):
        with pytest.raises(OSError, match='disk write failed'):
            manager.update({'resolution': {'height': 960}})
    assert changes[0]['resolution']['height'] == 960
    assert changes[1] == old_settings
    assert camera.controls[-1] == controls_for_settings(old_settings)
    assert manager.snapshot()['settings'] == old_settings
    assert path.read_bytes() == original
    assert list(path.parent.glob('.*.tmp')) == []


def test_file_flush_failure_keeps_original_and_removes_temporary_file(setup_manager):
    manager, camera, path, config, changes = setup_manager
    original = path.read_bytes()
    with patch('pistream.camera_settings.os.fsync', side_effect=OSError('disk full')):
        with pytest.raises(OSError, match='disk full'):
            manager.update({'awb_mode': 'indoor'})
    assert path.read_bytes() == original
    assert manager.snapshot()['settings']['awb_mode'] == 'auto'
    assert list(path.parent.glob('.*.tmp')) == []


def test_save_preserves_existing_file_permissions(setup_manager):
    manager, camera, path, config, changes = setup_manager
    path.chmod(0o640)
    manager.update({'awb_mode': 'indoor'})
    assert path.stat().st_mode & 0o777 == 0o640


def test_failed_rollback_reports_unknown_runtime_state(setup_manager):
    manager, camera, path, config, changes = setup_manager
    original = path.read_bytes()
    with patch.object(camera, 'set_controls', side_effect=RuntimeError('camera disconnected')):
        with pytest.raises(RuntimeError, match='camera rollback failed'):
            manager.update({'awb_mode': 'indoor'})
    assert path.read_bytes() == original
    assert config['camera']['awb_mode'] == 'auto'


def test_failed_reconfiguration_attempts_to_restore_previous_geometry(setup_manager):
    manager, camera, path, config, changes = setup_manager
    old_settings = manager.snapshot()['settings']
    def reconfigure(settings):
        changes.append(copy.deepcopy(settings))
        if len(changes) == 1:
            raise RuntimeError('stream restart failed')
    manager.reconfigure = reconfigure
    with pytest.raises(RuntimeError, match='stream restart failed'):
        manager.update({'framerate': 30})
    assert changes[-1] == old_settings
    assert manager.snapshot()['settings'] == old_settings
    assert load_config(str(path))['camera']['framerate'] == 25


def test_file_save_reads_latest_unrelated_configuration(setup_manager):
    manager, camera, path, config, changes = setup_manager
    document = yaml.safe_load(path.read_text())
    document['web'] = {'autostart_delay': None}
    document['camera']['token'] = 'changed-by-other-tool'
    path.write_text(yaml.safe_dump(document))
    manager.update({'awb_mode': 'daylight'})
    saved = yaml.safe_load(path.read_text())
    assert saved['web'] == {'autostart_delay': None}
    assert saved['camera']['token'] == 'changed-by-other-tool'


def test_invalid_latest_yaml_rolls_back_without_overwriting_external_edit(setup_manager):
    manager, camera, path, config, changes = setup_manager
    path.write_text('- invalid root\n')
    with pytest.raises(ValueError, match='mapping'):
        manager.update({'awb_mode': 'indoor'})
    assert path.read_text() == '- invalid root\n'
    assert manager.snapshot()['settings']['awb_mode'] == 'auto'


def test_concurrent_partial_updates_merge_after_shared_lock(setup_manager):
    manager, camera, path, config, changes = setup_manager
    entered = threading.Event()
    release = threading.Event()
    original_set_controls = camera.set_controls
    first = True
    def delayed_controls(controls):
        nonlocal first
        if first:
            first = False
            entered.set()
            assert release.wait(2)
        original_set_controls(controls)
    camera.set_controls = delayed_controls
    failures = []
    def update(settings):
        try:
            manager.update(settings)
        except Exception as exc:
            failures.append(exc)
    one = threading.Thread(target=update, args=({'awb_mode': 'indoor'},))
    two = threading.Thread(target=update, args=({'analogue_gain': 2},))
    one.start()
    assert entered.wait(2)
    two.start()
    release.set()
    one.join(2)
    two.join(2)
    assert not one.is_alive() and not two.is_alive()
    assert failures == []
    saved = load_config(str(path))['camera']
    assert saved['awb_mode'] == 'indoor'
    assert saved['analogue_gain'] == 2


