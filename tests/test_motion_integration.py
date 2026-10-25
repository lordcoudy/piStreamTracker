"""Hardware-free controller selection, camera addressing and immediate stop."""

from threading import Event
from unittest.mock import Mock, patch

from pistream.config import default_config
from pistream.lifecycle import TrackerLifecycle
from pistream.preview import camera_base_url, camera_stream_url
from pistream.motion import create_motor_controller
from pistream import web_app


def test_none_never_opens_ev3_or_gpio():
    config = default_config()
    config['motors'] = {'backend': 'none'}
    with patch('pistream.motors.MotorController') as ev3:
        controller = create_motor_controller(config)
        controller.connect()
        controller.update(100, 100)
        controller.manual_move(10, 10)
        controller.move_to_home()
        controller.disconnect()
    ev3.assert_not_called()
    assert controller.status()['backend'] == 'none'
    assert not controller.connected


def test_explicit_ev3_overrides_old_disabled_setting():
    config = default_config()
    config['motors'] = {'backend': 'ev3'}
    config['ev3']['enabled'] = False
    with patch('pistream.motors.MotorController') as ev3:
        create_motor_controller(config)
    assert ev3.call_args.args[0]['enabled'] is True
    assert config['ev3']['enabled'] is False


def test_explicit_stream_origin_is_used_for_camera_recordings():
    config = default_config()
    config['tracker']['stream_url'] = 'http://127.0.0.1:8123/stream'
    assert camera_stream_url(config) == 'http://127.0.0.1:8123/stream'
    assert camera_base_url(config) == 'http://127.0.0.1:8123'


def test_bind_any_host_is_not_used_as_camera_client_address():
    config = default_config()
    config['camera']['host'] = '0.0.0.0'
    assert camera_base_url(config) == 'http://192.168.100.1:8000'


def test_lifecycle_stop_disables_motion_before_waiting_for_loop():
    entered, release = Event(), Event()
    tracker = Mock(running=False, cleanup_pending=False, remote_stop_pending=False)
    tracker.connect.return_value = True
    tracker.request_stop.side_effect = release.set

    def loop(_tracker):
        entered.set()
        release.wait(1)

    life = TrackerLifecycle(lambda: tracker, loop)
    assert life.start()['status'] == 'ok'
    assert entered.wait(1)
    assert life.stop()['status'] == 'ok'
    tracker.request_stop.assert_called_once()


def test_failed_pwm_disable_remains_visible_after_web_stop():
    release = Event()
    tracker = Mock(running=False, cleanup_pending=False, remote_stop_pending=False,
                   recording=False, recording_error=None)
    tracker.connect.return_value = True
    tracker.request_stop.side_effect = release.set
    tracker.motors.error = 'Failed to disable pan servo pulses; PWM state unknown'
    life = TrackerLifecycle(lambda: tracker, lambda _: release.wait(1))
    config = default_config()
    config['motors']['backend'] = 'gpio_servo'
    with patch.object(web_app, '_lifecycle', life), patch.object(web_app, '_config', config):
        assert life.start()['status'] == 'ok'
        with web_app.app.test_client() as client:
            assert client.post('/api/stop').status_code == 200
            motors = client.get('/api/status').json['motors']
        assert life.tracker is None
        assert not motors['connected']
        assert 'PWM state unknown' in motors['error']
        assert motors['pulse_enabled'] is None
