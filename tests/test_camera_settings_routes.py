"""Camera host settings proxy and tracker lifecycle boundaries."""

import copy
import io
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch
from threading import Event, Thread
from urllib.error import HTTPError

import pytest

from pistream import web_app


SETTINGS = {
    'resolution': {'width': 1280, 'height': 720}, 'framerate': 25,
    'awb_mode': 'auto', 'colour_gains': [1.5, 1.5],
    'exposure_mode': 'auto', 'exposure_time_us': 20000, 'analogue_gain': 1,
}


@pytest.fixture
def routes(monkeypatch):
    config = {'camera': {**copy.deepcopy(SETTINGS), 'port': 8000, 'token': 'private'},
              'network': {'camera_ip': 'camera'}, 'tracker': {'recording_fps': 25}}
    tracker = SimpleNamespace(running=True, recording=False, remote_stop_pending=False,
                              cleanup_pending=False)
    lifecycle = SimpleNamespace(tracker=tracker, stopping=False,
                                stop=Mock(return_value={'status': 'ok'}),
                                start=Mock(return_value={'status': 'ok'}))
    monkeypatch.setattr(web_app, '_config', config)
    monkeypatch.setattr(web_app, '_lifecycle', lifecycle)
    return web_app.app.test_client(), lifecycle, config


def snapshot(settings=None):
    return {'settings': copy.deepcopy(settings or SETTINGS),
            'metadata': {'FrameDuration': 40000}, 'limits': {}}


def test_get_proxies_configured_camera_and_token(routes):
    client, lifecycle, _ = routes
    with patch.object(web_app, 'request_camera_settings', return_value=snapshot()) as proxy:
        response = client.get('/api/camera/settings')
    assert response.status_code == 200
    assert response.json['settings'] == SETTINGS
    proxy.assert_called_once_with('http://camera:8000', 'private')
    lifecycle.stop.assert_not_called()


def test_controls_only_save_does_not_interrupt_tracking(routes):
    client, lifecycle, _ = routes
    changed = {**SETTINGS, 'awb_mode': 'daylight'}
    with patch.object(web_app, 'request_camera_settings', return_value=snapshot(changed)) as proxy:
        response = client.post('/api/camera/settings', json={'awb_mode': 'daylight'})
    assert response.status_code == 200
    assert response.json['saved'] is True
    assert proxy.call_args_list[-1].args == (
        'http://camera:8000', 'private', {'awb_mode': 'daylight'})
    lifecycle.stop.assert_not_called()


def test_geometry_change_stops_then_applies_and_restarts_with_new_config(routes):
    client, lifecycle, config = routes
    events = []
    lifecycle.stop.side_effect = lambda: events.append('stop') or {'status': 'ok'}
    lifecycle.start.side_effect = lambda **_kwargs: events.append(('start', config['camera']['framerate'])) or {'status': 'ok'}
    changed = {**SETTINGS, 'framerate': 30}

    def proxy(_base, _token, patch=None):
        if patch is not None:
            events.append('apply')
            return snapshot(changed)
        return snapshot()

    with patch.object(web_app, 'request_camera_settings', side_effect=proxy):
        response = client.post('/api/camera/settings', json={'framerate': 30})
    assert response.status_code == 200
    assert events == ['stop', 'apply', ('start', 30)]
    assert config['tracker']['recording_fps'] == 30


def test_stopped_tracker_stays_stopped_after_format_change(routes):
    client, lifecycle, _ = routes
    lifecycle.tracker.running = False
    with patch.object(web_app, 'request_camera_settings', side_effect=[snapshot(), snapshot({**SETTINGS, 'framerate': 30})]):
        response = client.post('/api/camera/settings', json={'framerate': 30})
    assert response.status_code == 200
    lifecycle.stop.assert_called_once()  # Also consumes pending autostart.
    lifecycle.start.assert_not_called()


@pytest.mark.parametrize('field', ['recording', 'remote_stop_pending', 'cleanup_pending'])
def test_recording_or_unfinished_cleanup_blocks_changes(routes, field):
    client, lifecycle, _ = routes
    setattr(lifecycle.tracker, field, True)
    if field == 'cleanup_pending':
        lifecycle.tracker.running = False
    with patch.object(web_app, 'request_camera_settings') as proxy:
        response = client.post('/api/camera/settings', json={'awb_mode': 'daylight'})
    assert response.status_code == 409
    proxy.assert_not_called()
    lifecycle.stop.assert_not_called()


def test_running_tracker_owns_open_capture_without_blocking_settings(routes):
    client, lifecycle, _ = routes
    lifecycle.tracker.cleanup_pending = True  # True while capture/detector are open.
    with patch.object(web_app, 'request_camera_settings', return_value=snapshot()):
        response = client.post('/api/camera/settings', json={'awb_mode': 'manual'})
    assert response.status_code == 200
    lifecycle.stop.assert_not_called()


def test_stopping_owner_is_not_reconfigured(routes):
    client, lifecycle, _ = routes
    lifecycle.stop.return_value = {'status': 'stopping'}
    with patch.object(web_app, 'request_camera_settings', return_value=snapshot()) as proxy:
        response = client.post('/api/camera/settings', json={'framerate': 30})
    assert response.status_code == 409
    assert proxy.call_count == 1
    lifecycle.start.assert_not_called()


def test_unknown_apply_outcome_keeps_tracking_stopped(routes):
    client, lifecycle, config = routes
    with patch.object(web_app, 'request_camera_settings', side_effect=[snapshot(), TimeoutError('timed out')]):
        response = client.post('/api/camera/settings', json={'framerate': 30})
    assert response.status_code == 503
    assert 'Reload' in response.json['message']
    lifecycle.start.assert_not_called()
    assert config['camera']['framerate'] == 25


def test_saved_configuration_is_not_reported_failed_if_tracker_cannot_restart(routes):
    client, lifecycle, _ = routes
    lifecycle.start.return_value = {'status': 'error', 'message': 'stream unavailable'}
    with patch.object(web_app, 'request_camera_settings', side_effect=[snapshot(), snapshot({**SETTINGS, 'framerate': 30})]):
        response = client.post('/api/camera/settings', json={'framerate': 30})
    assert response.status_code == 200
    assert response.json['saved'] is True
    assert 'stream unavailable' in response.json['warning']


@pytest.mark.parametrize('body', ['[]', 'not json', '{"not_a_setting": 1}', '{"framerate": NaN}', '{"framerate": Infinity}'])
def test_invalid_payload_is_rejected_before_hardware(routes, body):
    client, lifecycle, _ = routes
    with patch.object(web_app, 'request_camera_settings') as proxy:
        response = client.post('/api/camera/settings', data=body, content_type='application/json')
    assert response.status_code == 400
    proxy.assert_not_called()
    lifecycle.stop.assert_not_called()


def test_stop_is_not_blocked_by_settings_network_request(routes):
    client, lifecycle, _ = routes
    entered, release, stopped = Event(), Event(), Event()

    def slow_proxy(*_args):
        entered.set()
        assert release.wait(2)
        return snapshot()

    lifecycle.stop.side_effect = lambda: stopped.set() or {'status': 'ok'}
    lifecycle.tracker.recording_error = None
    with patch.object(web_app, 'request_camera_settings', side_effect=slow_proxy):
        worker = Thread(target=lambda: web_app.app.test_client().get('/api/camera/settings'))
        worker.start()
        try:
            assert entered.wait(1)
            assert client.post('/api/stop').status_code == 200
            assert stopped.is_set()
        finally:
            release.set()
            worker.join(2)
    assert not worker.is_alive()


def test_stop_during_format_apply_cancels_automatic_restart(routes):
    client, lifecycle, _ = routes
    entered, release = Event(), Event()
    lifecycle.tracker.recording_error = None
    guards = []

    def proxy(_base, _token, patch=None):
        if patch is None:
            return snapshot()
        entered.set()
        assert release.wait(2)
        return snapshot({**SETTINGS, 'framerate': 30})

    def restart(*, unless_cancelled):
        guards.append(unless_cancelled)
        assert unless_cancelled.is_set()
        return {'status': 'cancelled'}

    lifecycle.start.side_effect = restart
    with patch.object(web_app, 'request_camera_settings', side_effect=proxy):
        worker = Thread(target=lambda: web_app.app.test_client().post('/api/camera/settings', json={'framerate': 30}))
        worker.start()
        try:
            assert entered.wait(1)
            assert client.post('/api/stop').status_code == 200
        finally:
            release.set()
            worker.join(2)
    assert not worker.is_alive()
    assert len(guards) == 1


def test_camera_client_forwards_token_and_json_without_retries():
    from pistream.camera_api import request_camera_settings

    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.read.return_value = json.dumps(snapshot()).encode()
    with patch('pistream.camera_api.urllib.request.urlopen', return_value=response) as opened:
        payload = request_camera_settings('http://camera:8000', 'private', {'framerate': 30})
    request = opened.call_args.args[0]
    assert request.full_url == 'http://camera:8000/camera/settings'
    assert request.get_header('Authorization') == 'Bearer private'
    assert json.loads(request.data) == {'framerate': 30}
    assert opened.call_count == 1
    assert payload == snapshot()


def test_camera_errors_are_forwarded_as_json_without_exposing_token(routes):
    client, _, _ = routes
    error = HTTPError('http://camera', 409, 'Conflict', {},
                      io.BytesIO(b'{"error":"Stop recording before changing camera settings"}'))
    with patch.object(web_app, 'request_camera_settings', side_effect=error):
        response = client.post('/api/camera/settings', json={'awb_mode': 'auto'})
    assert response.status_code == 409
    assert 'Stop recording' in response.json['message']
    assert 'private' not in response.json['message']
