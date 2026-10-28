"""Camera preview can run without constructing or controlling a tracker."""

from threading import Event, Timer
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from flask import Response

from pistream import web_app
from pistream.lifecycle import TrackerLifecycle


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(web_app, '_lifecycle', None)
    monkeypatch.setattr(web_app, '_overlay_enabled', True)
    monkeypatch.setattr(web_app, '_config', {})
    return web_app.app.test_client()


def test_enable_preview_consumes_only_pending_start(client, monkeypatch):
    tracker = SimpleNamespace(running=True, cleanup=Mock())
    lifecycle = SimpleNamespace(
        tracker=tracker, cancel_scheduled_start=Mock(), start=Mock(), stop=Mock(),
    )
    monkeypatch.setattr(web_app, '_lifecycle', lifecycle)
    response = client.post('/api/preview', json={'enabled': True})
    assert response.status_code == 200
    assert response.get_json() == {'status': 'ok', 'enabled': True}
    lifecycle.cancel_scheduled_start.assert_called_once_with()
    lifecycle.start.assert_not_called()
    lifecycle.stop.assert_not_called()
    tracker.cleanup.assert_not_called()
    assert tracker.running
    assert web_app._overlay_enabled


def test_disable_preview_does_not_control_the_tracker(client, monkeypatch):
    lifecycle = SimpleNamespace(
        tracker=None, cancel_scheduled_start=Mock(), start=Mock(), stop=Mock(),
    )
    monkeypatch.setattr(web_app, '_lifecycle', lifecycle)
    response = client.post('/api/preview', json={'enabled': False})
    assert response.status_code == 200
    assert response.get_json() == {'status': 'ok', 'enabled': False}
    lifecycle.cancel_scheduled_start.assert_not_called()
    lifecycle.start.assert_not_called()
    lifecycle.stop.assert_not_called()


def test_preview_is_available_without_a_lifecycle(client):
    response = client.post('/api/preview', json={'enabled': True})
    assert response.status_code == 200
    assert response.get_json()['enabled'] is True


@pytest.mark.parametrize('body', [
    b'', b'{', b'null', b'[]', b'{}', b'{"enabled":1}', b'{"enabled":"true"}',
    b'{"enabled":null}', b'{"enabled":true,"overlay":false}',
])
def test_preview_rejects_malformed_unknown_and_non_boolean_data(client, body):
    response = client.post('/api/preview', data=body, content_type='application/json')
    assert response.status_code == 400
    assert response.get_json()['status'] == 'error'


def test_preview_rejects_cross_origin_mutation(client):
    response = client.post('/api/preview', json={'enabled': True},
                           headers={'Origin': 'https://other.example'})
    assert response.status_code == 403


@pytest.mark.parametrize('overlay', [True, False])
def test_stopped_preview_proxies_camera_even_when_overlay_is_selected(client, monkeypatch, overlay):
    monkeypatch.setattr(web_app, '_overlay_enabled', overlay)
    with patch.object(web_app, '_proxy_camera_stream', return_value=Response(b'raw camera')) as proxy, \
            patch.object(web_app, '_generate_overlay_preview', return_value=iter([b'not raw'])), \
            patch.object(web_app, 'HumanTracker') as tracker_factory:
        response = client.get('/video_feed')
        assert response.data == b'raw camera'
    proxy.assert_called_once_with()
    tracker_factory.assert_not_called()
    assert web_app._overlay_enabled is overlay


def test_running_overlay_preview_uses_processed_frames(client, monkeypatch):
    tracker = SimpleNamespace(running=True, capture=object())
    monkeypatch.setattr(web_app, '_lifecycle', SimpleNamespace(tracker=tracker))
    with patch.object(web_app, '_proxy_camera_stream') as proxy, \
            patch.object(web_app, '_generate_overlay_preview', return_value=iter([b'annotated'])):
        response = client.get('/video_feed')
        assert response.data == b'annotated'
    proxy.assert_not_called()


def test_running_raw_preview_proxies_without_changing_tracker(client, monkeypatch):
    tracker = SimpleNamespace(running=True, capture=object())
    monkeypatch.setattr(web_app, '_lifecycle', SimpleNamespace(tracker=tracker))
    monkeypatch.setattr(web_app, '_overlay_enabled', False)
    with patch.object(web_app, '_proxy_camera_stream', return_value=Response(b'raw camera')):
        response = client.get('/video_feed')
        assert response.data == b'raw camera'
    assert tracker.running


def test_unavailable_camera_falls_back_without_starting_tracker(client):
    with patch.object(web_app, '_proxy_camera_stream', return_value=None) as proxy, \
            patch.object(web_app, '_generate_overlay_preview', return_value=iter([b'no signal'])), \
            patch.object(web_app, 'HumanTracker') as tracker_factory:
        response = client.get('/video_feed')
        assert response.data == b'no signal'
    proxy.assert_called_once_with()
    tracker_factory.assert_not_called()


def test_preview_consumes_real_pending_timer_without_constructing_tracker(client, monkeypatch):
    timers = []
    factory = Mock()
    lifecycle = TrackerLifecycle(factory, lambda _tracker: None)

    def make_timer(delay, callback):
        timer = Timer(delay, callback)
        timer.start = Mock()
        timers.append(timer)
        return timer

    monkeypatch.setattr(web_app, '_lifecycle', lifecycle)
    with patch('pistream.lifecycle.Timer', side_effect=make_timer):
        lifecycle.schedule_start(120)
        response = client.post('/api/preview', json={'enabled': True})
        assert response.status_code == 200
        timers[0].function()  # A timer callback may already have entered before cancel().
    factory.assert_not_called()
    assert lifecycle.tracker is None


def test_preview_preserves_real_running_session(client, monkeypatch):
    loop_entered = Event()
    tracker = SimpleNamespace(
        running=False, connect=Mock(return_value=True), cleanup=Mock(),
        cleanup_pending=False, remote_stop_pending=False, local_recording_path=None,
    )

    def loop(owner):
        loop_entered.set()
        while owner.running:
            Event().wait(0.01)

    factory = Mock(return_value=tracker)
    lifecycle = TrackerLifecycle(factory, loop)
    monkeypatch.setattr(web_app, '_lifecycle', lifecycle)
    try:
        assert lifecycle.start()['status'] == 'ok'
        assert loop_entered.wait(1)
        response = client.post('/api/preview', json={'enabled': True})
        assert response.status_code == 200
        assert lifecycle.tracker is tracker
        assert tracker.running
        tracker.cleanup.assert_not_called()
        factory.assert_called_once_with()
    finally:
        lifecycle.stop()
