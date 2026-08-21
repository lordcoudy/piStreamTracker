"""Drive shipped Flask listing/path routes (same handlers the UI calls)."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import quote


class WebRecordingsRouteTests(unittest.TestCase):
    def test_list_includes_mp4_and_avi_not_other_suffixes(self):
        from pistream import web_app

        with TemporaryDirectory() as tmp:
            rec = Path(tmp)
            (rec / 'talk.mp4').write_bytes(b'x')
            (rec / 'legacy.avi').write_bytes(b'y')
            (rec / 'notes.md').write_text('no')
            web_app._config = {'tracker': {'output_dir': tmp}}
            client = web_app.app.test_client()
            payload = client.get('/api/recordings').get_json()
            names = {item['name'] for item in payload['files']}
        self.assertIn('talk.mp4', names)
        self.assertIn('legacy.avi', names)
        self.assertNotIn('notes.md', names)

    def test_download_rejects_parent_escape(self):
        from pistream import web_app

        with TemporaryDirectory() as tmp:
            web_app._config = {'tracker': {'output_dir': tmp}}
            client = web_app.app.test_client()
            response = client.get('/api/recordings/' + quote('../secret.txt', safe=''))
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.get_json().get('status'), 'error')

    def test_delete_rejects_absolute_path(self):
        from pistream import web_app

        with TemporaryDirectory() as tmp:
            web_app._config = {'tracker': {'output_dir': tmp}}
            with web_app.app.app_context():
                body, status = web_app.api_recordings_delete('/etc/passwd')
        self.assertEqual(status, 403)
        self.assertEqual(body.get_json().get('status'), 'error')

    def test_status_includes_stream_lost_flag(self):
        from pistream import web_app

        web_app._lifecycle = None
        payload = web_app.app.test_client().get('/api/status').get_json()
        self.assertIn('stream_lost', payload)
        self.assertFalse(payload['stream_lost'])

    def test_camera_mode_combines_remote_and_local_artifacts(self):
        from pistream import web_app

        with TemporaryDirectory() as tmp:
            (Path(tmp) / 'screenshot.jpg').write_bytes(b'local')
            web_app._config = {
                'tracker': {'output_dir': tmp, 'recording_mode': 'camera'},
                'network': {'camera_ip': 'camera'},
                'camera': {'port': 8000, 'token': 'secret'},
            }
            remote = [{
                'name': 'lecture.mp4', 'size': '1.0 MB', 'bytes': 1_000_000,
                'mtime': 20, 'date': '2026-01-01 10:00', 'type': 'mp4',
            }]
            with patch.object(web_app, 'fetch_remote_recordings', return_value=remote):
                payload = web_app.app.test_client().get('/api/recordings').get_json()

        sources = {entry['name']: entry['source'] for entry in payload['files']}
        self.assertEqual(sources['lecture.mp4'], 'camera')
        self.assertEqual(sources['screenshot.jpg'], 'local')

    def test_camera_recording_download_is_proxied(self):
        from pistream import web_app

        class RemoteResponse:
            headers = {
                'Content-Type': 'video/mp4',
                'Content-Length': '4',
                'Content-Disposition': 'attachment; filename="talk.mp4"',
            }

            def __init__(self):
                self.chunks = iter((b'data', b''))
                self.closed = False

            def read(self, _size):
                return next(self.chunks)

            def close(self):
                self.closed = True

        web_app._config = {
            'tracker': {'recording_mode': 'camera'},
            'network': {'camera_ip': 'camera'},
            'camera': {'port': 8000, 'token': 'secret'},
        }
        remote = RemoteResponse()
        with patch.object(web_app, 'open_remote_recording', return_value=remote) as opened:
            response = web_app.app.test_client().get(
                '/api/recordings/talk.mp4?source=camera'
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b'data')
        self.assertTrue(remote.closed)
        opened.assert_called_once_with('http://camera:8000', 'talk.mp4', 'secret')

    def test_camera_recording_delete_is_forwarded(self):
        from pistream import web_app

        web_app._config = {
            'tracker': {'recording_mode': 'camera'},
            'network': {'camera_ip': 'camera'},
            'camera': {'port': 8000, 'token': 'secret'},
        }
        with patch.object(web_app, 'delete_remote_recording') as delete:
            response = web_app.app.test_client().delete(
                '/api/recordings/talk.mp4?source=camera'
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['status'], 'ok')
        delete.assert_called_once_with('http://camera:8000', 'talk.mp4', 'secret')

    def test_active_local_recording_cannot_be_deleted(self):
        from pistream import web_app

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / 'talk.mp4'
            target.write_bytes(b'active')
            tracker = SimpleNamespace(
                running=True,
                capture=object(),
                local_recording_path=target,
            )
            web_app._config = {'tracker': {'output_dir': tmp, 'recording_mode': 'local'}}
            with patch.object(web_app, '_lifecycle', SimpleNamespace(tracker=tracker)):
                response = web_app.app.test_client().delete('/api/recordings/talk.mp4')

        self.assertEqual(response.status_code, 409)
        self.assertIn('active', response.get_json()['message'].lower())


class WebInputValidationTests(unittest.TestCase):
    def test_cross_origin_control_request_is_rejected(self):
        from pistream import web_app

        response = web_app.app.test_client().post(
            '/api/reset', headers={'Origin': 'https://attacker.example'}
        )
        self.assertEqual(response.status_code, 403)
        self.assertIn('cross-origin', response.get_json()['message'].lower())

    def test_responses_include_clickjacking_and_mime_protections(self):
        from pistream import web_app

        response = web_app.app.test_client().get('/api/status')
        self.assertEqual(response.headers['X-Frame-Options'], 'DENY')
        self.assertEqual(response.headers['X-Content-Type-Options'], 'nosniff')

    def test_zero_interval_is_rejected_without_mutating_tracker(self):
        from pistream import web_app

        tracker = SimpleNamespace(
            running=True,
            capture=object(),
            detection_interval=10,
            motors=SimpleNamespace(),
            detector=SimpleNamespace(confidence=0.5),
        )
        lifecycle = SimpleNamespace(tracker=tracker)
        with patch.object(web_app, '_lifecycle', lifecycle):
            response = web_app.app.test_client().post('/api/settings', json={'interval': 0})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(tracker.detection_interval, 10)

    def test_settings_update_is_atomic_when_one_value_is_invalid(self):
        from pistream import web_app

        tracker = SimpleNamespace(
            running=True,
            capture=object(),
            detection_interval=10,
            motors=SimpleNamespace(speed_factor=1.0),
            detector=SimpleNamespace(confidence=0.5),
        )
        lifecycle = SimpleNamespace(tracker=tracker)
        with patch.object(web_app, '_lifecycle', lifecycle):
            response = web_app.app.test_client().post(
                '/api/settings', json={'ev3_speed': 1.5, 'interval': 0}
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(tracker.motors.speed_factor, 1.0)
        self.assertEqual(tracker.detection_interval, 10)

    def test_unknown_motor_direction_is_rejected(self):
        from pistream import web_app

        tracker = SimpleNamespace(
            running=True,
            capture=object(),
            motors=SimpleNamespace(connected=True),
        )
        lifecycle = SimpleNamespace(tracker=tracker)
        with patch.object(web_app, '_lifecycle', lifecycle):
            response = web_app.app.test_client().post(
                '/api/motor_move', json={'direction': 'diagonal', 'degrees': 10}
            )

        self.assertEqual(response.status_code, 400)
