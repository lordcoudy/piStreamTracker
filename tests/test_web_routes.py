"""Drive shipped Flask listing/path routes (same handlers the UI calls)."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, RLock, Thread
from types import SimpleNamespace
from unittest.mock import Mock, patch
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

    def test_stopping_tracker_keeps_its_open_recording_protected(self):
        from pistream import web_app

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / 'talk.mp4'
            target.write_bytes(b'encoder still owns this file')
            tracker = SimpleNamespace(
                running=False,
                recording=False,
                capture=None,
                local_recording_path=target,
            )
            with (
                patch.object(web_app, '_config', {'tracker': {'output_dir': tmp}}),
                patch.object(web_app, '_lifecycle', SimpleNamespace(tracker=tracker)),
            ):
                client = web_app.app.test_client()
                listed = client.get('/api/recordings').get_json()['files']
                response = client.delete('/api/recordings/talk.mp4')
                self.assertTrue(listed[0]['active'])
                self.assertEqual(response.status_code, 409)
                self.assertTrue(target.exists())

                tracker.local_recording_path = None
                response = client.delete('/api/recordings/talk.mp4')
                self.assertEqual(response.status_code, 200)
                self.assertFalse(target.exists())


class WebShutdownRouteTests(unittest.TestCase):
    def setUp(self):
        from pistream import web_app

        remembered_error = patch.object(web_app, '_last_recording_error', None, create=True)
        remembered_error.start()
        self.addCleanup(remembered_error.stop)

    def tracker(self, **changes):
        values = dict(
            running=False,
            recording=True,
            current_detection=None,
            capture=None,
            motors=SimpleNamespace(connected=False),
            fps=0.0,
            zoom_level=1.0,
            horizon_correction=False,
            recording_error=None,
            remote_stop_pending=False,
            cleanup_pending=False,
        )
        values.update(changes)
        return SimpleNamespace(**values)

    def assert_cleanup_response_does_not_wait_for_writer(self, endpoint, expected_status,
                                                       status_poll=False):
        from pistream import web_app
        from pistream.lifecycle import TrackerLifecycle
        from pistream.record import _RecordingThread
        from pistream.track import HumanTracker

        closing = Event()
        finish_close = Event()
        loop_started = Event()

        class GatedVideoWriter:
            failed_once = False

            def isOpened(self):
                return True

            def release(self):
                closing.set()
                finish_close.wait(3)
                if not self.failed_once:
                    self.failed_once = True
                    raise OSError('Encoder failed while finalizing the file')

        with patch('pistream.record.cv2.VideoWriter', return_value=GatedVideoWriter()):
            recorder = _RecordingThread('lecture.mp4', 64, 48, 30, encoder='mjpg')
        tracker = HumanTracker.__new__(HumanTracker)
        tracker._record_lock = RLock()
        tracker._recording = True
        tracker._recording_backend = 'local'
        tracker._rec_thread = recorder
        tracker._recording_error = None
        tracker.remote_stop_pending = False
        tracker._cleaned = False
        tracker._async_detector = None
        tracker._shift_logger = None
        tracker.motors = Mock()
        tracker.capture = SimpleNamespace(stop=Mock())
        tracker.running = False
        tracker.current_detection = None
        tracker._fps = 0.0
        tracker.zoom_level = 1.0
        tracker.horizon_correction = False
        tracker.connect = Mock(return_value=True)

        def loop(trk):
            loop_started.set()
            while trk.running:
                finish_close.wait(0.001)

        lifecycle = TrackerLifecycle(lambda: tracker, loop, join_timeout=0.01)
        completed = Event()
        responses = []

        def request(path, signal=None):
            response = web_app.app.test_client().post(path)
            responses.append((path, response.status_code))
            if signal:
                signal.set()

        with patch.object(web_app, '_lifecycle', lifecycle):
            self.assertEqual(lifecycle.start()['status'], 'ok')
            self.assertTrue(loop_started.wait(1))
            stop_request = Thread(
                target=request, args=('/api/stop', completed if endpoint == 'stop' else None)
            )
            start_request = None
            poll_request = None
            getter_patch = None
            stop_request.start()
            try:
                self.assertTrue(closing.wait(1))
                if status_poll:
                    reading_error = Event()
                    read_error = HumanTracker.recording_error.fget

                    def recording_error(trk):
                        reading_error.set()
                        return read_error(trk)

                    getter_patch = patch.object(
                        HumanTracker, 'recording_error', property(recording_error)
                    )
                    getter_patch.start()
                    poll_request = Thread(
                        target=lambda: web_app.app.test_client().get('/api/status')
                    )
                    poll_request.start()
                    self.assertTrue(reading_error.wait(1))
                if endpoint == 'start':
                    start_request = Thread(target=request, args=('/api/start', completed))
                    start_request.start()
                self.assertTrue(completed.wait(0.5), 'HTTP control waited for encoder finalization')
                self.assertIn(('/api/' + endpoint, expected_status), responses)
            finally:
                finish_close.set()
                stop_request.join(2)
                if start_request:
                    start_request.join(2)
                if poll_request:
                    poll_request.join(2)
                if getter_patch:
                    getter_patch.stop()
                if lifecycle.thread:
                    lifecycle.thread.join(2)
                response = web_app.app.test_client().post('/api/stop')
                self.assertEqual(response.status_code, 200)
                self.assertIsNone(lifecycle.tracker)
                self.assertIn('finalizing', web_app._last_recording_error)

    def test_stopping_response_returns_before_local_writer_closes(self):
        self.assert_cleanup_response_does_not_wait_for_writer('stop', 202)

    def test_start_conflict_returns_before_local_writer_closes(self):
        self.assert_cleanup_response_does_not_wait_for_writer('start', 409)

    def test_start_conflict_returns_while_status_poll_waits_for_writer(self):
        self.assert_cleanup_response_does_not_wait_for_writer('start', 409, status_poll=True)

    def test_status_preserves_pending_camera_stop_without_retrying_it(self):
        from pistream import web_app

        tracker = self.tracker(remote_stop_pending=True)
        stop = Mock(return_value={'status': 'recording_stop_pending'})
        lifecycle = SimpleNamespace(tracker=tracker, stopping=False, stop=stop)
        with patch.object(web_app, '_lifecycle', lifecycle):
            client = web_app.app.test_client()
            for _ in range(3):
                payload = client.get('/api/status').get_json()
                self.assertFalse(payload['running'])
                self.assertTrue(payload['recording'])
                self.assertTrue(payload['recording_stop_pending'])
                self.assertFalse(payload['stopping'])
                self.assertIn('unconfirmed', payload['warning'].lower())
        stop.assert_not_called()

    def test_status_exposes_sticky_writer_failure_and_unfinished_cleanup(self):
        from pistream import web_app

        tracker = self.tracker(
            recording=False,
            recording_error='Encoder exited before finishing the file',
            cleanup_pending=True,
        )
        with patch.object(
            web_app, '_lifecycle', SimpleNamespace(tracker=tracker, stopping=True)
        ):
            payload = web_app.app.test_client().get('/api/status').get_json()
        self.assertFalse(payload['recording'])
        self.assertTrue(payload['stopping'])
        self.assertEqual(payload['recording_error'], tracker.recording_error)
        self.assertTrue(payload['warning'])

    def test_failed_recording_stop_keeps_tracking_and_record_retry_available(self):
        from pistream import web_app

        tracker = self.tracker(
            running=True,
            capture=SimpleNamespace(stream_lost=False),
            remote_stop_pending=True,
        )
        with patch.object(
            web_app, '_lifecycle', SimpleNamespace(tracker=tracker, stopping=False)
        ):
            payload = web_app.app.test_client().get('/api/status').get_json()
        self.assertTrue(payload['running'])
        self.assertTrue(payload['recording_stop_pending'])
        self.assertIn('Stop Rec', payload['warning'])
        self.assertNotIn('Tracking has stopped', payload['warning'])

    def test_healthy_active_local_recorder_does_not_report_cleanup_warning(self):
        from pistream import web_app

        tracker = self.tracker(
            running=True,
            capture=SimpleNamespace(stream_lost=False),
            cleanup_pending=True,
        )
        with patch.object(
            web_app, '_lifecycle', SimpleNamespace(tracker=tracker, stopping=False)
        ):
            payload = web_app.app.test_client().get('/api/status').get_json()
        self.assertTrue(payload['recording'])
        self.assertFalse(payload['stopping'])
        self.assertIsNone(payload['warning'])

    def test_failed_active_local_writer_reports_failure_instead_of_cleanup(self):
        from pistream import web_app

        tracker = self.tracker(
            running=True,
            recording=False,
            capture=SimpleNamespace(stream_lost=False),
            cleanup_pending=True,
            recording_error='Encoder exited unexpectedly',
        )
        with patch.object(
            web_app, '_lifecycle', SimpleNamespace(tracker=tracker, stopping=False)
        ):
            payload = web_app.app.test_client().get('/api/status').get_json()
        self.assertIn('Encoder exited unexpectedly', payload['warning'])
        self.assertNotIn('cleanup', payload['warning'].lower())

    def test_idle_status_clears_shutdown_and_error_fields(self):
        from pistream import web_app

        with patch.object(web_app, '_lifecycle', None):
            payload = web_app.app.test_client().get('/api/status').get_json()
        self.assertFalse(payload['recording_stop_pending'])
        self.assertFalse(payload['stopping'])
        self.assertIsNone(payload['recording_error'])
        self.assertIsNone(payload['warning'])

    def test_writer_error_survives_stop_and_new_session_until_recording_starts(self):
        from pistream import web_app

        old_tracker = self.tracker(
            running=True,
            recording=False,
            capture=SimpleNamespace(stream_lost=False),
        )
        new_tracker = self.tracker(
            running=True,
            recording=False,
            capture=SimpleNamespace(stream_lost=False),
        )
        lifecycle = SimpleNamespace(tracker=old_tracker, stopping=False)

        def stop():
            old_tracker.running = False
            old_tracker.recording_error = 'Encoder failed while closing the file'
            lifecycle.tracker = None
            return {'status': 'ok'}

        def start():
            lifecycle.tracker = new_tracker
            return {'status': 'ok'}

        lifecycle.stop = stop
        lifecycle.start = start
        new_tracker.start_recording = lambda: False
        with patch.object(web_app, '_lifecycle', lifecycle):
            client = web_app.app.test_client()
            self.assertEqual(client.post('/api/stop').status_code, 200)
            self.assertIsNone(lifecycle.tracker)
            self.assertEqual(
                client.get('/api/status').get_json()['recording_error'],
                old_tracker.recording_error,
            )
            self.assertEqual(client.post('/api/start').status_code, 200)
            self.assertEqual(
                client.get('/api/status').get_json()['recording_error'],
                old_tracker.recording_error,
            )
            self.assertEqual(client.post('/api/record').status_code, 503)
            self.assertIn('Encoder failed', client.get('/api/status').get_json()['warning'])

            def successful_recording():
                new_tracker.recording = True
                return True

            new_tracker.start_recording = successful_recording
            self.assertEqual(client.post('/api/record').status_code, 200)
            payload = client.get('/api/status').get_json()
            self.assertIsNone(payload['recording_error'])
            self.assertIsNone(payload['warning'])

    def test_start_returns_conflict_until_camera_stop_is_resolved(self):
        from pistream import web_app

        lifecycle = SimpleNamespace(tracker=None, start=Mock(return_value={
            'status': 'recording_stop_pending', 'message': 'Camera stop is unconfirmed',
        }))
        with patch.object(web_app, '_lifecycle', lifecycle):
            response = web_app.app.test_client().post('/api/start')
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json()['status'], 'recording_stop_pending')

    def test_start_returns_conflict_while_local_cleanup_is_unfinished(self):
        from pistream import web_app

        lifecycle = SimpleNamespace(tracker=None, start=Mock(return_value={'status': 'stopping'}))
        with patch.object(web_app, '_lifecycle', lifecycle):
            response = web_app.app.test_client().post('/api/start')
        self.assertEqual(response.status_code, 409)

    def test_failed_explicit_camera_stop_returns_unavailable(self):
        from pistream import web_app

        lifecycle = SimpleNamespace(tracker=None, stop=Mock(return_value={
            'status': 'recording_stop_pending', 'message': 'Camera stop is unconfirmed',
        }))
        with patch.object(web_app, '_lifecycle', lifecycle):
            response = web_app.app.test_client().post('/api/stop')
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()['status'], 'recording_stop_pending')
        lifecycle.stop.assert_called_once_with()

    def test_failed_local_record_stop_reports_unfinished_cleanup(self):
        from pistream import web_app

        tracker = self.tracker(
            running=True,
            capture=SimpleNamespace(stream_lost=False),
        )
        tracker.stop_recording = Mock(return_value=False)
        with patch.object(web_app, '_lifecycle', SimpleNamespace(tracker=tracker)):
            response = web_app.app.test_client().post('/api/record')
        self.assertEqual(response.status_code, 503)
        self.assertIn('cleanup', response.get_json()['message'].lower())
        self.assertNotIn('camera', response.get_json()['message'].lower())

    def test_failed_camera_record_stop_reports_missing_confirmation(self):
        from pistream import web_app

        tracker = self.tracker(
            running=True,
            capture=SimpleNamespace(stream_lost=False),
            remote_stop_pending=True,
        )
        tracker.stop_recording = Mock(return_value=False)
        with patch.object(web_app, '_lifecycle', SimpleNamespace(tracker=tracker)):
            response = web_app.app.test_client().post('/api/record')
        self.assertEqual(response.status_code, 503)
        self.assertIn('Camera did not confirm', response.get_json()['message'])

    def test_late_old_cleanup_error_cannot_replace_successful_new_recording(self):
        from pistream import web_app

        old_tracker = self.tracker(recording_error='Previous encoder failure')
        new_tracker = self.tracker(
            running=True,
            recording=False,
            capture=SimpleNamespace(stream_lost=False),
        )
        lifecycle = SimpleNamespace(tracker=old_tracker, stopping=False)

        def start_recording():
            new_tracker.recording = True
            return True

        new_tracker.start_recording = start_recording

        def stop():
            # Another request starts recording after the old owner is released,
            # before this Stop request publishes its final cleanup error.
            lifecycle.tracker = new_tracker
            response = web_app.app.test_client().post('/api/record')
            self.assertEqual(response.status_code, 200)
            return {'status': 'ok'}

        lifecycle.stop = stop
        with patch.object(web_app, '_lifecycle', lifecycle):
            client = web_app.app.test_client()
            self.assertEqual(client.post('/api/stop').status_code, 200)
            payload = client.get('/api/status').get_json()
        self.assertTrue(payload['recording'])
        self.assertIsNone(payload['recording_error'])
        self.assertIsNone(payload['warning'])


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

    def test_unknown_api_path_returns_json(self):
        from pistream import web_app

        response = web_app.app.test_client().get('/api/does-not-exist')
        self.assertEqual(response.status_code, 404)
        self.assertTrue(response.is_json)
        self.assertEqual(response.get_json()['status'], 'error')

    def test_wrong_method_returns_json(self):
        from pistream import web_app

        response = web_app.app.test_client().get('/api/start')
        self.assertEqual(response.status_code, 405)
        self.assertTrue(response.is_json)
        self.assertEqual(response.get_json()['status'], 'error')

    def test_unhandled_api_exception_returns_json(self):
        from pistream import web_app

        with patch.object(web_app, '_tracker', side_effect=RuntimeError('boom')):
            response = web_app.app.test_client().get('/api/status')
        self.assertEqual(response.status_code, 500)
        self.assertTrue(response.is_json)
        self.assertEqual(response.get_json()['status'], 'error')
        self.assertNotIn('boom', response.get_json()['message'])

    def test_invalid_json_body_is_reported(self):
        from pistream import web_app

        response = web_app.app.test_client().post(
            '/api/settings',
            data='{not json',
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()['message'], 'Invalid JSON')

    def test_status_serializes_numpy_bbox_shift(self):
        import numpy as np
        from pistream import web_app

        tracker = SimpleNamespace(
            running=True,
            recording=False,
            fps=1.5,
            zoom_level=np.float64(1.0),
            horizon_correction=False,
            current_detection={'bbox': np.array([10, 20, 30, 40], dtype=np.int32)},
            capture=SimpleNamespace(width=640, height=480, stream_lost=False),
            motors=SimpleNamespace(connected=False),
        )
        with patch.object(web_app, '_lifecycle', SimpleNamespace(tracker=tracker)):
            response = web_app.app.test_client().get('/api/status')
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIsInstance(payload['shift_x'], int)
        self.assertIsInstance(payload['shift_y'], int)
        self.assertIsInstance(payload['fps'], float)
