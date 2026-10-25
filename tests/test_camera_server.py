"""Camera HTTP/file helpers without requiring Raspberry Pi camera libraries."""

import importlib
import io
import os
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import BoundedSemaphore, Condition
from unittest.mock import Mock, patch


def import_camera_module(fresh=False):
    if not fresh and 'camera' in sys.modules:
        return sys.modules['camera']

    picamera2 = types.ModuleType('picamera2')
    encoders = types.ModuleType('picamera2.encoders')
    outputs = types.ModuleType('picamera2.outputs')

    class Placeholder:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    picamera2.Picamera2 = Placeholder
    encoders.H264Encoder = Placeholder
    encoders.JpegEncoder = Placeholder
    outputs.FfmpegOutput = Placeholder
    outputs.FileOutput = Placeholder

    with patch.dict(sys.modules, {
        'picamera2': picamera2,
        'picamera2.encoders': encoders,
        'picamera2.outputs': outputs,
    }):
        if fresh:
            source = Path(__file__).resolve().parents[1] / 'camera.py'
            spec = importlib.util.spec_from_file_location('camera_import_probe', source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        return importlib.import_module('camera')


class CameraServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.camera = import_camera_module()

    def handler(self):
        handler = self.camera.StreamingHandler.__new__(self.camera.StreamingHandler)
        handler.client_address = ('127.0.0.1', 12345)
        handler.wfile = io.BytesIO()
        handler.send_response = lambda _status: None
        handler.send_header = lambda _name, _value: None
        handler.end_headers = lambda: None
        return handler

    def test_import_does_not_load_config_or_create_recording_directory(self):
        with patch('pistream.config.load_config') as load, patch.object(Path, 'mkdir') as mkdir:
            import_camera_module(fresh=True)
        load.assert_not_called()
        mkdir.assert_not_called()

    def test_main_without_flags_loads_default_config(self):
        from pistream.config import default_config

        camera = import_camera_module(fresh=True)
        server = Mock()
        server.serve_forever.side_effect = KeyboardInterrupt
        with TemporaryDirectory() as tmp:
            config = default_config()
            config['camera']['recording_dir'] = tmp
            with patch.object(camera, 'load_config', return_value=config) as load, \
                    patch.object(camera, 'Picamera2', return_value=Mock()), \
                    patch.object(camera, 'StreamingServer', return_value=server):
                camera.main([])
        load.assert_called_once_with('config.yaml')
        self.assertEqual(camera.WIDTH, 1280)
        self.assertEqual(camera.HEIGHT, 960)
        self.assertEqual(camera.PORT, 8000)

    def test_main_applies_custom_config_before_initializing_camera_and_server(self):
        camera = import_camera_module(fresh=True)
        picam = Mock()
        server = Mock()
        server.serve_forever.side_effect = KeyboardInterrupt
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / 'custom recordings'
            config_path = Path(tmp) / 'custom config.yaml'
            config_path.write_text(
                'network:\n  camera_ip: "10.10.0.1"\n'
                'camera:\n  host: "127.0.0.1"\n  port: 8123\n'
                '  resolution: {width: 640, height: 360}\n  framerate: 17\n'
                '  jpeg_quality: 62\n  max_stream_clients: 2\n'
                f'  recording_dir: "{output_dir}"\n  token: custom-secret\n'
            )
            with patch.object(camera, 'Picamera2', return_value=picam), \
                    patch.object(camera, 'StreamingServer', return_value=server) as make_server, \
                    patch.dict(os.environ, {'PISTREAM_CAMERA_TOKEN': 'custom-secret'}):
                camera.main(['--config', str(config_path)])
            self.assertTrue(output_dir.is_dir())
            self.assertEqual(camera.OUTPUT_DIR, output_dir)
            self.assertTrue(camera.STREAM_LIMIT.acquire(blocking=False))
            self.assertTrue(camera.STREAM_LIMIT.acquire(blocking=False))
            self.assertFalse(camera.STREAM_LIMIT.acquire(blocking=False))
            camera.STREAM_LIMIT.release()
            camera.STREAM_LIMIT.release()
            self.assertEqual(camera.CAMERA_TOKEN, 'custom-secret')
            self.assertIn('640 x 360', camera.HTML_PAGE)
            self.assertIn('http://10.10.0.1:8123/stream', camera.HTML_PAGE)
        picam.create_video_configuration.assert_called_once_with(
            main={'size': (640, 360)}, controls={'FrameRate': 17},
        )
        make_server.assert_called_once_with(('127.0.0.1', 8123), camera.StreamingHandler)
        self.assertEqual(picam.start_recording.call_args.args[0].kwargs, {'q': 62})
        picam.stop_recording.assert_called_once()

    def test_termination_signal_closes_stream_camera_and_listener(self):
        import signal

        camera = import_camera_module(fresh=True)
        picam, srv = Mock(), Mock()
        handlers = {}

        def install(signum, handler):
            handlers[signum] = handler
            return signal.SIG_DFL

        srv.serve_forever.side_effect = lambda: handlers[signal.SIGTERM](signal.SIGTERM, None)
        with TemporaryDirectory() as tmp, \
                patch.object(camera, 'Picamera2', return_value=picam), \
                patch.object(camera, 'StreamingServer', return_value=srv), \
                patch('signal.signal', side_effect=install):
            config = Path(tmp) / 'config.yaml'
            config.write_text(f'camera:\n  recording_dir: "{tmp}/recordings"\n')
            camera.main(['--config', str(config)])
        picam.stop_recording.assert_called_once()
        picam.close.assert_called_once()
        srv.server_close.assert_called_once()
        self.assertEqual(handlers[signal.SIGTERM], signal.SIG_DFL)

    def test_partial_camera_startup_failure_releases_csi_owner(self):
        camera = import_camera_module(fresh=True)
        picam = Mock()
        picam.configure.side_effect = RuntimeError('configuration failed')
        with TemporaryDirectory() as tmp, patch.object(camera, 'Picamera2', return_value=picam):
            config = Path(tmp) / 'config.yaml'
            config.write_text(f'camera:\n  recording_dir: "{tmp}/recordings"\n')
            with self.assertRaisesRegex(RuntimeError, 'configuration failed'):
                camera.main(['--config', str(config)])
        picam.close.assert_called_once()

    def test_stream_output_reports_bytes_written(self):
        output = self.camera.StreamingOutput()
        self.assertEqual(output.write(b'jpeg'), 4)
        self.assertEqual(output.frame, b'jpeg')

    def test_stream_client_cap_rejects_without_waiting_or_releasing_a_slot(self):
        handler = self.handler()
        handler.path = '/stream'
        handler.send_error = Mock()
        limiter = Mock()
        limiter.acquire.return_value = False
        with patch.object(self.camera, 'STREAM_LIMIT', limiter):
            handler.do_GET()
        limiter.acquire.assert_called_once_with(blocking=False)
        limiter.release.assert_not_called()
        handler.send_error.assert_called_once_with(503, 'Too many stream clients')

    def test_stream_disconnect_releases_its_slot(self):
        handler = self.handler()
        handler.path = '/stream'
        handler.wfile = Mock()
        handler.wfile.write.side_effect = BrokenPipeError
        handler.output = types.SimpleNamespace(condition=Condition(), frame=b'jpeg')
        limiter = BoundedSemaphore(1)
        with patch.object(self.camera, 'STREAM_LIMIT', limiter), \
                patch.object(handler.output.condition, 'wait', return_value=True):
            handler.do_GET()
        self.assertTrue(limiter.acquire(blocking=False))
        self.assertFalse(limiter.acquire(blocking=False))

    def test_every_record_get_path_requires_the_camera_token(self):
        for path in ('/record/list', '/record/status', '/record/files/clip.mp4', '/record/nope'):
            with self.subTest(path=path):
                handler = self.handler()
                handler.path = path
                handler.headers = {}
                with patch.object(self.camera, 'CAMERA_TOKEN', 'secret'):
                    handler.do_GET()
                self.assertEqual(handler.wfile.getvalue(), b'{"error": "unauthorized"}')

    def test_recording_download_is_limited_to_stat_size(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'clip.mp4'
            path.write_bytes(b'video-data')
            handler = self.handler()
            with patch.object(self.camera, 'OUTPUT_DIR', Path(tmp)):
                handler._send_recording('clip.mp4')
        self.assertEqual(handler.wfile.getvalue(), b'video-data')

    def test_active_camera_recording_cannot_be_deleted(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'clip.mp4'
            path.write_bytes(b'active')
            handler = self.handler()
            handler.path = '/record/files/clip.mp4'
            handler._authorized = lambda: True
            handler.recorder = types.SimpleNamespace(
                recording=True,
                current_file=str(path),
            )
            with patch.object(self.camera, 'OUTPUT_DIR', Path(tmp)):
                handler.do_DELETE()
                self.assertTrue(path.exists())

    def test_nested_camera_recording_path_is_rejected(self):
        with TemporaryDirectory() as tmp:
            handler = self.handler()
            with patch.object(self.camera, 'OUTPUT_DIR', Path(tmp)):
                with self.assertRaises(ValueError):
                    handler._recording_path('../clip.mp4')

    def test_encoder_stop_failure_keeps_retryable_recording_state(self):
        class BrokenCamera:
            def stop_encoder(self, _encoder):
                raise OSError('busy')

        with TemporaryDirectory() as tmp:
            recorder = self.camera.CameraRecorder(BrokenCamera(), Path(tmp))
            recorder.recording = True
            recorder.current_file = str(Path(tmp) / 'clip.mp4')
            with self.assertRaisesRegex(RuntimeError, 'did not stop'):
                recorder.stop()

        self.assertTrue(recorder.recording)
        self.assertIsNotNone(recorder.current_file)

    def test_unknown_record_path_returns_json(self):
        handler = self.handler()
        handler.path = '/record/not-a-route'
        handler._authorized = lambda: True
        handler.do_GET()
        body = handler.wfile.getvalue()
        self.assertTrue(body.startswith(b'{'), body)
        self.assertIn(b'not found', body.lower())

    def test_unknown_record_post_returns_json(self):
        handler = self.handler()
        handler.path = '/record/nope'
        handler._authorized = lambda: True
        handler.do_POST()
        body = handler.wfile.getvalue()
        self.assertTrue(body.startswith(b'{'), body)
        self.assertIn(b'not found', body.lower())

    def test_unknown_record_delete_returns_json(self):
        handler = self.handler()
        handler.path = '/record/nope'
        handler._authorized = lambda: True
        handler.do_DELETE()
        body = handler.wfile.getvalue()
        self.assertTrue(body.startswith(b'{'), body)
        self.assertIn(b'not found', body.lower())

    def test_camera_busy_hint_points_at_pitracker_service(self):
        hint = self.camera.camera_busy_hint('192.168.100.1', 8000)
        self.assertIn('pitracker.service', hint)
        self.assertIn('http://192.168.100.1:8000/stream', hint)
        self.assertIn('systemctl stop pitracker', hint)
