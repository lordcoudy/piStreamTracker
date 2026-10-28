"""Camera HTTP/file helpers without requiring Raspberry Pi camera libraries."""

import importlib
import io
import json
import os
import sys
import types
import unittest
from contextlib import chdir
from http.client import HTTPConnection
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import BoundedSemaphore, Condition, Thread, current_thread
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
    outputs.PyavOutput = Placeholder
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
                    patch.object(camera, 'Picamera2', return_value=Mock()) as make_camera, \
                    patch.object(camera, 'StreamingServer', return_value=server):
                camera.main([])
        load.assert_called_once_with('config.yaml')
        make_camera.assert_called_once_with()
        make_camera.load_tuning_file.assert_not_called()
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

    def test_pi4_tuning_is_loaded_from_checkout_even_with_a_different_working_directory(self):
        camera = import_camera_module(fresh=True)
        picam, srv = Mock(), Mock()
        srv.serve_forever.side_effect = KeyboardInterrupt
        root = Path(__file__).resolve().parents[1]
        profile = root / 'tuning' / 'imx219-160-no-alsc.json'
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / 'pi4.yaml'
            config.write_text(
                'camera:\n  tuning_file: tuning/imx219-160-no-alsc.json\n'
                '  resolution: {width: 1280, height: 720}\n  framerate: 45\n'
                f'  recording_dir: "{tmp}/recordings"\n'
            )
            with chdir(tmp), \
                    patch.object(camera, 'Picamera2', return_value=picam) as make_camera, \
                    patch.object(camera, 'StreamingServer', return_value=srv):
                make_camera.load_tuning_file.side_effect = lambda path: json.loads(
                    Path(path).read_text()
                )
                camera.main(['--config', str(config)])
        make_camera.load_tuning_file.assert_called_once_with(str(profile))
        tuning = make_camera.call_args.kwargs['tuning']
        self.assertEqual(tuning['target'], 'bcm2835')
        self.assertNotIn('rpi.alsc', {name for item in tuning['algorithms'] for name in item})
        picam.create_video_configuration.assert_called_once_with(
            main={'size': (1280, 720)}, controls={'FrameRate': 45},
        )
        picam.close.assert_called_once()

    def test_camera_accepts_an_absolute_tuning_path(self):
        camera = import_camera_module(fresh=True)
        srv = Mock()
        srv.serve_forever.side_effect = KeyboardInterrupt
        with TemporaryDirectory() as tmp:
            profile = Path(tmp) / 'custom.json'
            config = Path(tmp) / 'camera.yaml'
            config.write_text(
                f'camera:\n  tuning_file: "{profile}"\n  recording_dir: "{tmp}/recordings"\n'
            )
            with patch.object(camera, 'Picamera2', return_value=Mock()) as make_camera, \
                    patch.object(camera, 'StreamingServer', return_value=srv):
                camera.main(['--config', str(config)])
        make_camera.load_tuning_file.assert_called_once_with(str(profile))
        make_camera.assert_called_once_with(tuning=make_camera.load_tuning_file.return_value)

    def test_invalid_tuning_does_not_open_camera_and_restores_signal_handler(self):
        import signal

        for error in (FileNotFoundError('missing profile'), ValueError('invalid tuning JSON')):
            with self.subTest(error=error):
                camera = import_camera_module(fresh=True)
                previous = signal.getsignal(signal.SIGTERM)
                with TemporaryDirectory() as tmp:
                    config = Path(tmp) / 'camera.yaml'
                    config.write_text(
                        f'camera:\n  tuning_file: missing.json\n  recording_dir: "{tmp}/rec"\n'
                    )
                    with patch.object(camera, 'Picamera2') as make_camera, \
                            patch.object(camera, 'StreamingServer') as make_server:
                        make_camera.load_tuning_file.side_effect = error
                        with self.assertRaisesRegex(type(error), str(error)):
                            camera.main(['--config', str(config)])
                make_camera.assert_not_called()
                make_server.assert_not_called()
                self.assertIs(signal.getsignal(signal.SIGTERM), previous)

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

    def settings_handler(self, body=b'{}'):
        handler = self.handler()
        handler.path = '/camera/settings'
        handler.rfile = io.BytesIO(body)
        handler.headers = {'Content-Length': str(len(body)), 'Content-Type': 'application/json'}
        handler.send_response = Mock()
        handler.camera_settings = Mock()
        handler.camera_settings.snapshot.return_value = {'settings': {}, 'metadata': {}, 'limits': {}}
        handler.camera_settings.update.return_value = {'settings': {'awb_mode': 'indoor'}}
        handler.recorder = self.camera.CameraRecorder(Mock(), Path('.'))
        handler._authorized = lambda: True
        return handler

    def test_camera_settings_get_and_post_require_camera_token(self):
        for method in ('do_GET', 'do_POST'):
            with self.subTest(method=method):
                handler = self.settings_handler()
                del handler._authorized
                with patch.object(self.camera, 'CAMERA_TOKEN', 'secret'):
                    getattr(handler, method)()
                handler.send_response.assert_called_once_with(401)
                handler.camera_settings.snapshot.assert_not_called()
                handler.camera_settings.update.assert_not_called()

    def test_camera_settings_get_returns_runtime_values(self):
        handler = self.settings_handler()
        handler.do_GET()
        handler.send_response.assert_called_once_with(200)
        self.assertEqual(json.loads(handler.wfile.getvalue()), {
            'status': 'ok', 'settings': {}, 'metadata': {}, 'limits': {},
        })

    def test_camera_settings_post_saves_valid_json_object(self):
        handler = self.settings_handler(b'{"awb_mode": "indoor"}')
        handler.do_POST()
        handler.camera_settings.update.assert_called_once_with({'awb_mode': 'indoor'})
        handler.send_response.assert_called_once_with(200)
        self.assertEqual(json.loads(handler.wfile.getvalue()), {
            'status': 'ok', 'saved': True, 'settings': {'awb_mode': 'indoor'},
        })

    def test_camera_settings_post_rejects_invalid_or_unbounded_json(self):
        for body, headers in (
            (b'[]', {}), (b'{', {}), (b'', {}), (b'{"framerate": NaN}', {}),
            (b'{}', {'Content-Length': '-1'}),
            (b'{}', {'Content-Length': 'four'}),
            (b'{}', {'Content-Length': '5000'}),
            (b'{}', {'Content-Length': '10'}),
            (b'{}', {'Transfer-Encoding': 'chunked'}),
        ):
            with self.subTest(body=body, headers=headers):
                handler = self.settings_handler(body)
                handler.headers.update(headers)
                handler.do_POST()
                handler.send_response.assert_called_once_with(400)
                handler.camera_settings.update.assert_not_called()

    def test_camera_settings_post_refuses_active_recording(self):
        handler = self.settings_handler()
        handler.recorder.recording = True
        handler.do_POST()
        handler.send_response.assert_called_once_with(409)
        handler.camera_settings.update.assert_not_called()

    def test_camera_settings_post_rejects_excessively_nested_json(self):
        handler = self.settings_handler(b'[' * 1500 + b']' * 1500)
        handler.do_POST()
        handler.send_response.assert_called_once_with(400)
        handler.camera_settings.update.assert_not_called()

    def test_camera_settings_errors_have_correct_http_status(self):
        for error, status in ((ValueError('invalid gain'), 400),
                              (RuntimeError('camera restart failed'), 500),
                              (OSError('disk full'), 500)):
            with self.subTest(error=error):
                handler = self.settings_handler()
                handler.camera_settings.update.side_effect = error
                handler.do_POST()
                handler.send_response.assert_called_once_with(status)
                self.assertIn(str(error), json.loads(handler.wfile.getvalue())['error'])

    def test_camera_settings_get_reports_unavailable_manager(self):
        handler = self.settings_handler()
        handler.camera_settings = None
        handler.do_GET()
        handler.send_response.assert_called_once_with(500)

    def test_camera_settings_roundtrip_through_threaded_http_server(self):
        manager = Mock()
        manager.update.return_value = {'settings': {'awb_mode': 'indoor'}, 'metadata': {}, 'limits': {}}
        recorder = self.camera.CameraRecorder(Mock(), Path('.'))
        handler = type('SettingsHandler', (self.camera.StreamingHandler,), {
            'camera_settings': manager, 'recorder': recorder,
        })
        with self.camera.StreamingServer(('127.0.0.1', 0), handler) as srv, \
                patch.object(self.camera, 'CAMERA_TOKEN', 'secret'):
            worker = Thread(target=srv.serve_forever, kwargs={'poll_interval': 0.01})
            worker.start()
            try:
                connection = HTTPConnection(*srv.server_address, timeout=2)
                connection.request('GET', '/camera/settings')
                response = connection.getresponse()
                self.assertEqual(response.status, 401)
                response.read()
                connection.close()
                connection = HTTPConnection(*srv.server_address, timeout=2)
                connection.request('POST', '/camera/settings', b'{"awb_mode":"indoor"}', {
                    'Authorization': 'Bearer secret', 'Content-Type': 'application/json',
                })
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertTrue(json.loads(response.read())['saved'])
                connection.close()
            finally:
                srv.shutdown()
                worker.join(2)
        manager.update.assert_called_once_with({'awb_mode': 'indoor'})

    def test_recorder_lock_can_be_reentered_for_settings_transaction(self):
        recorder = self.camera.CameraRecorder(Mock(), Path('.'))
        with recorder._lock:
            acquired = recorder._lock.acquire(blocking=False)
            self.assertTrue(acquired, 'Settings and recording need one shared reentrant lock')
            if acquired:
                recorder._lock.release()

    def test_startup_applies_persisted_white_balance_exposure_and_flicker(self):
        camera = import_camera_module(fresh=True)
        picam, srv = Mock(), Mock()
        picam.camera_controls = {'AeFlickerMode': (0, 1, 0), 'AeFlickerPeriod': (0, 100000, 0)}
        srv.serve_forever.side_effect = KeyboardInterrupt
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / 'settings.yaml'
            config.write_text(
                'camera:\n  framerate: 25\n  awb_mode: manual\n'
                '  colour_gains: [1.7, 1.3]\n  exposure_mode: manual\n'
                '  exposure_time_us: 10000\n  analogue_gain: 2\n'
                f'  recording_dir: "{tmp}/recordings"\n'
            )
            with patch.object(camera, 'Picamera2', return_value=picam), \
                    patch.object(camera, 'StreamingServer', return_value=srv):
                camera.main(['--config', str(config)])
            self.assertEqual(camera.CONFIG_PATH, config.resolve())
        picam.set_controls.assert_called_once_with({
            'AwbEnable': False, 'AeEnable': False, 'ColourGains': (1.7, 1.3),
            'ExposureTime': 10000, 'AnalogueGain': 2.0,
            'FrameDurationLimits': (40000, 40000), 'AeFlickerMode': 1, 'AeFlickerPeriod': 10000,
        })

    def test_live_geometry_change_rebuilds_stream_and_keeps_output_buffer(self):
        camera = import_camera_module(fresh=True)
        picam, srv = Mock(), Mock()
        picam.camera_controls = {}
        picam.capture_metadata.return_value = {}
        output_seen = []
        def serve():
            output = camera.StreamingHandler.output
            output.write(b'stale frame')
            output_seen.append(output)
            result = camera.StreamingHandler.camera_settings.update({
                'resolution': {'width': 1280, 'height': 960}, 'framerate': 30,
            })
            self.assertEqual(result['settings']['resolution']['height'], 960)
            self.assertIs(camera.StreamingHandler.output, output)
            self.assertIsNone(output.frame)
            self.assertEqual((camera.WIDTH, camera.HEIGHT, camera.FRAMERATE), (1280, 960, 30))
            self.assertIn('1280 x 960', camera.HTML_PAGE)
            raise KeyboardInterrupt
        srv.serve_forever.side_effect = serve
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / 'settings.yaml'
            config.write_text('camera:\n  framerate: 25\n'
                              '  resolution: {width: 1280, height: 720}\n'
                              f'  recording_dir: "{tmp}/recordings"\n')
            with patch.object(camera, 'Picamera2', return_value=picam), \
                    patch.object(camera, 'StreamingServer', return_value=srv):
                camera.main(['--config', str(config)])
        self.assertEqual(picam.configure.call_count, 2)
        self.assertEqual(picam.start_recording.call_count, 2)
        self.assertEqual(picam.stop_recording.call_count, 2)
        for call in picam.start_recording.call_args_list:
            self.assertIs(call.args[1].args[0], output_seen[0])
        srv.server_close.assert_called_once()

    def test_failed_live_configuration_restarts_previous_stream_without_closing_server(self):
        camera = import_camera_module(fresh=True)
        picam, srv = Mock(), Mock()
        picam.camera_controls = {}
        picam.capture_metadata.return_value = {}
        picam.configure.side_effect = [None, RuntimeError('bad mode'), None]
        def serve():
            with self.assertRaisesRegex(RuntimeError, 'bad mode'):
                camera.StreamingHandler.camera_settings.update({'framerate': 30})
            self.assertEqual((camera.WIDTH, camera.HEIGHT, camera.FRAMERATE), (1280, 720, 25))
            srv.server_close.assert_not_called()
            raise KeyboardInterrupt
        srv.serve_forever.side_effect = serve
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / 'settings.yaml'
            config.write_text('camera:\n  framerate: 25\n'
                              '  resolution: {width: 1280, height: 720}\n'
                              f'  recording_dir: "{tmp}/recordings"\n')
            original = config.read_bytes()
            with patch.object(camera, 'Picamera2', return_value=picam), \
                    patch.object(camera, 'StreamingServer', return_value=srv):
                camera.main(['--config', str(config)])
            self.assertEqual(config.read_bytes(), original)
        self.assertEqual(picam.configure.call_count, 3)
        self.assertEqual(picam.start_recording.call_count, 2)
        self.assertEqual(picam.stop_recording.call_count, 2)

    def test_partial_live_encoder_start_is_stopped_before_old_stream_is_restored(self):
        camera = import_camera_module(fresh=True)
        picam, srv = Mock(), Mock()
        picam.camera_controls = {}
        picam.capture_metadata.return_value = {}
        picam.start_recording.side_effect = [None, RuntimeError('encoder start failed'), None]
        def serve():
            with self.assertRaisesRegex(RuntimeError, 'encoder start failed'):
                camera.StreamingHandler.camera_settings.update({'framerate': 30})
            self.assertEqual((camera.WIDTH, camera.HEIGHT, camera.FRAMERATE), (1280, 720, 25))
            self.assertEqual(picam.stop_recording.call_count, 2)
            raise KeyboardInterrupt
        srv.serve_forever.side_effect = serve
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / 'settings.yaml'
            config.write_text('camera:\n  framerate: 25\n'
                              '  resolution: {width: 1280, height: 720}\n'
                              f'  recording_dir: "{tmp}/recordings"\n')
            with patch.object(camera, 'Picamera2', return_value=picam), \
                    patch.object(camera, 'StreamingServer', return_value=srv):
                camera.main(['--config', str(config)])
        self.assertEqual(picam.start_recording.call_count, 3)
        self.assertEqual(picam.stop_recording.call_count, 3)

    def test_imx219_full_width_mode_is_cached_before_streaming_and_reused(self):
        camera = import_camera_module(fresh=True)
        class SensorCamera:
            camera_properties = {'Model': 'imx219'}
            camera_controls = {}

            def __init__(self):
                self.mode_queries = 0
                self.streaming = False
                self.configurations = []

            @property
            def sensor_modes(self):
                assert not self.streaming, 'Sensor mode enumeration must never interrupt a live stream'
                self.mode_queries += 1
                return [{'size': (1920, 1080), 'bit_depth': 10},
                        {'size': (1640, 1232), 'bit_depth': 10}]

            def create_video_configuration(self, **kwargs):
                self.configurations.append(kwargs)
                return kwargs

            def configure(self, _configuration):
                assert not self.streaming

            def set_controls(self, _controls):
                pass

            def start_recording(self, _encoder, _output):
                self.streaming = True

            def stop_recording(self):
                self.streaming = False

            def capture_metadata(self):
                return {}

            def close(self):
                pass

        picam, srv = SensorCamera(), Mock()
        def serve():
            manager = camera.StreamingHandler.camera_settings
            manager.update({'resolution': {'width': 1280, 'height': 960}})
            manager.update({'resolution': {'width': 1920, 'height': 1080}})
            manager.update({'resolution': {'width': 1280, 'height': 720}})
            raise KeyboardInterrupt
        srv.serve_forever.side_effect = serve
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / 'settings.yaml'
            config.write_text('camera:\n  framerate: 25\n'
                              '  resolution: {width: 1280, height: 720}\n'
                              f'  recording_dir: "{tmp}/recordings"\n')
            with patch.object(camera, 'Picamera2', return_value=picam), \
                    patch.object(camera, 'StreamingServer', return_value=srv):
                camera.main(['--config', str(config)])
        self.assertEqual(picam.mode_queries, 1)
        selected = {'output_size': (1640, 1232), 'bit_depth': 10}
        self.assertEqual(picam.configurations[0]['sensor'], selected)
        self.assertEqual(picam.configurations[1]['sensor'], selected)
        self.assertNotIn('sensor', picam.configurations[2])
        self.assertEqual(picam.configurations[3]['sensor'], selected)

    def test_other_camera_keeps_automatic_sensor_selection(self):
        camera = import_camera_module(fresh=True)
        picam, srv = Mock(), Mock()
        picam.camera_properties = {'Model': 'ov5647'}
        picam.camera_controls = {}
        srv.serve_forever.side_effect = KeyboardInterrupt
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / 'settings.yaml'
            config.write_text('camera:\n  framerate: 25\n'
                              '  resolution: {width: 1280, height: 720}\n'
                              f'  recording_dir: "{tmp}/recordings"\n')
            with patch.object(camera, 'Picamera2', return_value=picam), \
                    patch.object(camera, 'StreamingServer', return_value=srv):
                camera.main(['--config', str(config)])
        picam.create_video_configuration.assert_called_once_with(
            main={'size': (1280, 720)}, controls={'FrameRate': 25},
        )

    def test_imx219_without_matching_sensor_mode_keeps_automatic_selection(self):
        camera = import_camera_module(fresh=True)
        picam, srv = Mock(), Mock()
        picam.camera_properties = {'Model': 'imx219'}
        picam.camera_controls = {}
        picam.sensor_modes = [{'size': (1640, 1232), 'bit_depth': 8}]
        srv.serve_forever.side_effect = KeyboardInterrupt
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / 'settings.yaml'
            config.write_text('camera:\n  framerate: 25\n'
                              '  resolution: {width: 1280, height: 720}\n'
                              f'  recording_dir: "{tmp}/recordings"\n')
            with patch.object(camera, 'Picamera2', return_value=picam), \
                    patch.object(camera, 'StreamingServer', return_value=srv):
                camera.main(['--config', str(config)])
        picam.create_video_configuration.assert_called_once_with(
            main={'size': (1280, 720)}, controls={'FrameRate': 25},
        )

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

    def test_recording_keeps_frames_after_start_request_thread_exits(self):
        class InProcessOutput:
            def __init__(self, filename):
                self.path = Path(filename)
                self.frames = bytearray()

            def start(self):
                pass

            def outputframe(self, frame):
                self.frames.extend(frame)

            def stop(self):
                if self.frames:
                    self.path.write_bytes(self.frames)

        class ThreadBoundFfmpegOutput(InProcessOutput):
            # FfmpegOutput sets PR_SET_PDEATHSIG: Linux kills its subprocess
            # when the thread that spawned it exits, including an HTTP worker.
            def start(self):
                self.owner = current_thread()

            def outputframe(self, frame):
                if self.owner.is_alive():
                    super().outputframe(frame)

        class Camera:
            def start_encoder(self, _encoder, output):
                self.output = output
                output.start()

            def stop_encoder(self, _encoder):
                self.output.stop()

        camera = self.camera
        with TemporaryDirectory() as tmp, \
                patch.object(camera, 'FfmpegOutput', ThreadBoundFfmpegOutput, create=True), \
                patch.object(camera, 'PyavOutput', InProcessOutput, create=True):
            picam = Camera()
            recorder = camera.CameraRecorder(picam, Path(tmp))
            handler = self.handler()
            handler.path = '/record/start'
            handler._authorized = lambda: True
            handler.recorder = recorder
            request_thread = Thread(target=handler.do_POST)
            request_thread.start()
            request_thread.join(timeout=1)
            self.assertFalse(request_thread.is_alive())
            result = json.loads(handler.wfile.getvalue())
            self.assertTrue(result['recording'])

            picam.output.outputframe(b'encoded-frame-1')
            picam.output.outputframe(b'encoded-frame-2')
            recorder.stop()

            path = Path(result['file'])
            self.assertTrue(path.is_file(), 'No recording survived the completed HTTP request')
            self.assertEqual(path.read_bytes(), b'encoded-frame-1encoded-frame-2')
            self.assertFalse(recorder.recording)
            self.assertIsNone(recorder.current_file)

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
