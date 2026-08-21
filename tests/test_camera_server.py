"""Camera HTTP/file helpers without requiring Raspberry Pi camera libraries."""

import importlib
import io
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


def import_camera_module():
    if 'camera' in sys.modules:
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
        return importlib.import_module('camera')


class CameraServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.camera = import_camera_module()

    def handler(self):
        handler = self.camera.StreamingHandler.__new__(self.camera.StreamingHandler)
        handler.wfile = io.BytesIO()
        handler.send_response = lambda _status: None
        handler.send_header = lambda _name, _value: None
        handler.end_headers = lambda: None
        return handler

    def test_stream_output_reports_bytes_written(self):
        output = self.camera.StreamingOutput()
        self.assertEqual(output.write(b'jpeg'), 4)
        self.assertEqual(output.frame, b'jpeg')

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
