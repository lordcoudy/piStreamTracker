#!/usr/bin/env python3
"""
Camera Streaming Server for piStreamTracker
Streams MJPEG video from Raspberry Pi Camera to network
"""

import io
import json
import logging
import socketserver
import time
from datetime import datetime
from http import server
from pathlib import Path
from threading import Condition, Lock

import yaml
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, JpegEncoder
from picamera2.outputs import FfmpegOutput, FileOutput

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from file."""
    config_file = Path(path)
    if config_file.exists():
        with open(config_file) as f:
            return yaml.safe_load(f) or {}
    return {}


# Load configuration
config = load_config()
camera_cfg = config.get('camera', {})
network_cfg = config.get('network', {})

# Settings with defaults
HOST = camera_cfg.get('host', '0.0.0.0')
PORT = camera_cfg.get('port', 8000)
WIDTH = camera_cfg.get('resolution', {}).get('width', 1280)
HEIGHT = camera_cfg.get('resolution', {}).get('height', 960)
CAMERA_IP = network_cfg.get('camera_ip', '192.168.100.1')
OUTPUT_DIR = Path(camera_cfg.get('recording_dir', 'recordings'))
OUTPUT_DIR.mkdir(exist_ok=True)


# HTML page template
HTML_PAGE = f"""<!DOCTYPE html>
<html>
<head>
    <title>Pi Camera Stream</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            text-align: center;
        }}
        h1 {{ color: #4ade80; margin-bottom: 10px; }}
        .info {{
            background: #16213e;
            padding: 15px;
            border-radius: 8px;
            display: inline-block;
            margin-bottom: 20px;
        }}
        .info p {{ margin: 5px 0; font-size: 14px; color: #94a3b8; }}
        img {{
            max-width: 100%;
            height: auto;
            border: 2px solid #4ade80;
            border-radius: 8px;
        }}
    </style>
</head>
<body>
    <h1>Pi Camera Stream</h1>
    <div class="info">
        <p>Resolution: {WIDTH} x {HEIGHT}</p>
        <p>Stream: http://{CAMERA_IP}:{PORT}/stream</p>
    </div>
    <br>
    <img src="stream" width="{WIDTH}" height="{HEIGHT}" alt="Camera Stream">
</body>
</html>
"""


class StreamingOutput(io.BufferedIOBase):
    """Thread-safe streaming output buffer."""

    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


class CameraRecorder:
    """Server-side H.264 recording using picamera2's hardware encoder.

    This offloads recording entirely from the tracker Pi.  The Pi 3B+
    camera hardware encodes H.264 at near-zero CPU cost.
    """

    def __init__(self, picam2: Picamera2, output_dir: Path):
        self._picam2 = picam2
        self._output_dir = output_dir
        self._encoder: H264Encoder | None = None
        self._output: FfmpegOutput | None = None
        self._lock = Lock()
        self.recording = False
        self.current_file: str | None = None

    def start(self, fps: int = 30, bitrate: int = 4_000_000) -> str:
        """Start H.264 recording.  Returns the filename."""
        with self._lock:
            if self.recording:
                return self.current_file

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self._output_dir / f"cam_rec_{ts}.mp4"
            self._encoder = H264Encoder(bitrate=bitrate)
            self._output = FfmpegOutput(str(path))
            self._picam2.start_encoder(self._encoder, self._output)
            self.recording = True
            self.current_file = str(path)
            logger.info(f"Camera recording started: {path}")
            return self.current_file

    def stop(self) -> None:
        """Stop recording."""
        with self._lock:
            if not self.recording:
                return
            try:
                self._picam2.stop_encoder(self._encoder)
            except Exception as e:
                logger.warning(f"Stop encoder error: {e}")
            self.recording = False
            logger.info(f"Camera recording stopped: {self.current_file}")
            self.current_file = None


class StreamingHandler(server.BaseHTTPRequestHandler):
    """HTTP request handler for camera stream and recording control."""

    output = None  # Set by main
    recorder = None  # Set by main

    def log_message(self, format, *args):
        logger.debug(format % args)

    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()

        elif self.path == '/index.html':
            content = HTML_PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)

        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()

            try:
                while True:
                    with self.output.condition:
                        self.output.condition.wait()
                        frame = self.output.frame

                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')

            except Exception as e:
                logger.debug(f'Client disconnected: {self.client_address} - {e}')

        elif self.path == '/record/status':
            self._json_response({
                'recording': self.recorder.recording if self.recorder else False,
                'file': self.recorder.current_file if self.recorder else None,
            })

        else:
            self.send_error(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/record/start':
            if not self.recorder:
                self._json_response({'error': 'recorder not available'}, 500)
                return
            fname = self.recorder.start()
            self._json_response({'recording': True, 'file': fname})

        elif self.path == '/record/stop':
            if not self.recorder:
                self._json_response({'error': 'recorder not available'}, 500)
                return
            self.recorder.stop()
            self._json_response({'recording': False})

        else:
            self.send_error(404)
            self.end_headers()

    def _json_response(self, data: dict, code: int = 200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    """Threaded HTTP server."""
    allow_reuse_address = True
    daemon_threads = True


def main():
    logger.info("Initializing Raspberry Pi Camera...")

    # Initialize camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(
        main={"size": (WIDTH, HEIGHT)}
    ))

    # Setup streaming
    output = StreamingOutput()
    StreamingHandler.output = output
    picam2.start_recording(JpegEncoder(), FileOutput(output))

    # Setup server-side recorder (hardware H.264)
    recorder = CameraRecorder(picam2, OUTPUT_DIR)
    StreamingHandler.recorder = recorder

    try:
        srv = StreamingServer((HOST, PORT), StreamingHandler)

        logger.info("=" * 50)
        logger.info("Camera Server Started!")
        logger.info("=" * 50)
        logger.info(f"Resolution:    {WIDTH}x{HEIGHT}")
        logger.info(f"Stream URL:    http://{CAMERA_IP}:{PORT}/stream")
        logger.info(f"Web Interface: http://{CAMERA_IP}:{PORT}/")
        logger.info(f"Record API:    POST /record/start  POST /record/stop")
        logger.info(f"Recordings:    {OUTPUT_DIR}/")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 50)

        srv.serve_forever()

    except KeyboardInterrupt:
        logger.info("\nShutting down...")

    finally:
        recorder.stop()
        picam2.stop_recording()
        logger.info("Camera server stopped")


if __name__ == '__main__':
    main()
