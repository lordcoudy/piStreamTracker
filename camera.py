#!/usr/bin/env python3
"""
Camera Streaming Server for piStreamTracker
Streams MJPEG video from Raspberry Pi Camera to network
"""

import io
import logging
import socketserver
from http import server
from pathlib import Path
from threading import Condition

import yaml
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput

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


class StreamingHandler(server.BaseHTTPRequestHandler):
    """HTTP request handler for camera stream."""

    output = None  # Set by main

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

        else:
            self.send_error(404)
            self.end_headers()


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

    try:
        server = StreamingServer((HOST, PORT), StreamingHandler)

        logger.info("=" * 50)
        logger.info("Camera Server Started!")
        logger.info("=" * 50)
        logger.info(f"Resolution:    {WIDTH}x{HEIGHT}")
        logger.info(f"Stream URL:    http://{CAMERA_IP}:{PORT}/stream")
        logger.info(f"Web Interface: http://{CAMERA_IP}:{PORT}/")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 50)

        server.serve_forever()

    except KeyboardInterrupt:
        logger.info("\nShutting down...")

    finally:
        picam2.stop_recording()
        logger.info("Camera server stopped")


if __name__ == '__main__':
    main()
