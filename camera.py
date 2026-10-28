#!/usr/bin/env python3
"""
Camera Streaming Server for piStreamTracker
Streams MJPEG video from Raspberry Pi Camera to network
"""

import argparse
import signal
import io
import json
import logging
import mimetypes
from datetime import datetime
from http import server
from pathlib import Path
from threading import BoundedSemaphore, Condition, Lock
from urllib.parse import quote, unquote, urlsplit

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, JpegEncoder
from picamera2.outputs import FfmpegOutput, FileOutput

from pistream.camera_auth import extract_bearer, token_ok
from pistream.config import camera_bind_host, load_config
from pistream.recordings import is_recording_file, list_recording_files, safe_recording_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Initialized by main before the camera or HTTP server starts.
HOST = PORT = WIDTH = HEIGHT = FRAMERATE = JPEG_QUALITY = CAMERA_IP = None
OUTPUT_DIR = None
TUNING_FILE = None
CAMERA_TOKEN = ''
STREAM_LIMIT = None
HTML_PAGE = ''


def configure_camera(config_path: str) -> None:
    """Load startup settings before creating their dependent resources."""
    global HOST, PORT, WIDTH, HEIGHT, FRAMERATE, JPEG_QUALITY, CAMERA_IP
    global OUTPUT_DIR, TUNING_FILE, CAMERA_TOKEN, STREAM_LIMIT, HTML_PAGE
    config = load_config(config_path)
    camera_cfg = config['camera']
    HOST = camera_bind_host(config)
    PORT = camera_cfg['port']
    WIDTH = camera_cfg['resolution']['width']
    HEIGHT = camera_cfg['resolution']['height']
    FRAMERATE = camera_cfg['framerate']
    JPEG_QUALITY = camera_cfg['jpeg_quality']
    tuning_file = camera_cfg.get('tuning_file')
    TUNING_FILE = Path(tuning_file) if tuning_file else None
    if TUNING_FILE is not None and not TUNING_FILE.is_absolute():
        TUNING_FILE = Path(__file__).resolve().parent / TUNING_FILE
    CAMERA_IP = config['network']['camera_ip']
    OUTPUT_DIR = Path(camera_cfg['recording_dir'])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CAMERA_TOKEN = camera_cfg.get('token') or ''
    STREAM_LIMIT = BoundedSemaphore(camera_cfg['max_stream_clients'])
    HTML_PAGE = render_html_page()


def render_html_page() -> str:
    return f"""<!DOCTYPE html>
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
        return len(buf)


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

    def start(self, bitrate: int = 4_000_000) -> str:
        """Start H.264 recording.  Returns the filename."""
        with self._lock:
            if self.recording:
                return self.current_file

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
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
                raise RuntimeError(f"camera encoder did not stop: {e}") from e
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
        path = unquote(urlsplit(self.path).path)
        if path.startswith('/record/') and not self._authorized():
            return
        if path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()

        elif path == '/index.html':
            content = HTML_PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)

        elif path == '/stream':
            if not STREAM_LIMIT.acquire(blocking=False):
                logger.warning('Stream client cap reached')
                self.send_error(503, 'Too many stream clients')
                return
            try:
                self.send_response(200)
                self.send_header('Age', 0)
                self.send_header('Cache-Control', 'no-cache, private')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
                self.send_header('Connection', 'keep-alive')
                self.end_headers()

                while True:
                    with self.output.condition:
                        if not self.output.condition.wait(timeout=2.0):
                            continue  # No new frame within timeout, retry
                        frame = self.output.frame

                    if frame is None:
                        continue

                    self.wfile.write(b'--FRAME\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(frame)}\r\n'.encode())
                    self.wfile.write(b'\r\n')
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
                    self.wfile.flush()

            except Exception as e:
                logger.debug(f'Client disconnected: {self.client_address} - {e}')
            finally:
                STREAM_LIMIT.release()

        elif path == '/record/list':
            active_name = None
            if self.recorder and self.recorder.recording and self.recorder.current_file:
                active_name = Path(self.recorder.current_file).name
            files = [
                {**entry, 'active': entry['name'] == active_name}
                for entry in list_recording_files(OUTPUT_DIR)
            ]
            self._json_response({'files': files})

        elif path == '/record/status':
            self._json_response({
                'recording': self.recorder.recording if self.recorder else False,
                'file': self.recorder.current_file if self.recorder else None,
            })

        elif path.startswith('/record/files/'):
            self._send_recording(path.removeprefix('/record/files/'))

        elif path.startswith('/record/'):
            self._json_response({'error': 'not found'}, 404)

        else:
            self.send_error(404)

    def _authorized(self) -> bool:
        provided = extract_bearer(self.headers.get('Authorization'))
        if token_ok(provided, CAMERA_TOKEN or None):
            return True
        self._json_response({'error': 'unauthorized'}, 401)
        return False

    def do_POST(self):
        path = unquote(urlsplit(self.path).path)
        if path.startswith('/record/') and not self._authorized():
            return
        if path == '/record/start':
            if not self.recorder:
                self._json_response({'error': 'recorder not available'}, 500)
                return
            try:
                fname = self.recorder.start()
                self._json_response({'recording': True, 'file': fname})
            except Exception as exc:
                logger.exception('Could not start camera recording')
                self._json_response({'error': str(exc)}, 500)

        elif path == '/record/stop':
            if not self.recorder:
                self._json_response({'error': 'recorder not available'}, 500)
                return
            try:
                self.recorder.stop()
                self._json_response({'recording': False})
            except Exception as exc:
                logger.exception('Could not stop camera recording')
                self._json_response({'error': str(exc)}, 500)

        elif path.startswith('/record/'):
            self._json_response({'error': 'not found'}, 404)

        else:
            self.send_error(404)

    def do_DELETE(self):
        path = unquote(urlsplit(self.path).path)
        if not path.startswith('/record/'):
            self.send_error(404)
            return
        if not self._authorized():
            return
        if not path.startswith('/record/files/'):
            self._json_response({'error': 'not found'}, 404)
            return
        try:
            file_path = self._recording_path(path.removeprefix('/record/files/'))
            if (
                self.recorder
                and self.recorder.recording
                and self.recorder.current_file
                and file_path == Path(self.recorder.current_file).resolve()
            ):
                self._json_response({'error': 'cannot delete an active recording'}, 409)
                return
            file_path.unlink()
        except ValueError:
            self._json_response({'error': 'invalid path'}, 403)
        except FileNotFoundError:
            self._json_response({'error': 'file not found'}, 404)
        except OSError as exc:
            self._json_response({'error': str(exc)}, 500)
        else:
            self._json_response({'status': 'ok'})

    def _recording_path(self, filename: str) -> Path:
        file_path = safe_recording_path(OUTPUT_DIR, filename)
        if not is_recording_file(file_path):
            raise FileNotFoundError(filename)
        return file_path

    def _send_recording(self, filename: str) -> None:
        try:
            file_path = self._recording_path(filename)
            stat = file_path.stat()
            file_handle = file_path.open('rb')
        except ValueError:
            self._json_response({'error': 'invalid path'}, 403)
            return
        except (FileNotFoundError, OSError):
            self._json_response({'error': 'file not found'}, 404)
            return

        disposition = 'inline' if file_path.suffix.lower() in {'.jpg', '.png'} else 'attachment'
        encoded_name = quote(file_path.name)
        try:
            self.send_response(200)
            self.send_header(
                'Content-Type', mimetypes.guess_type(file_path.name)[0] or 'application/octet-stream'
            )
            self.send_header('Content-Length', stat.st_size)
            self.send_header(
                'Content-Disposition', f"{disposition}; filename*=UTF-8''{encoded_name}"
            )
            self.end_headers()
            remaining = stat.st_size
            while remaining and (chunk := file_handle.read(min(64 * 1024, remaining))):
                self.wfile.write(chunk)
                remaining -= len(chunk)
        except (BrokenPipeError, ConnectionError):
            logger.debug('Recording download client disconnected')
        finally:
            file_handle.close()

    def _json_response(self, data: dict, code: int = 200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)


def camera_busy_hint(ip: str, port: int) -> str:
    """How to recover when libcamera is already held (usually pitracker.service)."""
    return (
        f'Camera device is busy (another process holds libcamera). '
        f'After ./setup.sh --camera, pitracker.service already streams at '
        f'http://{ip}:{port}/stream. Check: sudo systemctl status pitracker. '
        f'For standalone also check pistream-camera.service. '
        f'To run camera.py yourself: sudo systemctl stop pitracker pistream-camera'
    )


class StreamingServer(server.ThreadingHTTPServer):
    """Threaded HTTP server with tuning for low-latency streaming."""
    request_queue_size = 8
    daemon_threads = True

    def server_bind(self):
        """Override to set socket options for streaming performance."""
        import socket
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        super().server_bind()


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Raspberry Pi camera streaming server')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    args = parser.parse_args(argv)
    try:
        configure_camera(args.config)
    except (OSError, ValueError) as exc:
        raise SystemExit(f'Configuration error: {exc}') from exc
    logger.info("Initializing Raspberry Pi Camera...")

    def terminate(_signum, _frame):
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, terminate)
    picam2 = recorder = srv = None
    streaming = False
    try:
        camera_options = {}
        if TUNING_FILE is not None:
            logger.info('Camera tuning: %s', TUNING_FILE)
            camera_options['tuning'] = Picamera2.load_tuning_file(str(TUNING_FILE))
        try:
            picam2 = Picamera2(**camera_options)
        except RuntimeError:
            logger.error(camera_busy_hint(CAMERA_IP, PORT))
            raise
        video_config = picam2.create_video_configuration(
            main={"size": (WIDTH, HEIGHT)},
            controls={"FrameRate": FRAMERATE},
        )
        picam2.configure(video_config)

        output = StreamingOutput()
        StreamingHandler.output = output
        jpeg_encoder = JpegEncoder(q=JPEG_QUALITY)
        picam2.start_recording(jpeg_encoder, FileOutput(output))
        streaming = True
        recorder = CameraRecorder(picam2, OUTPUT_DIR)
        StreamingHandler.recorder = recorder
        srv = StreamingServer((HOST, PORT), StreamingHandler)

        logger.info("=" * 50)
        logger.info("Camera Server Started!")
        logger.info("=" * 50)
        logger.info(f"Resolution:    {WIDTH}x{HEIGHT}")
        logger.info(f"Stream URL:    http://{CAMERA_IP}:{PORT}/stream")
        logger.info(f"Web Interface: http://{CAMERA_IP}:{PORT}/")
        logger.info("Record API:    POST /record/start  POST /record/stop")
        logger.info(f"Recordings:    {OUTPUT_DIR}/")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 50)

        srv.serve_forever()

    except KeyboardInterrupt:
        logger.info("\nShutting down...")

    finally:
        try:
            if srv is not None:
                srv.server_close()
            if recorder is not None:
                recorder.stop()
        except Exception:
            logger.exception("Camera recorder did not stop cleanly")
        finally:
            try:
                if picam2 is not None:
                    try:
                        if streaming:
                            picam2.stop_recording()
                    finally:
                        picam2.close()
            finally:
                signal.signal(signal.SIGTERM, previous_handler)
        logger.info("Camera server stopped")


if __name__ == '__main__':
    main()
