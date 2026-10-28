"""Local-only MJPEG power-cycle fixture for browser_preview_recovery.cjs."""

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flask import jsonify, request  # noqa: E402
from pistream import web_app  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=5099)
    args = parser.parse_args()
    lock = Lock()
    state = {'online': False, 'generation': 0, 'attempts': 0, 'active': 0}
    settings = {
        'resolution': {'width': 1280, 'height': 720}, 'framerate': 25,
        'awb_mode': 'auto', 'colour_gains': [1.5, 1.5],
        'exposure_mode': 'auto', 'exposure_time_us': 20000, 'analogue_gain': 1,
    }
    # Different height from the 320x240 No Signal image lets the browser verify
    # actual decoded recovery without a canvas or pixel polling.
    _, encoded = cv2.imencode('.jpg', np.full((180, 320, 3), 180, dtype=np.uint8))
    jpeg = encoded.tobytes()

    class Camera(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            if self.path == '/camera/settings':
                body = json.dumps({'status': 'ok', 'settings': settings}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            with lock:
                state['attempts'] += 1
                generation = state['generation']
                online = state['online']
                if online:
                    state['active'] += 1
            if not online:
                self.send_error(503, 'Camera is powered off')
                return
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=CAMERA')
            self.send_header('Connection', 'close')
            self.end_headers()
            header = b'--CAMERA\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(jpeg)).encode() + b'\r\n\r\n'
            try:
                while True:
                    with lock:
                        connected = state['online'] and state['generation'] == generation
                    if not connected:
                        self.wfile.write(header + jpeg[:len(jpeg) // 2])
                        self.wfile.flush()
                        return  # Simulate power loss in the middle of a JPEG.
                    self.wfile.write(header + jpeg + b'\r\n')
                    self.wfile.flush()
                    time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with lock:
                    state['active'] -= 1

    upstream = ThreadingHTTPServer(('127.0.0.1', 0), Camera)
    Thread(target=upstream.serve_forever, daemon=True).start()
    web_app._lifecycle = None
    web_app._overlay_enabled = True
    web_app._config = {
        'network': {'camera_ip': '127.0.0.1'},
        'camera': {'port': upstream.server_port, **settings},
    }

    @web_app.app.route('/__fixture/camera', methods=['GET', 'POST'])
    def camera_state():
        with lock:
            if request.method == 'POST':
                state['online'] = bool(request.get_json()['online'])
                state['generation'] += 1
            return jsonify(state)

    try:
        web_app.app.run(host='127.0.0.1', port=args.port, threaded=True, debug=False)
    finally:
        upstream.shutdown()
        upstream.server_close()


if __name__ == '__main__':
    main()
