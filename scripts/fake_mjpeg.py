#!/usr/bin/env python3
"""Minimal MJPEG test source (no picamera). GET /stream."""

import io
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Condition

import cv2
import numpy as np


class FrameBuf:
    def __init__(self):
        self.frame = None
        self.cond = Condition()

    def set(self, data: bytes):
        with self.cond:
            self.frame = data
            self.cond.notify_all()

    def wait(self) -> bytes:
        with self.cond:
            self.cond.wait(timeout=1.0)
            return self.frame


BUF = FrameBuf()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        if self.path != '/stream':
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
        self.end_headers()
        try:
            while True:
                frame = BUF.wait()
                if frame is None:
                    continue
                self.wfile.write(b'--FRAME\r\n')
                self.wfile.write(b'Content-Type: image/jpeg\r\n')
                self.wfile.write(f'Content-Length: {len(frame)}\r\n\r\n'.encode())
                self.wfile.write(frame)
                self.wfile.write(b'\r\n')
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return


def producer(width=320, height=240, fps=15):
    i = 0
    interval = 1.0 / fps
    while True:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (40, 40, 80)
        cv2.putText(img, f'FAKE {i}', (20, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 180), 2)
        ok, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            BUF.set(jpeg.tobytes())
        i += 1
        time.sleep(interval)


def main():
    import threading
    threading.Thread(target=producer, daemon=True).start()
    srv = ThreadingHTTPServer(('127.0.0.1', 8000), Handler)
    print('fake mjpeg http://127.0.0.1:8000/stream', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
