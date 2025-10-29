import io
import logging
import socketserver
from http import server
from threading import Condition

import yaml
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

camera_config = config['camera_server']

# HTML page with video stream
PAGE = f"""\
<html>
<head>
    <title>Raspberry Pi Camera Stream</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background-color: #222;
            color: white;
            font-family: Arial, sans-serif;
            text-align: center;
        }}
        h1 {{
            color: #4CAF50;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 2px solid #4CAF50;
        }}
        .info {{
            margin: 20px;
            padding: 10px;
            background-color: #333;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <h1>Raspberry Pi Camera Stream</h1>
    <div class="info">
        <p>Resolution: {camera_config['resolution']['width']} x {camera_config['resolution']['height']}</p>
        <p>Server IP: {camera_config['host']}:{camera_config['port']}</p>
    </div>
    <img src="stream" width="{camera_config['resolution']['width']}" height="{camera_config['resolution']['height']}" />
</body>
</html>
"""


class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
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
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info("Initializing Raspberry Pi Camera...")
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (camera_config['resolution']['width'], camera_config['resolution']['height'])}))
    output = StreamingOutput()
    picam2.start_recording(JpegEncoder(), FileOutput(output))

    try:
        # Bind to all interfaces (0.0.0.0) so it's accessible from Ethernet
        address = (camera_config['host'], camera_config['port'])
        server = StreamingServer(address, StreamingHandler)

        logging.info("=" * 60)
        logging.info("Camera Server Started!")
        logging.info("=" * 60)
        logging.info(f"Stream URL: http://{camera_config['host']}:{camera_config['port']}/stream")
        logging.info(f"Web Interface: http://{camera_config['host']}:{camera_config['port']}/")
        logging.info("Press Ctrl+C to stop")
        logging.info("=" * 60)

        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("\nShutting down camera server...")
    finally:
        picam2.stop_recording()
        logging.info("Camera server stopped.")
