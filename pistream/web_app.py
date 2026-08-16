#!/usr/bin/env python3
"""
Web Interface for piStreamTracker
Simple Flask-based control panel for the tracking system
"""

import logging
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

import cv2
import numpy as np
from flask import (Flask, Response, jsonify, render_template, request,
                   send_from_directory)

from pistream.config import (apply_cli_overrides, configure_logging, load_config,
                             web_bind_host)
from pistream.lifecycle import TrackerLifecycle
from pistream.preview import accept_new_frame, camera_stream_url, preview_gate, preview_target_size
from pistream.recordings import is_recording_file, safe_recording_path
from pistream.track import HumanTracker

logger = logging.getLogger(__name__)

# =============================================================================
# Web Application
# =============================================================================

_PKG = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(_PKG / 'templates'),
            static_folder=str(_PKG / 'static'))

# Global tracker session
_lifecycle: Optional[TrackerLifecycle] = None
_frame_lock = Lock()
_latest_frame: Optional[np.ndarray] = None
_latest_seq: int = 0
_overlay_enabled: bool = True
_config: dict = {}


def _tracker() -> Optional[HumanTracker]:
    return _lifecycle.tracker if _lifecycle else None



@app.route('/')
def index():
    return render_template('index.html')


def _preview_settings():
    web = _config.get('web') or {}
    quality = int(web.get('preview_quality', 70))
    max_edge = int(web.get('preview_max_edge', 640))
    max_fps = float(web.get('preview_max_fps', 15) or 15)
    return quality, max_edge, max(1.0, max_fps)


def _mjpeg_part(jpeg: bytes) -> bytes:
    return (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')


def _placeholder_jpeg() -> bytes:
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(blank, 'No Signal', (70, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
    _, jpeg = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return jpeg.tobytes()


def _encode_preview(frame: np.ndarray, quality: int, max_edge: int) -> Optional[bytes]:
    h, w = frame.shape[:2]
    tw, th = preview_target_size(w, h, max_edge)
    if (tw, th) != (w, h):
        frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return jpeg.tobytes() if ok else None


def _generate_overlay_preview():
    quality, max_edge, max_fps = _preview_settings()
    min_interval = 1.0 / max_fps
    last_seq = -1
    last_emit = 0.0
    placeholder_sent = False
    while True:
        now = time.monotonic()
        with _frame_lock:
            seq = _latest_seq
            frame = _latest_frame
        if frame is None:
            if not placeholder_sent or (now - last_emit) >= 1.0:
                yield _mjpeg_part(_placeholder_jpeg())
                last_emit = now
                placeholder_sent = True
            time.sleep(min(0.05, min_interval))
            continue
        placeholder_sent = False
        if not preview_gate(last_seq, seq, now, last_emit, min_interval):
            time.sleep(0.005)
            continue
        jpeg = _encode_preview(frame, quality, max_edge)
        if jpeg is None:
            time.sleep(0.01)
            continue
        yield _mjpeg_part(jpeg)
        last_seq = seq
        last_emit = now


def _proxy_camera_stream():
    """Pass through camera MJPEG so the tracker does not re-encode."""
    url = camera_stream_url(_config)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'piStreamTracker/preview'})
        resp = urllib.request.urlopen(req, timeout=5)
    except Exception as exc:
        logger.warning(f"Camera proxy failed ({url}): {exc}")
        return None

    content_type = resp.headers.get(
        'Content-Type', 'multipart/x-mixed-replace; boundary=FRAME'
    )

    def generate():
        try:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                yield chunk
        except Exception:
            return
        finally:
            try:
                resp.close()
            except Exception:
                pass

    return Response(generate(), mimetype=content_type)


@app.route('/video_feed')
def video_feed():
    """Annotated preview, or raw camera MJPEG when overlay is off."""
    if not _overlay_enabled:
        proxied = _proxy_camera_stream()
        if proxied is not None:
            return proxied
    return Response(
        _generate_overlay_preview(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


@app.route('/api/status')
def api_status():
    """Get current tracker status."""
    trk = _tracker()
    if trk:
        detection = trk.current_detection
        shift_x = shift_y = None
        if detection:
            x, y, w, h = detection['bbox']
            cx = trk.capture.width // 2 if trk.capture else 640
            cy = trk.capture.height // 2 if trk.capture else 480
            shift_x = x + w // 2 - cx
            shift_y = y + h // 4 - cy

        return jsonify({
            'running': trk.running,
            'recording': trk.recording,
            'ev3_connected': trk.motors.connected,
            'fps': trk.fps,
            'detected': detection is not None,
            'shift_x': shift_x,
            'shift_y': shift_y,
            'zoom': trk.zoom_level,
            'horizon': trk.horizon_correction,
            'overlay': _overlay_enabled,
        })

    return jsonify({
        'running': False,
        'recording': False,
        'ev3_connected': False,
        'fps': 0,
        'detected': False,
        'shift_x': None,
        'shift_y': None,
        'zoom': 1.0,
        'horizon': False,
        'overlay': _overlay_enabled,
    })


@app.route('/api/start', methods=['POST'])
def api_start():
    """Start tracking. Returns only after the stream connects (or fails)."""
    if _lifecycle is None:
        return jsonify({'status': 'error', 'message': 'Web app not initialized'}), 500
    try:
        result = _lifecycle.start()
        if result['status'] == 'error':
            return jsonify(result), 503
        return jsonify(result)
    except Exception as e:
        logger.error(f"Start failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Stop tracking."""
    if _lifecycle is None:
        return jsonify({'status': 'ok'})
    return jsonify(_lifecycle.stop())


@app.route('/api/reset', methods=['POST'])
def api_reset():
    """Reset detection and move camera to home position."""
    trk = _tracker()
    if trk:
        trk.tracker.reset()
        trk.current_detection = None
        trk.motors.move_to_home()
    return jsonify({'status': 'ok'})


@app.route('/api/record', methods=['POST'])
def api_record():
    """Toggle recording."""
    trk = _tracker()
    if trk:
        if trk.recording:
            trk.stop_recording()
        else:
            trk.start_recording()
        return jsonify({'status': 'ok', 'recording': trk.recording})
    return jsonify({'status': 'error', 'message': 'Tracker not running'})


@app.route('/api/screenshot', methods=['POST'])
def api_screenshot():
    """Take screenshot."""
    global _latest_frame

    trk = _tracker()
    if trk and _latest_frame is not None:
        with _frame_lock:
            trk.screenshot(_latest_frame)
        return jsonify({'status': 'ok', 'path': str(trk.output_dir)})
    return jsonify({'status': 'error', 'message': 'No frame available'})


@app.route('/api/ev3', methods=['POST'])
def api_ev3():
    """Toggle EV3 connection."""
    trk = _tracker()
    if trk:
        data = request.get_json() or {}
        if data.get('enabled', False):
            trk.motors.connect()
        else:
            trk.motors.disconnect()
        return jsonify({'status': 'ok', 'connected': trk.motors.connected})
    return jsonify({'status': 'error', 'message': 'Tracker not running'})


@app.route('/api/settings', methods=['POST'])
def api_settings():
    """Update settings. Overlay can change before tracking starts."""
    global _overlay_enabled

    data = request.get_json() or {}
    if 'overlay' in data:
        _overlay_enabled = bool(data['overlay'])
        _config.setdefault('web', {})['overlay'] = _overlay_enabled

    trk = _tracker()
    if not trk:
        extra = set(data.keys()) - {'overlay'}
        if extra:
            return jsonify({'status': 'error', 'message': 'Tracker not running'})
        return jsonify({'status': 'ok'})

    if 'ev3_speed' in data:
        trk.motors.speed_factor = min(float(data['ev3_speed']), 2.0)
    if 'ev3_deadzone' in data:
        v = int(data['ev3_deadzone'])
        trk.motors.deadzone_x = v
        trk.motors.deadzone_y = v
    if 'confidence' in data:
        trk.detector.confidence = float(data['confidence'])
    if 'interval' in data:
        trk.detection_interval = int(data['interval'])
    if 'horizon' in data:
        trk.horizon_correction = bool(data['horizon'])

    return jsonify({'status': 'ok'})


@app.route('/api/motor_move', methods=['POST'])
def api_motor_move():
    """Manually move camera motors."""
    trk = _tracker()
    if not trk:
        return jsonify({'status': 'error', 'message': 'Tracker not running'})
    if not trk.motors.connected:
        return jsonify({'status': 'error', 'message': 'EV3 not connected'})

    data = request.get_json() or {}
    direction = data.get('direction', '')
    degrees = int(data.get('degrees', 10))

    pan = tilt = 0
    if direction == 'left':
        pan = -degrees
    elif direction == 'right':
        pan = degrees
    elif direction == 'up':
        tilt = -degrees
    elif direction == 'down':
        tilt = degrees

    trk.motors.manual_move(pan_degrees=pan, tilt_degrees=tilt)
    return jsonify({'status': 'ok', 'pan': pan, 'tilt': tilt})


@app.route('/api/zoom', methods=['POST'])
def api_zoom():
    """Control digital zoom."""
    trk = _tracker()
    if not trk:
        return jsonify({'status': 'error', 'message': 'Tracker not running'})

    data = request.get_json() or {}
    action = data.get('action', '')
    level = data.get('level')

    if level is not None:
        trk.zoom_level = max(1.0, min(float(level), 4.0))
    elif action == 'in':
        trk.zoom_level = min(trk.zoom_level + 0.25, 4.0)
    elif action == 'out':
        trk.zoom_level = max(trk.zoom_level - 0.25, 1.0)
    elif action == 'reset':
        trk.zoom_level = 1.0

    return jsonify({'status': 'ok', 'zoom': trk.zoom_level})


@app.route('/api/config')
def api_config():
    """Return camera resolution and detection defaults for the UI."""
    cam = _config.get('camera', {})
    res = cam.get('resolution', {})
    det = _config.get('tracker', {}).get('detection', {})
    web = _config.get('web') or {}
    return jsonify({
        'width': res.get('width', 1280),
        'height': res.get('height', 960),
        'confidence': det.get('confidence', 0.5),
        'interval': det.get('interval', 10),
        'overlay': web.get('overlay', True),
    })


def _recording_dir() -> Path:
    return Path(_config.get('tracker', {}).get('output_dir', 'recordings'))


@app.route('/api/recordings')
def api_recordings():
    """List recording files."""
    rec_path = _recording_dir()
    if not rec_path.exists():
        return jsonify({'files': []})

    files = []
    for f in sorted(rec_path.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not is_recording_file(f):
            continue
        stat = f.stat()
        size = stat.st_size
        if size < 1024:
            size_str = f"{size} B"
        elif size < 1024 * 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size / (1024 * 1024):.1f} MB"
        date_str = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
        files.append({
            'name': f.name,
            'size': size_str,
            'bytes': stat.st_size,
            'date': date_str,
            'type': f.suffix.lower().lstrip('.')
        })

    return jsonify({'files': files})


@app.route('/api/recordings/<path:filename>')
def api_recordings_download(filename):
    """Download or view a recording file."""
    rec_path = _recording_dir()
    try:
        file_path = safe_recording_path(rec_path, filename)
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Invalid path'}), 403

    if not file_path.exists():
        return jsonify({'status': 'error', 'message': 'File not found'}), 404

    as_attachment = file_path.suffix.lower() not in ('.jpg', '.png')
    return send_from_directory(str(rec_path.resolve()), file_path.name, as_attachment=as_attachment)


@app.route('/api/recordings/<path:filename>', methods=['DELETE'])
def api_recordings_delete(filename):
    """Delete a recording file."""
    rec_path = _recording_dir()
    try:
        file_path = safe_recording_path(rec_path, filename)
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Invalid path'}), 403

    if not file_path.exists():
        return jsonify({'status': 'error', 'message': 'File not found'}), 404

    try:
        file_path.unlink()
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# =============================================================================
# Tracker Loop
# =============================================================================

def _run_tracker_loop(trk: HumanTracker):
    """Background tracker loop. connect() has already succeeded."""
    global _latest_frame, _latest_seq

    logger.info("Web tracker loop started")
    prev_frame = None
    try:
        while trk.running:
            ret, frame = trk.capture.read()
            prev_frame, is_new = accept_new_frame(prev_frame, ret, frame)
            if not is_new:
                time.sleep(0.002)
                continue

            rec_frame = (
                frame.copy()
                if trk.recording and trk.recording_mode != 'camera'
                else None
            )
            annotated, _ = trk.process_frame(frame)

            with _frame_lock:
                _latest_frame = annotated
                _latest_seq += 1

            trk.update_fps()
            trk.write_frame(rec_frame)

    except Exception as e:
        logger.error(f"Tracker loop error: {e}")


# =============================================================================
# Entry Point
# =============================================================================

def run_web(config: dict, host: str = '0.0.0.0', port: int = 5000):
    """Run the web interface."""
    global _config, _lifecycle, _overlay_enabled
    _config = config
    _overlay_enabled = bool((config.get('web') or {}).get('overlay', True))
    _lifecycle = TrackerLifecycle(lambda: HumanTracker(_config), _run_tracker_loop)

    logger.info(f"Starting web interface at http://{host}:{port}")
    app.run(host=host, port=port, threaded=True, debug=False)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='piStreamTracker Web Interface')
    parser.add_argument('--config', default='config.yaml', help='Config file')
    parser.add_argument('--host', default=None, help='Bind address (default: tracker_ip from config)')
    parser.add_argument('--port', type=int, default=None, help='Port number')
    parser.add_argument('--url', help='Stream URL (overrides config)')
    parser.add_argument('--output-dir', help='Output directory')
    parser.add_argument('--detection-interval', type=int, help='Detection interval (frames)')
    parser.add_argument('--process-scale', type=float, help='Detection scale (0.2-1.0)')
    parser.add_argument('--confidence', type=float, help='Confidence threshold')
    parser.add_argument('--movenet-threads', type=int, help='Inference threads')
    parser.add_argument('--no-ev3', action='store_true', help='Disable EV3')
    parser.add_argument('--preset', help='Performance preset from config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    apply_cli_overrides(config, args)
    configure_logging(config)
    host = args.host or web_bind_host(config)
    port = args.port if args.port is not None else int(config.get('web', {}).get('port', 5000))
    run_web(config, host, port)


if __name__ == '__main__':
    main()
