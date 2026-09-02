#!/usr/bin/env python3
"""
Web Interface for piStreamTracker
Simple Flask-based control panel for the tracking system
"""

import logging
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from threading import Lock
from typing import Optional
from urllib.parse import urlsplit

import cv2
import numpy as np
from flask import (
    Flask,
    Response,
    has_request_context,
    jsonify,
    render_template,
    request,
    send_from_directory,
    stream_with_context,
)

from pistream.config import apply_cli_overrides, configure_logging, load_config, web_bind_host
from pistream.lifecycle import TrackerLifecycle
from pistream.preview import accept_new_frame, camera_stream_url, preview_gate, preview_target_size
from pistream.recordings import (
    delete_remote_recording,
    fetch_remote_recordings,
    list_recording_files,
    open_remote_recording,
    safe_recording_path,
)
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


def _active_tracker() -> Optional[HumanTracker]:
    trk = _tracker()
    return trk if trk is not None and trk.running and trk.capture is not None else None


def _clear_latest_frame() -> None:
    global _latest_frame, _latest_seq
    with _frame_lock:
        _latest_frame = None
        _latest_seq += 1


def _json_object() -> Optional[dict]:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else None


def _finite(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a number')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a number') from exc
    if not math.isfinite(number):
        raise ValueError(f'{name} must be finite')
    return number


def _bounded(value, name: str, minimum: float, maximum: float) -> float:
    number = _finite(value, name)
    if not minimum <= number <= maximum:
        raise ValueError(f'{name} must be between {minimum:g} and {maximum:g}')
    return number


def _bounded_int(value, name: str, minimum: int, maximum: int) -> int:
    number = _bounded(value, name, minimum, maximum)
    if not number.is_integer():
        raise ValueError(f'{name} must be an integer')
    return int(number)


def _error(message: str, status: int = 400):
    return jsonify({'status': 'error', 'message': message}), status


@app.before_request
def reject_cross_origin_mutation():
    """Block browser-based cross-site requests to motor and recording controls."""
    if request.method in {'GET', 'HEAD', 'OPTIONS'}:
        return None
    origin = request.headers.get('Origin')
    if not origin:
        return None  # Preserve CLI and embedded-controller clients.
    actual = urlsplit(origin)
    expected = urlsplit(request.host_url)
    if (actual.scheme, actual.netloc) != (expected.scheme, expected.netloc):
        return _error('Cross-origin control requests are not allowed', 403)
    return None


@app.after_request
def add_security_headers(response):
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    response.headers.setdefault('X-Frame-Options', 'DENY')
    response.headers.setdefault('Referrer-Policy', 'no-referrer')
    return response



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

    return Response(generate(), content_type=content_type)


@app.route('/video_feed')
def video_feed():
    """Annotated preview, or raw camera MJPEG when overlay is off."""
    if not _overlay_enabled:
        proxied = _proxy_camera_stream()
        if proxied is not None:
            return proxied
    return Response(
        _generate_overlay_preview(),
        content_type='multipart/x-mixed-replace; boundary=frame',
    )


@app.route('/api/status')
def api_status():
    """Get current tracker status."""
    trk = _tracker()
    if trk:
        detection = trk.current_detection if trk.running else None
        shift_x = shift_y = None
        if detection:
            x, y, w, h = detection['bbox']
            cx = trk.capture.width // 2 if trk.capture else 640
            cy = trk.capture.height // 2 if trk.capture else 480
            shift_x = x + w // 2 - cx
            shift_y = y + h // 4 - cy

        stream_lost = bool(trk.capture and trk.capture.stream_lost)
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
            'stream_lost': stream_lost,
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
        'stream_lost': False,
    })


@app.route('/api/start', methods=['POST'])
def api_start():
    """Start tracking. Returns only after the stream connects (or fails)."""
    if _lifecycle is None:
        return jsonify({'status': 'error', 'message': 'Web app not initialized'}), 500
    try:
        _clear_latest_frame()
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
    result = _lifecycle.stop()
    _clear_latest_frame()
    return jsonify(result), (202 if result['status'] == 'stopping' else 200)


@app.route('/api/reset', methods=['POST'])
def api_reset():
    """Reset detection and move camera to home position."""
    trk = _active_tracker()
    if not trk:
        return _error('Tracker not running', 409)
    trk.reset_tracking(home=True)
    return jsonify({'status': 'ok'})


@app.route('/api/record', methods=['POST'])
def api_record():
    """Toggle recording."""
    trk = _active_tracker()
    if trk:
        was_recording = trk.recording
        if was_recording:
            if trk.stop_recording():
                return jsonify({'status': 'ok', 'recording': False})
            return _error('Camera did not confirm that recording stopped', 503)
        else:
            trk.start_recording()
        if trk.recording:
            return jsonify({'status': 'ok', 'recording': True})
        return _error('Could not start recording', 503)
    return _error('Tracker not running', 409)


@app.route('/api/screenshot', methods=['POST'])
def api_screenshot():
    """Take screenshot."""
    global _latest_frame

    trk = _active_tracker()
    if trk:
        with _frame_lock:
            frame = _latest_frame.copy() if _latest_frame is not None else None
        if frame is not None:
            try:
                path = trk.screenshot(frame)
            except OSError as exc:
                return _error(str(exc), 500)
            return jsonify({'status': 'ok', 'path': str(path)})
    return _error('No frame available', 409)


@app.route('/api/ev3', methods=['POST'])
def api_ev3():
    """Toggle EV3 connection."""
    trk = _active_tracker()
    if trk:
        data = _json_object()
        if data is None or not isinstance(data.get('enabled'), bool):
            return _error('enabled must be a boolean')
        if data.get('enabled', False):
            if not trk.motors.connect():
                return _error('Could not connect to EV3', 503)
        else:
            trk.motors.disconnect()
        return jsonify({'status': 'ok', 'connected': trk.motors.connected})
    return _error('Tracker not running', 409)


@app.route('/api/settings', methods=['POST'])
def api_settings():
    """Update settings. Overlay can change before tracking starts."""
    global _overlay_enabled

    data = _json_object()
    if data is None:
        return _error('Expected a JSON object')
    allowed = {'overlay', 'ev3_speed', 'ev3_deadzone', 'confidence', 'interval', 'horizon'}
    unknown = set(data) - allowed
    if unknown:
        return _error(f"Unknown setting: {sorted(unknown)[0]}")
    trk = _tracker()
    tracker_settings = set(data) - {'overlay'}
    if tracker_settings and not trk:
        return _error('Tracker not running', 409)

    values = {}
    try:
        if 'ev3_speed' in data:
            values['ev3_speed'] = _bounded(data['ev3_speed'], 'ev3_speed', 0.1, 2)
        if 'ev3_deadzone' in data:
            values['ev3_deadzone'] = _bounded_int(
                data['ev3_deadzone'], 'ev3_deadzone', 0, 10000
            )
        if 'confidence' in data:
            values['confidence'] = _bounded(data['confidence'], 'confidence', 0, 1)
        if 'interval' in data:
            values['interval'] = _bounded_int(data['interval'], 'interval', 1, 10000)
    except ValueError as exc:
        return _error(str(exc))
    if 'horizon' in data:
        if not isinstance(data['horizon'], bool):
            return _error('horizon must be a boolean')
        values['horizon'] = data['horizon']
    if 'overlay' in data:
        if not isinstance(data['overlay'], bool):
            return _error('overlay must be a boolean')
        values['overlay'] = data['overlay']

    if 'overlay' in values:
        _overlay_enabled = values['overlay']
        _config.setdefault('web', {})['overlay'] = _overlay_enabled
    if trk:
        if 'ev3_speed' in values:
            trk.motors.speed_factor = values['ev3_speed']
        if 'ev3_deadzone' in values:
            trk.motors.deadzone_x = values['ev3_deadzone']
            trk.motors.deadzone_y = values['ev3_deadzone']
        if 'confidence' in values:
            trk.detector.confidence = values['confidence']
        if 'interval' in values:
            trk.detection_interval = values['interval']
        if 'horizon' in values:
            trk.horizon_correction = values['horizon']
            if not trk.horizon_correction:
                trk.reset_horizon()

    return jsonify({'status': 'ok'})


@app.route('/api/motor_move', methods=['POST'])
def api_motor_move():
    """Manually move camera motors."""
    trk = _active_tracker()
    if not trk:
        return _error('Tracker not running', 409)
    if not trk.motors.connected:
        return _error('EV3 not connected', 409)

    data = _json_object()
    if data is None:
        return _error('Expected a JSON object')
    direction = data.get('direction')
    if direction not in {'left', 'right', 'up', 'down'}:
        return _error('direction must be left, right, up, or down')
    try:
        degrees = _bounded_int(data.get('degrees', 10), 'degrees', 1, 90)
    except ValueError as exc:
        return _error(str(exc))

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
    trk = _active_tracker()
    if not trk:
        return _error('Tracker not running', 409)

    data = _json_object()
    if data is None:
        return _error('Expected a JSON object')
    action = data.get('action', '')
    level = data.get('level')

    if level is not None:
        try:
            trk.zoom_level = _bounded(level, 'level', 1, 4)
        except ValueError as exc:
            return _error(str(exc))
    elif action == 'in':
        trk.zoom_level = min(trk.zoom_level + 0.25, 4.0)
    elif action == 'out':
        trk.zoom_level = max(trk.zoom_level - 0.25, 1.0)
    elif action == 'reset':
        trk.zoom_level = 1.0
    else:
        return _error('action must be in, out, or reset')

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


def _camera_recording_config() -> tuple[str, str]:
    net = _config.get('network') or {}
    cam = _config.get('camera') or {}
    base = f"http://{net.get('camera_ip', '127.0.0.1')}:{cam.get('port', 8000)}"
    return base, cam.get('token') or ''


def _camera_recording_mode() -> bool:
    return (_config.get('tracker') or {}).get('recording_mode', 'local') == 'camera'


def _recording_source() -> str:
    source = request.args.get('source', 'local') if has_request_context() else 'local'
    if source not in {'local', 'camera'}:
        raise ValueError('source must be local or camera')
    if source == 'camera' and not _camera_recording_mode():
        raise ValueError('camera recordings are not enabled')
    return source


def _camera_http_error(exc: urllib.error.HTTPError):
    messages = {
        401: 'Camera recording token was rejected',
        403: 'Camera rejected the recording path',
        404: 'Camera recording not found',
        409: 'Cannot delete an active camera recording',
    }
    return _error(messages.get(exc.code, 'Camera recording request failed'), exc.code)


@app.route('/api/recordings')
def api_recordings():
    """List local artifacts plus camera files when camera recording is configured."""
    trk = _active_tracker()
    active_path = trk.local_recording_path if trk is not None else None
    active_name = active_path.name if active_path is not None else None
    files = [
        {**entry, 'source': 'local', 'active': entry['name'] == active_name}
        for entry in list_recording_files(_recording_dir())
    ]
    warning = None
    if _camera_recording_mode():
        base, token = _camera_recording_config()
        try:
            remote = fetch_remote_recordings(base, token)
            files.extend(
                {**entry, 'source': 'camera'}
                for entry in remote
                if isinstance(entry, dict) and isinstance(entry.get('name'), str)
            )
        except Exception as exc:
            logger.warning(f"Camera recording list failed: {exc}")
            warning = 'Camera recordings are temporarily unavailable'
    def mtime(entry):
        try:
            return float(entry.get('mtime') or 0)
        except (TypeError, ValueError):
            return 0.0

    files.sort(key=mtime, reverse=True)
    return jsonify({'files': files, 'warning': warning})


@app.route('/api/recordings/<path:filename>')
def api_recordings_download(filename):
    """Download or view a recording file."""
    try:
        source = _recording_source()
    except ValueError as exc:
        return _error(str(exc))

    if source == 'camera':
        base, token = _camera_recording_config()
        try:
            remote = open_remote_recording(base, filename, token)
        except urllib.error.HTTPError as exc:
            return _camera_http_error(exc)
        except (OSError, urllib.error.URLError) as exc:
            logger.warning(f"Camera recording download failed: {exc}")
            return _error('Camera recording is unavailable', 502)

        headers = {}
        for name in ('Content-Length', 'Content-Disposition'):
            value = remote.headers.get(name)
            if value:
                headers[name] = value

        def generate():
            try:
                while chunk := remote.read(64 * 1024):
                    yield chunk
            finally:
                remote.close()

        return Response(
            stream_with_context(generate()),
            content_type=remote.headers.get('Content-Type', 'application/octet-stream'),
            headers=headers,
        )

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
    try:
        source = _recording_source()
    except ValueError as exc:
        return _error(str(exc))

    if source == 'camera':
        base, token = _camera_recording_config()
        try:
            delete_remote_recording(base, filename, token)
        except urllib.error.HTTPError as exc:
            return _camera_http_error(exc)
        except (OSError, urllib.error.URLError) as exc:
            logger.warning(f"Camera recording delete failed: {exc}")
            return _error('Camera recording is unavailable', 502)
        return jsonify({'status': 'ok'})

    rec_path = _recording_dir()
    try:
        file_path = safe_recording_path(rec_path, filename)
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Invalid path'}), 403

    if not file_path.exists():
        return jsonify({'status': 'error', 'message': 'File not found'}), 404

    trk = _active_tracker()
    active_path = trk.local_recording_path if trk is not None else None
    if active_path is not None and file_path == active_path.resolve():
        return _error('Cannot delete an active recording', 409)

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

            annotated, _ = trk.process_frame(frame)

            with _frame_lock:
                _latest_frame = annotated
                _latest_seq += 1

            trk.update_fps()
            trk.write_frame(trk.recording_frame())

    except Exception as e:
        logger.error(f"Tracker loop error: {e}")
    finally:
        _clear_latest_frame()


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

    try:
        config = load_config(args.config)
        apply_cli_overrides(config, args)
    except (OSError, ValueError) as exc:
        parser.error(f"configuration error: {exc}")
    configure_logging(config)
    host = args.host or web_bind_host(config)
    port = args.port if args.port is not None else int(config.get('web', {}).get('port', 5000))
    run_web(config, host, port)


if __name__ == '__main__':
    main()
