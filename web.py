#!/usr/bin/env python3
"""
Web Interface for piStreamTracker
Simple Flask-based control panel for the tracking system
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread
from typing import Optional

import cv2
import numpy as np
from flask import (Flask, Response, jsonify, render_template_string, request,
                   send_from_directory)

# Import tracker components
from tracker import HumanTracker, load_config

logger = logging.getLogger(__name__)

# =============================================================================
# Web Application
# =============================================================================

app = Flask(__name__)

# Global tracker instance
_tracker: Optional[HumanTracker] = None
_tracker_thread: Optional[Thread] = None
_frame_lock = Lock()
_latest_frame: Optional[np.ndarray] = None
_config: dict = {}


# =============================================================================
# HTML Template
# =============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>piStreamTracker</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 0;
            border-bottom: 1px solid #333;
            margin-bottom: 20px;
        }
        h1 {
            font-size: 1.5rem;
            color: #4ade80;
        }
        .status {
            display: flex;
            gap: 15px;
        }
        .status-item {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.85rem;
        }
        .dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }
        .dot.green { background: #4ade80; }
        .dot.red { background: #f87171; }
        .dot.yellow { background: #fbbf24; }

        .main-grid {
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 20px;
        }
        @media (max-width: 900px) {
            .main-grid { grid-template-columns: 1fr; }
        }

        .video-container {
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            position: relative;
            max-width: var(--stream-width, 1280px);
            aspect-ratio: var(--stream-aspect, 4/3);
        }
        .video-container img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }
        .video-overlay {
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(0,0,0,0.7);
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 0.8rem;
        }

        .controls {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .panel {
            background: #16213e;
            border-radius: 8px;
            padding: 15px;
        }
        .panel h3 {
            font-size: 0.9rem;
            color: #94a3b8;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .btn-group {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        button {
            background: #3b82f6;
            color: white;
            border: none;
            padding: 10px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: background 0.2s;
        }
        button:hover { background: #2563eb; }
        button:disabled { background: #475569; cursor: not-allowed; }
        button.danger { background: #ef4444; }
        button.danger:hover { background: #dc2626; }
        button.success { background: #22c55e; }
        button.success:hover { background: #16a34a; }
        button.secondary { background: #475569; }
        button.secondary:hover { background: #64748b; }

        .slider-group {
            margin-bottom: 12px;
        }
        .slider-group label {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            margin-bottom: 6px;
            color: #cbd5e1;
        }
        input[type="range"] {
            width: 100%;
            height: 6px;
            background: #334155;
            border-radius: 3px;
            outline: none;
            -webkit-appearance: none;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            background: #3b82f6;
            border-radius: 50%;
            cursor: pointer;
        }

        .info-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }
        .info-item {
            background: #1e293b;
            padding: 10px;
            border-radius: 4px;
            text-align: center;
        }
        .info-item .value {
            font-size: 1.2rem;
            font-weight: bold;
            color: #4ade80;
        }
        .info-item .label {
            font-size: 0.75rem;
            color: #94a3b8;
        }

        .toggle-switch {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 0;
        }
        .toggle-switch input {
            display: none;
        }
        .toggle-switch .slider {
            width: 44px;
            height: 24px;
            background: #475569;
            border-radius: 12px;
            position: relative;
            cursor: pointer;
            transition: background 0.2s;
        }
        .toggle-switch .slider:before {
            content: '';
            position: absolute;
            width: 18px;
            height: 18px;
            background: white;
            border-radius: 50%;
            top: 3px;
            left: 3px;
            transition: transform 0.2s;
        }
        .toggle-switch input:checked + .slider {
            background: #22c55e;
        }
        .toggle-switch input:checked + .slider:before {
            transform: translateX(20px);
        }

        .log-output {
            background: #0f172a;
            border-radius: 4px;
            padding: 10px;
            height: 120px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 0.75rem;
            color: #94a3b8;
        }
        .log-output .log-line {
            margin-bottom: 2px;
        }
        .log-output .log-line.error { color: #f87171; }
        .log-output .log-line.success { color: #4ade80; }

        /* Recordings panel */
        .recordings-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }
        .recordings-header h3 { margin-bottom: 0; }
        .recordings-toggle {
            font-size: 0.8rem;
            color: #94a3b8;
            transition: transform 0.2s;
        }
        .recordings-toggle.open { transform: rotate(180deg); }
        .recordings-body {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }
        .recordings-body.open {
            max-height: 600px;
            overflow-y: auto;
        }
        .rec-list {
            margin-top: 12px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .rec-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #1e293b;
            padding: 8px 10px;
            border-radius: 4px;
            font-size: 0.8rem;
            gap: 8px;
        }
        .rec-item .rec-info {
            flex: 1;
            min-width: 0;
            overflow: hidden;
        }
        .rec-item .rec-name {
            color: #e2e8f0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .rec-item .rec-meta {
            color: #64748b;
            font-size: 0.7rem;
        }
        .rec-item .rec-thumb {
            width: 48px;
            height: 36px;
            object-fit: cover;
            border-radius: 3px;
            flex-shrink: 0;
        }
        .rec-actions {
            display: flex;
            gap: 4px;
            flex-shrink: 0;
        }
        .rec-actions button {
            padding: 4px 8px;
            font-size: 0.7rem;
        }
        .rec-empty {
            color: #64748b;
            font-size: 0.8rem;
            text-align: center;
            padding: 15px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>piStreamTracker</h1>
            <div class="status">
                <div class="status-item">
                    <span class="dot" id="status-tracking"></span>
                    <span>Tracking</span>
                </div>
                <div class="status-item">
                    <span class="dot" id="status-ev3"></span>
                    <span>EV3</span>
                </div>
                <div class="status-item">
                    <span class="dot" id="status-recording"></span>
                    <span>Recording</span>
                </div>
            </div>
        </header>

        <div class="main-grid">
            <div class="video-container">
                <img id="video-feed" src="/video_feed" alt="Video Feed">
                <div class="video-overlay" id="fps-display">-- FPS</div>
            </div>

            <div class="controls">
                <div class="panel">
                    <h3>Tracking Control</h3>
                    <div class="btn-group">
                        <button id="btn-start" class="success" onclick="startTracking()">Start</button>
                        <button id="btn-stop" class="danger" onclick="stopTracking()" disabled>Stop</button>
                        <button id="btn-reset" class="secondary" onclick="resetDetection()">Reset</button>
                    </div>
                </div>

                <div class="panel">
                    <h3>Recording</h3>
                    <div class="btn-group">
                        <button id="btn-record" onclick="toggleRecording()">Record</button>
                        <button onclick="takeScreenshot()">Screenshot</button>
                    </div>
                </div>

                <div class="panel">
                    <h3>EV3 Motor Control</h3>
                    <label class="toggle-switch">
                        <span>Motors Enabled</span>
                        <input type="checkbox" id="ev3-toggle" onchange="toggleEV3()">
                        <span class="slider"></span>
                    </label>
                    <div class="slider-group">
                        <label>
                            <span>Speed Factor</span>
                            <span id="speed-value">1.0</span>
                        </label>
                        <input type="range" id="speed-slider" min="0.1" max="2.0" step="0.1" value="1.0"
                               onchange="updateSetting('ev3_speed', this.value)">
                    </div>
                    <div class="slider-group">
                        <label>
                            <span>Deadzone</span>
                            <span id="deadzone-value">90</span>
                        </label>
                        <input type="range" id="deadzone-slider" min="20" max="200" step="10" value="90"
                               onchange="updateSetting('ev3_deadzone', this.value)">
                    </div>
                </div>

                <div class="panel">
                    <h3>Detection Settings</h3>
                    <div class="slider-group">
                        <label>
                            <span>Confidence</span>
                            <span id="conf-value">0.5</span>
                        </label>
                        <input type="range" id="conf-slider" min="0.2" max="0.9" step="0.05" value="0.5"
                               onchange="updateSetting('confidence', this.value)">
                    </div>
                    <div class="slider-group">
                        <label>
                            <span>Detection Interval</span>
                            <span id="interval-value">8</span>
                        </label>
                        <input type="range" id="interval-slider" min="2" max="20" step="1" value="8"
                               onchange="updateSetting('interval', this.value)">
                    </div>
                </div>

                <div class="panel">
                    <h3>Status</h3>
                    <div class="info-grid">
                        <div class="info-item">
                            <div class="value" id="info-fps">--</div>
                            <div class="label">FPS</div>
                        </div>
                        <div class="info-item">
                            <div class="value" id="info-detection">--</div>
                            <div class="label">Detection</div>
                        </div>
                        <div class="info-item">
                            <div class="value" id="info-shift-x">--</div>
                            <div class="label">Shift X</div>
                        </div>
                        <div class="info-item">
                            <div class="value" id="info-shift-y">--</div>
                            <div class="label">Shift Y</div>
                        </div>
                    </div>
                </div>

                <div class="panel">
                    <h3>Log</h3>
                    <div class="log-output" id="log-output"></div>
                </div>

                <div class="panel">
                    <div class="recordings-header" onclick="toggleRecordingsPanel()">
                        <h3>Recordings</h3>
                        <span class="recordings-toggle" id="rec-toggle">&#9660;</span>
                    </div>
                    <div class="recordings-body" id="rec-body">
                        <div style="margin-top:10px">
                            <button class="secondary" onclick="loadRecordings()" style="padding:6px 12px;font-size:0.8rem">Refresh</button>
                        </div>
                        <div class="rec-list" id="rec-list">
                            <div class="rec-empty">Click Refresh to load</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let isTracking = false;
        let isRecording = false;

        function log(message, type = 'info') {
            const logEl = document.getElementById('log-output');
            const line = document.createElement('div');
            line.className = 'log-line ' + type;
            line.textContent = new Date().toLocaleTimeString() + ' ' + message;
            logEl.appendChild(line);
            logEl.scrollTop = logEl.scrollHeight;
        }

        async function api(endpoint, method = 'GET', data = null) {
            try {
                const opts = { method };
                if (data) {
                    opts.headers = { 'Content-Type': 'application/json' };
                    opts.body = JSON.stringify(data);
                }
                const res = await fetch('/api/' + endpoint, opts);
                return await res.json();
            } catch (e) {
                log('API error: ' + e.message, 'error');
                return null;
            }
        }

        async function startTracking() {
            log('Starting tracker...');
            const res = await api('start', 'POST');
            if (res && res.status === 'ok') {
                isTracking = true;
                updateUI();
                log('Tracking started', 'success');
            }
        }

        async function stopTracking() {
            log('Stopping tracker...');
            const res = await api('stop', 'POST');
            if (res && res.status === 'ok') {
                isTracking = false;
                updateUI();
                log('Tracking stopped', 'success');
            }
        }

        async function resetDetection() {
            const res = await api('reset', 'POST');
            if (res) log('Detection reset', 'success');
        }

        async function toggleRecording() {
            const res = await api('record', 'POST');
            if (res) {
                isRecording = res.recording;
                updateUI();
                log(isRecording ? 'Recording started' : 'Recording stopped', 'success');
            }
        }

        async function takeScreenshot() {
            const res = await api('screenshot', 'POST');
            if (res && res.path) log('Screenshot: ' + res.path, 'success');
        }

        async function toggleEV3() {
            const enabled = document.getElementById('ev3-toggle').checked;
            const res = await api('ev3', 'POST', { enabled });
            if (res) log('EV3 ' + (enabled ? 'connected' : 'disconnected'), 'success');
        }

        async function updateSetting(key, value) {
            // Update display
            if (key === 'ev3_speed') document.getElementById('speed-value').textContent = value;
            if (key === 'ev3_deadzone') document.getElementById('deadzone-value').textContent = value;
            if (key === 'confidence') document.getElementById('conf-value').textContent = value;
            if (key === 'interval') document.getElementById('interval-value').textContent = value;

            await api('settings', 'POST', { [key]: parseFloat(value) });
        }

        function updateUI() {
            document.getElementById('btn-start').disabled = isTracking;
            document.getElementById('btn-stop').disabled = !isTracking;
            document.getElementById('btn-record').textContent = isRecording ? 'Stop Rec' : 'Record';
            document.getElementById('btn-record').className = isRecording ? 'danger' : '';
            document.getElementById('status-tracking').className = 'dot ' + (isTracking ? 'green' : 'red');
            document.getElementById('status-recording').className = 'dot ' + (isRecording ? 'red' : 'yellow');
        }

        async function updateStatus() {
            const res = await api('status');
            if (res) {
                isTracking = res.running;
                isRecording = res.recording;
                document.getElementById('status-ev3').className = 'dot ' + (res.ev3_connected ? 'green' : 'red');
                document.getElementById('ev3-toggle').checked = res.ev3_connected;
                document.getElementById('info-fps').textContent = res.fps.toFixed(1);
                document.getElementById('info-detection').textContent = res.detected ? 'Yes' : 'No';
                if (res.shift_x !== null) {
                    document.getElementById('info-shift-x').textContent = res.shift_x;
                    document.getElementById('info-shift-y').textContent = res.shift_y;
                }
                updateUI();
            }
        }

        // --- Recordings ---
        function toggleRecordingsPanel() {
            const body = document.getElementById('rec-body');
            const toggle = document.getElementById('rec-toggle');
            body.classList.toggle('open');
            toggle.classList.toggle('open');
            if (body.classList.contains('open') && document.getElementById('rec-list').querySelector('.rec-empty')) {
                loadRecordings();
            }
        }

        async function loadRecordings() {
            const res = await api('recordings');
            const list = document.getElementById('rec-list');
            if (!res || !res.files || res.files.length === 0) {
                list.innerHTML = '<div class="rec-empty">No recordings found</div>';
                return;
            }
            list.innerHTML = '';
            res.files.forEach(f => {
                const item = document.createElement('div');
                item.className = 'rec-item';
                const isImg = f.name.toLowerCase().endsWith('.jpg') || f.name.toLowerCase().endsWith('.png');
                const thumbHtml = isImg
                    ? '<img class="rec-thumb" src="/api/recordings/' + encodeURIComponent(f.name) + '" alt="thumb">'
                    : '';
                item.innerHTML = thumbHtml +
                    '<div class="rec-info">' +
                        '<div class="rec-name" title="' + f.name + '">' + f.name + '</div>' +
                        '<div class="rec-meta">' + f.size + ' &middot; ' + f.date + '</div>' +
                    '</div>' +
                    '<div class="rec-actions">' +
                        '<a href="/api/recordings/' + encodeURIComponent(f.name) + '" download style="text-decoration:none">' +
                            '<button class="secondary">&#8595;</button>' +
                        '</a>' +
                        '<button class="danger" onclick="deleteRecording(\'' + f.name.replace(/'/g, "\\'") + '\')">&#10005;</button>' +
                    '</div>';
                list.appendChild(item);
            });
        }

        async function deleteRecording(name) {
            if (!confirm('Delete ' + name + '?')) return;
            const res = await fetch('/api/recordings/' + encodeURIComponent(name), { method: 'DELETE' });
            const data = await res.json();
            if (data.status === 'ok') {
                log('Deleted ' + name, 'success');
                loadRecordings();
            } else {
                log('Delete failed: ' + (data.message || 'unknown error'), 'error');
            }
        }

        // --- Stream dimensions ---
        async function loadConfig() {
            const res = await api('config');
            if (res && res.width && res.height) {
                const root = document.documentElement;
                root.style.setProperty('--stream-width', res.width + 'px');
                root.style.setProperty('--stream-aspect', res.width + '/' + res.height);
                // Adapt grid: for wider streams use more space
                const ratio = res.width / res.height;
                const grid = document.querySelector('.main-grid');
                if (ratio >= 1.6) {
                    grid.style.gridTemplateColumns = '1fr 280px';
                } else {
                    grid.style.gridTemplateColumns = '1fr 300px';
                }
            }
        }

        // --- Auto-refresh recordings after actions ---
        const _origToggleRecording = toggleRecording;
        toggleRecording = async function() {
            await _origToggleRecording();
            if (!isRecording && document.getElementById('rec-body').classList.contains('open')) {
                setTimeout(loadRecordings, 500);
            }
        };
        const _origTakeScreenshot = takeScreenshot;
        takeScreenshot = async function() {
            await _origTakeScreenshot();
            if (document.getElementById('rec-body').classList.contains('open')) {
                setTimeout(loadRecordings, 500);
            }
        };

        // Poll status every second
        setInterval(updateStatus, 1000);
        updateStatus();
        loadConfig();
        log('Web interface loaded');
    </script>
</body>
</html>
"""


# =============================================================================
# Routes
# =============================================================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/video_feed')
def video_feed():
    """MJPEG video stream."""
    def generate():
        while True:
            with _frame_lock:
                frame = _latest_frame

            if frame is not None:
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            else:
                # Placeholder frame
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, 'No Signal', (220, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                _, jpeg = cv2.imencode('.jpg', blank)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

            time.sleep(0.033)  # ~30 FPS

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def api_status():
    """Get current tracker status."""
    if _tracker:
        detection = _tracker.current_detection
        shift_x = shift_y = None
        if detection:
            x, y, w, h = detection['bbox']
            cx = _tracker.capture.width // 2 if _tracker.capture else 640
            cy = _tracker.capture.height // 2 if _tracker.capture else 480
            shift_x = x + w // 2 - cx
            shift_y = y + h // 2 - cy

        return jsonify({
            'running': _tracker.running,
            'recording': _tracker.recording,
            'ev3_connected': _tracker.motors.connected,
            'fps': _tracker.fps,
            'detected': detection is not None,
            'shift_x': shift_x,
            'shift_y': shift_y
        })

    return jsonify({
        'running': False,
        'recording': False,
        'ev3_connected': False,
        'fps': 0,
        'detected': False,
        'shift_x': None,
        'shift_y': None
    })


@app.route('/api/start', methods=['POST'])
def api_start():
    """Start tracking."""
    global _tracker, _tracker_thread

    if _tracker and _tracker.running:
        return jsonify({'status': 'already_running'})

    try:
        _tracker = HumanTracker(_config)
        _tracker_thread = Thread(target=_run_tracker_loop, daemon=True)
        _tracker_thread.start()
        return jsonify({'status': 'ok'})
    except Exception as e:
        logger.error(f"Start failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Stop tracking."""
    global _tracker

    if _tracker:
        _tracker.running = False
        _tracker.cleanup()
        _tracker = None

    return jsonify({'status': 'ok'})


@app.route('/api/reset', methods=['POST'])
def api_reset():
    """Reset detection."""
    if _tracker:
        _tracker.tracker.reset()
        _tracker.current_detection = None
    return jsonify({'status': 'ok'})


@app.route('/api/record', methods=['POST'])
def api_record():
    """Toggle recording."""
    if _tracker:
        if _tracker.recording:
            _tracker.stop_recording()
        else:
            _tracker.start_recording()
        return jsonify({'status': 'ok', 'recording': _tracker.recording})
    return jsonify({'status': 'error', 'message': 'Tracker not running'})


@app.route('/api/screenshot', methods=['POST'])
def api_screenshot():
    """Take screenshot."""
    global _latest_frame

    if _tracker and _latest_frame is not None:
        with _frame_lock:
            _tracker.screenshot(_latest_frame)
        return jsonify({'status': 'ok', 'path': str(_tracker.output_dir)})
    return jsonify({'status': 'error', 'message': 'No frame available'})


@app.route('/api/ev3', methods=['POST'])
def api_ev3():
    """Toggle EV3 connection."""
    if _tracker:
        data = request.get_json() or {}
        if data.get('enabled', False):
            _tracker.motors.connect()
        else:
            _tracker.motors.disconnect()
        return jsonify({'status': 'ok', 'connected': _tracker.motors.connected})
    return jsonify({'status': 'error', 'message': 'Tracker not running'})


@app.route('/api/settings', methods=['POST'])
def api_settings():
    """Update settings."""
    if not _tracker:
        return jsonify({'status': 'error', 'message': 'Tracker not running'})

    data = request.get_json() or {}

    if 'ev3_speed' in data:
        _tracker.motors.speed_factor = min(float(data['ev3_speed']), 2.0)
    if 'ev3_deadzone' in data:
        v = int(data['ev3_deadzone'])
        _tracker.motors.deadzone_x = v
        _tracker.motors.deadzone_y = v
    if 'confidence' in data:
        _tracker.detector.confidence = float(data['confidence'])
    if 'interval' in data:
        _tracker.detection_interval = int(data['interval'])

    return jsonify({'status': 'ok'})


@app.route('/api/config')
def api_config():
    """Return camera resolution for frontend layout adaptation."""
    cam = _config.get('camera', {})
    res = cam.get('resolution', {})
    return jsonify({
        'width': res.get('width', 1280),
        'height': res.get('height', 960)
    })


@app.route('/api/recordings')
def api_recordings():
    """List recording files."""
    output_dir = _config.get('tracker', {}).get('output_dir', 'recordings')
    rec_path = Path(output_dir)
    if not rec_path.exists():
        return jsonify({'files': []})

    files = []
    for f in sorted(rec_path.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if f.is_file() and f.suffix.lower() in ('.avi', '.jpg', '.png', '.txt'):
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
    output_dir = _config.get('tracker', {}).get('output_dir', 'recordings')
    rec_path = Path(output_dir).resolve()
    file_path = (rec_path / filename).resolve()

    # Prevent path traversal
    if not str(file_path).startswith(str(rec_path)):
        return jsonify({'status': 'error', 'message': 'Invalid path'}), 403

    if not file_path.exists():
        return jsonify({'status': 'error', 'message': 'File not found'}), 404

    # Images served inline, everything else as attachment
    as_attachment = file_path.suffix.lower() not in ('.jpg', '.png')
    return send_from_directory(str(rec_path), filename, as_attachment=as_attachment)


@app.route('/api/recordings/<path:filename>', methods=['DELETE'])
def api_recordings_delete(filename):
    """Delete a recording file."""
    output_dir = _config.get('tracker', {}).get('output_dir', 'recordings')
    rec_path = Path(output_dir).resolve()
    file_path = (rec_path / filename).resolve()

    # Prevent path traversal
    if not str(file_path).startswith(str(rec_path)):
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

def _run_tracker_loop():
    """Background tracker loop for web interface."""
    global _latest_frame, _tracker

    if not _tracker.connect():
        logger.error("Failed to connect to stream")
        _tracker = None
        return

    _tracker.running = True
    logger.info("Web tracker loop started")

    try:
        while _tracker and _tracker.running:
            ret, frame = _tracker.capture.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            annotated, _ = _tracker.process_frame(frame)

            with _frame_lock:
                _latest_frame = annotated.copy()

            _tracker.update_fps()
            _tracker.write_frame(annotated)

    except Exception as e:
        logger.error(f"Tracker loop error: {e}")
    finally:
        if _tracker:
            _tracker.cleanup()


# =============================================================================
# Entry Point
# =============================================================================

def run_web(config: dict, host: str = '0.0.0.0', port: int = 5000):
    """Run the web interface."""
    global _config
    _config = config

    logger.info(f"Starting web interface at http://{host}:{port}")
    app.run(host=host, port=port, threaded=True, debug=False)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='piStreamTracker Web Interface')
    parser.add_argument('--config', default='config.yaml', help='Config file')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    args = parser.parse_args()

    config = load_config(args.config)
    run_web(config, args.host, args.port)


if __name__ == '__main__':
    main()
