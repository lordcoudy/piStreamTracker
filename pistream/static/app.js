        let isTracking = false;
        let isRecording = false;
        let isStopping = false;
        let recordingStopPending = false;
        let controlRequestPending = false;
        let lastWarning = null;
        let motorBackend = 'ev3';

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
                const text = await res.text();
                let payload;
                try {
                    payload = text ? JSON.parse(text) : {};
                } catch (e) {
                    log('API error: invalid JSON (' + res.status + ')', 'error');
                    return { status: 'error', message: 'Invalid JSON from server' };
                }
                if (payload === null || typeof payload !== 'object' || Array.isArray(payload)) {
                    return { status: 'error', message: 'Unexpected API response' };
                }
                return payload;
            } catch (e) {
                log('API error: ' + e.message, 'error');
                return { status: 'error', message: e.message };
            }
        }

        async function startTracking() {
            if (controlRequestPending) return;
            controlRequestPending = true;
            updateUI();
            log('Starting tracker...');
            try {
                const res = await api('start', 'POST');
                if (res && (res.status === 'ok' || res.status === 'already_running')) {
                    isTracking = true;
                    log('Tracking started', 'success');
                } else {
                    log('Start failed: ' + ((res && res.message) || 'unknown error'), 'error');
                }
                await updateStatus();
            } finally {
                controlRequestPending = false;
                updateUI();
            }
        }

        async function stopTracking() {
            if (controlRequestPending) return;
            controlRequestPending = true;
            updateUI();
            log('Stopping tracker...');
            try {
                const res = await api('stop', 'POST');
                if (res && res.status === 'ok') {
                    isTracking = false;
                    isRecording = false;
                    isStopping = false;
                    recordingStopPending = false;
                    showRecordingWarning(null);
                    log('Tracking stopped', 'success');
                } else if (res && res.status === 'stopping') {
                    isTracking = false;
                    isStopping = true;
                    log('Tracker cleanup is still in progress.');
                } else if (res && res.status === 'recording_stop_pending') {
                    isTracking = false;
                    recordingStopPending = true;
                    showRecordingWarning('Tracking has stopped. Camera recording stop is unconfirmed. Press Stop again to retry.');
                } else {
                    log('Stop failed: ' + ((res && res.message) || 'unknown error'), 'error');
                }
                await updateStatus();
            } finally {
                controlRequestPending = false;
                updateUI();
            }
        }

        async function resetDetection() {
            const res = await api('reset', 'POST');
            if (res && res.status === 'ok') log('Detection reset + camera homed', 'success');
            else if (res) log('Reset failed: ' + (res.message || 'unknown error'), 'error');
        }

        async function moveCamera(direction) {
            const degrees = parseInt(document.getElementById('step-slider').value) || 10;
            const res = await api('motor_move', 'POST', { direction, degrees });
            if (res && res.status === 'ok') {
                log('Move: ' + direction + ' ' + degrees + '°');
            } else if (res) log('Move failed: ' + (res.message || 'unknown error'), 'error');
        }

        async function zoomCamera(action) {
            const res = await api('zoom', 'POST', { action });
            if (res && res.zoom !== undefined) {
                document.getElementById('zoom-value').textContent = res.zoom.toFixed(2) + 'x';
                log('Zoom: ' + res.zoom.toFixed(2) + 'x');
            } else if (res) log('Zoom failed: ' + (res.message || 'unknown error'), 'error');
        }

        async function toggleHorizon() {
            const enabled = document.getElementById('horizon-toggle').checked;
            const res = await api('settings', 'POST', { horizon: enabled });
            if (res && res.status === 'ok') {
                log('Horizon stabilization ' + (enabled ? 'ON' : 'OFF'), 'success');
            } else {
                document.getElementById('horizon-toggle').checked = !enabled;
                if (res) log('Horizon update failed: ' + (res.message || 'unknown error'), 'error');
            }
        }

        async function toggleOverlay() {
            const enabled = document.getElementById('overlay-toggle').checked;
            const res = await api('settings', 'POST', { overlay: enabled });
            if (res && res.status === 'ok') {
                document.getElementById('video-feed').src = '/video_feed?t=' + Date.now();
                log(enabled ? 'Overlay ON' : 'Overlay OFF (raw camera)', 'success');
            } else {
                document.getElementById('overlay-toggle').checked = !enabled;
                if (res) log('Overlay update failed: ' + (res.message || 'unknown error'), 'error');
            }
        }

        async function toggleRecording() {
            const res = await api('record', 'POST');
            if (res && res.status === 'ok') {
                isRecording = res.recording;
                updateUI();
                log(isRecording ? 'Recording started' : 'Recording stopped', 'success');
            } else if (res) log('Recording failed: ' + (res.message || 'unknown error'), 'error');
            await updateStatus();
            if (!isRecording && document.getElementById('rec-body').classList.contains('open')) {
                setTimeout(loadRecordings, 500);
            }
        }

        async function takeScreenshot() {
            const res = await api('screenshot', 'POST');
            if (res && res.status === 'ok') log('Screenshot: ' + res.path, 'success');
            else if (res) log('Screenshot failed: ' + (res.message || 'unknown error'), 'error');
            if (document.getElementById('rec-body').classList.contains('open')) {
                setTimeout(loadRecordings, 500);
            }
        }

        async function toggleEV3() {
            const enabled = document.getElementById('ev3-toggle').checked;
            const res = await api(motorBackend === 'ev3' ? 'ev3' : 'motors', 'POST', { enabled });
            if (res && res.status === 'ok') {
                log('Motors ' + (enabled ? 'connected' : 'disconnected'), 'success');
            } else {
                document.getElementById('ev3-toggle').checked = !enabled;
                if (res) log('Motor update failed: ' + (res.message || 'unknown error'), 'error');
            }
            await updateStatus();
        }

        function showMotorStatus(motors) {
            if (!motors) return;
            motorBackend = motors.backend || motorBackend;
            const servo = motorBackend === 'gpio_servo';
            const name = servo ? 'GPIO servos' : motorBackend === 'none' ? 'No motors' : 'EV3';
            document.getElementById('motor-label').textContent = name;
            document.getElementById('motor-panel-title').textContent = name + ' Control';
            document.getElementById('status-ev3').className = 'dot ' + (motors.connected ? 'green' : 'red');
            document.getElementById('ev3-toggle').checked = Boolean(motors.connected);
            document.getElementById('ev3-toggle').disabled = !isTracking || motorBackend === 'none';
            document.getElementById('motor-connection').textContent = motors.connected ? 'Connected' : 'Disconnected';
            document.getElementById('ev3-settings').hidden = motorBackend !== 'ev3';
            document.getElementById('servo-status').hidden = !servo;
            const error = document.getElementById('motor-error');
            error.textContent = motors.error || '';
            error.hidden = !motors.error;
            if (servo) {
                document.getElementById('servo-calibration').textContent = motors.calibrated
                    ? 'Calibrated · ' + (motors.mode || 'idle')
                    : 'Calibration required: automatic motion disabled. Use the calibration utility before tracking.';
                for (const axis of ['pan', 'tilt']) {
                    const state = motors[axis] || {};
                    const degrees = value => typeof value === 'number' ? value.toFixed(1) + '°' : '—';
                    document.getElementById('servo-' + axis).textContent =
                        axis[0].toUpperCase() + axis.slice(1) + ': commanded ' + degrees(state.commanded_angle) +
                        ', target ' + degrees(state.target_angle) +
                        '; limits ' + degrees(state.min_angle) + ' … ' + degrees(state.max_angle);
                }
            }
        }

        async function updateSetting(key, value) {
            // Update display
            if (key === 'ev3_speed') document.getElementById('speed-value').textContent = value;
            if (key === 'ev3_deadzone') document.getElementById('deadzone-value').textContent = value;
            if (key === 'confidence') document.getElementById('conf-value').textContent = value;
            if (key === 'interval') document.getElementById('interval-value').textContent = value;

            const res = await api('settings', 'POST', { [key]: parseFloat(value) });
            if (res && res.status !== 'ok') {
                log('Setting update failed: ' + (res.message || 'unknown error'), 'error');
            }
        }

        function updateUI() {
            document.getElementById('btn-start').disabled = isTracking || isStopping || recordingStopPending || controlRequestPending;
            document.getElementById('btn-stop').disabled = controlRequestPending;
            document.getElementById('btn-stop').textContent = recordingStopPending && !isTracking ? 'Retry camera stop' : 'Stop';
            document.getElementById('btn-record').disabled = !isTracking || isStopping || controlRequestPending;
            document.getElementById('btn-record').textContent = isRecording ? 'Stop Rec' : 'Record';
            document.getElementById('btn-record').className = isRecording ? 'danger' : '';
            document.getElementById('status-recording').className = 'dot ' + (isRecording ? 'red' : 'yellow');
        }

        function showRecordingWarning(warning) {
            const element = document.getElementById('recording-warning');
            element.textContent = warning || '';
            element.hidden = !warning;
            if (warning && warning !== lastWarning) log(warning, 'error');
            lastWarning = warning || null;
        }

        async function updateStatus() {
            const res = await api('status');
            if (!res || typeof res.running !== 'boolean' || typeof res.fps !== 'number') {
                return;
            }
            isTracking = res.running;
            isRecording = res.recording;
            isStopping = Boolean(res.stopping);
            recordingStopPending = Boolean(res.recording_stop_pending);
            showRecordingWarning(res.warning || null);
            const trackingDot = document.getElementById('status-tracking');
            if (res.running && res.stream_lost) {
                trackingDot.className = 'dot yellow';
            } else {
                trackingDot.className = 'dot ' + (isTracking ? 'green' : 'red');
            }
            showMotorStatus(res.motors || { backend: motorBackend, connected: res.ev3_connected });
            document.getElementById('info-fps').textContent = res.fps.toFixed(1);
            document.getElementById('info-detection').textContent = res.detected ? 'Yes' : 'No';
            if (res.shift_x !== null) {
                document.getElementById('info-shift-x').textContent = res.shift_x;
                document.getElementById('info-shift-y').textContent = res.shift_y;
            } else {
                document.getElementById('info-shift-x').textContent = '--';
                document.getElementById('info-shift-y').textContent = '--';
            }
            if (res.zoom !== undefined) {
                document.getElementById('zoom-value').textContent = res.zoom.toFixed(2) + 'x';
            }
            if (res.horizon !== undefined) {
                document.getElementById('horizon-toggle').checked = res.horizon;
            }
            if (res.overlay !== undefined) {
                document.getElementById('overlay-toggle').checked = res.overlay;
            }
            updateUI();
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
            if (res && res.warning) log(res.warning, 'error');
            if (!res || !res.files || res.files.length === 0) {
                list.innerHTML = '<div class="rec-empty">No recordings found</div>';
                return;
            }
            list.innerHTML = '';
            res.files.forEach(f => {
                const sourceQuery = f.source === 'camera' ? '?source=camera' : '';
                const item = document.createElement('div');
                item.className = 'rec-item';

                if (f.name.toLowerCase().endsWith('.jpg') || f.name.toLowerCase().endsWith('.png')) {
                    const thumb = document.createElement('img');
                    thumb.className = 'rec-thumb';
                    thumb.src = '/api/recordings/' + encodeURIComponent(f.name) + sourceQuery;
                    thumb.alt = 'thumb';
                    item.appendChild(thumb);
                }

                const info = document.createElement('div');
                info.className = 'rec-info';
                const nameEl = document.createElement('div');
                nameEl.className = 'rec-name';
                nameEl.title = f.name;
                nameEl.textContent = f.name;
                const metaEl = document.createElement('div');
                metaEl.className = 'rec-meta';
                metaEl.textContent = f.size + ' · ' + f.date;
                info.appendChild(nameEl);
                info.appendChild(metaEl);
                item.appendChild(info);

                const actions = document.createElement('div');
                actions.className = 'rec-actions';

                const dlLink = document.createElement('a');
                dlLink.href = '/api/recordings/' + encodeURIComponent(f.name) + sourceQuery;
                dlLink.download = f.name;
                dlLink.style.textDecoration = 'none';
                const dlBtn = document.createElement('button');
                dlBtn.className = 'secondary';
                dlBtn.textContent = '↓';
                dlLink.appendChild(dlBtn);
                actions.appendChild(dlLink);

                const delBtn = document.createElement('button');
                delBtn.className = 'danger';
                delBtn.textContent = '✕';
                delBtn.dataset.filename = f.name;
                delBtn.dataset.source = f.source || 'local';
                delBtn.disabled = Boolean(f.active);
                delBtn.title = f.active ? 'Stop recording before deleting' : 'Delete';
                delBtn.addEventListener('click', function() {
                    deleteRecording(this.dataset.filename, this.dataset.source);
                });
                actions.appendChild(delBtn);

                item.appendChild(actions);
                list.appendChild(item);
            });
        }

        async function deleteRecording(name, source = 'local') {
            if (!confirm('Delete ' + name + '?')) return;
            const query = source === 'camera' ? '?source=camera' : '';
            const data = await api('recordings/' + encodeURIComponent(name) + query, 'DELETE');
            if (data && data.status === 'ok') {
                log('Deleted ' + name, 'success');
                loadRecordings();
            } else {
                log('Delete failed: ' + (data.message || 'unknown error'), 'error');
            }
        }

        // --- Stream dimensions ---
        async function loadConfig() {
            const res = await api('config');
            if (!res) return;
            if (res.width && res.height) {
                const root = document.documentElement;
                root.style.setProperty('--stream-width', res.width + 'px');
                root.style.setProperty('--stream-aspect', res.width + '/' + res.height);
                const ratio = res.width / res.height;
                root.style.setProperty('--controls-width', ratio >= 1.6 ? '280px' : '300px');
            }
            if (res.confidence !== undefined) {
                document.getElementById('conf-slider').value = res.confidence;
                document.getElementById('conf-value').textContent = res.confidence;
            }
            if (res.interval !== undefined) {
                document.getElementById('interval-slider').value = res.interval;
                document.getElementById('interval-value').textContent = res.interval;
            }
            if (res.overlay !== undefined) {
                document.getElementById('overlay-toggle').checked = res.overlay;
            }
            if (res.motors) {
                showMotorStatus(res.motors);
                if (res.motors.backend === 'gpio_servo') {
                    document.getElementById('step-slider').value = 2;
                    document.getElementById('step-value').textContent = '2';
                    document.querySelector('.dpad-center').textContent = '2°';
                }
            }
        }

        // Poll status every second
        setInterval(updateStatus, 1000);
        updateStatus();
        loadConfig();
        log('Web interface loaded');
