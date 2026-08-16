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
            } else {
                log('Start failed: ' + ((res && res.message) || 'unknown error'), 'error');
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
            if (res) log('Detection reset + camera homed', 'success');
        }

        async function moveCamera(direction) {
            const degrees = parseInt(document.getElementById('step-slider').value) || 10;
            const res = await api('motor_move', 'POST', { direction, degrees });
            if (res && res.status === 'ok') {
                log('Move: ' + direction + ' ' + degrees + '°');
            }
        }

        async function zoomCamera(action) {
            const res = await api('zoom', 'POST', { action });
            if (res && res.zoom !== undefined) {
                document.getElementById('zoom-value').textContent = res.zoom.toFixed(2) + 'x';
                log('Zoom: ' + res.zoom.toFixed(2) + 'x');
            }
        }

        async function toggleHorizon() {
            const enabled = document.getElementById('horizon-toggle').checked;
            await api('settings', 'POST', { horizon: enabled });
            log('Horizon stabilization ' + (enabled ? 'ON' : 'OFF'), 'success');
        }

        async function toggleOverlay() {
            const enabled = document.getElementById('overlay-toggle').checked;
            const res = await api('settings', 'POST', { overlay: enabled });
            if (res && res.status === 'ok') {
                document.getElementById('video-feed').src = '/video_feed?t=' + Date.now();
                log(enabled ? 'Overlay ON' : 'Overlay OFF (raw camera)', 'success');
            }
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

                if (f.name.toLowerCase().endsWith('.jpg') || f.name.toLowerCase().endsWith('.png')) {
                    const thumb = document.createElement('img');
                    thumb.className = 'rec-thumb';
                    thumb.src = '/api/recordings/' + encodeURIComponent(f.name);
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
                dlLink.href = '/api/recordings/' + encodeURIComponent(f.name);
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
                delBtn.addEventListener('click', function() {
                    deleteRecording(this.dataset.filename);
                });
                actions.appendChild(delBtn);

                item.appendChild(actions);
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
            if (!res) return;
            if (res.width && res.height) {
                const root = document.documentElement;
                root.style.setProperty('--stream-width', res.width + 'px');
                root.style.setProperty('--stream-aspect', res.width + '/' + res.height);
                const ratio = res.width / res.height;
                const grid = document.querySelector('.main-grid');
                if (ratio >= 1.6) {
                    grid.style.gridTemplateColumns = '1fr 280px';
                } else {
                    grid.style.gridTemplateColumns = '1fr 300px';
                }
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
