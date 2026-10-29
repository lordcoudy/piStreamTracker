        let isTracking = false;
        let isRecording = false;
        let isStopping = false;
        let recordingStopPending = false;
        let controlRequestPending = false;
        let lastWarning = null;
        let motorBackend = 'ev3';
        let cameraSettings = null;
        let cameraLimits = {};
        let cameraSettingsPending = false;
        let previewEnabled = false;
        let previewRequestPending = false;
        let previewMode = null;
        let previewRevision = 0;
        let previewSettings = null;
        let previewSettingsPending = false;
        const cameraImageFields = [
            { key: 'brightness', control: 'Brightness', label: 'Brightness', min: -1, max: 1, default: 0 },
            { key: 'contrast', control: 'Contrast', label: 'Contrast', min: 0, max: 32, default: 1 },
            { key: 'saturation', control: 'Saturation', label: 'Saturation', min: 0, max: 32, default: 1 },
            { key: 'sharpness', control: 'Sharpness', label: 'Sharpness', min: 0, max: 16, default: 1 },
            { key: 'exposure_value', control: 'ExposureValue', label: 'Exposure compensation', min: -8, max: 8, default: 0 },
        ];
        const noiseReductionModes = ['off', 'fast', 'high_quality', 'minimal'];

        function cameraImageId(key) {
            return 'camera-' + key.replaceAll('_', '-');
        }

        function hasCameraControl(name) {
            return Object.prototype.hasOwnProperty.call(cameraLimits, name);
        }

        function reconnectPreview() {
            if (!previewEnabled) return;
            previewMode = isTracking && document.getElementById('overlay-toggle').checked ? 'overlay' : 'camera';
            document.getElementById('video-feed').src = '/video_feed?t=' + Date.now() + '&v=' + (++previewRevision);
        }

        function updatePreviewUI() {
            const button = document.getElementById('btn-preview');
            button.disabled = previewRequestPending;
            button.textContent = previewRequestPending ? 'Opening camera…' : previewEnabled ? 'Hide camera' : 'Show camera';
            button.setAttribute('aria-pressed', String(previewEnabled));
            document.getElementById('video-feed').hidden = !previewEnabled;
            document.getElementById('preview-placeholder').hidden = previewEnabled;
            const mode = isTracking && document.getElementById('overlay-toggle').checked ? 'overlay' : 'camera';
            const display = document.getElementById('fps-display');
            display.hidden = !previewEnabled;
            display.textContent = !isTracking ? 'Camera only · tracking off' : mode === 'overlay' ? 'Tracking preview' : 'Camera · tracking on';
            if (previewEnabled && previewMode !== mode) reconnectPreview();
        }

        function setPreviewVisible(enabled) {
            previewEnabled = enabled;
            previewMode = null;
            if (!enabled) document.getElementById('video-feed').removeAttribute('src');
            updatePreviewUI();
        }

        async function togglePreview() {
            if (previewRequestPending) return;
            const status = document.getElementById('preview-status');
            status.className = '';
            if (previewEnabled) {
                setPreviewVisible(false);
                status.textContent = 'Preview is off. Tracking and recording continue unchanged.';
                return;
            }
            previewRequestPending = true;
            updatePreviewUI();
            try {
                const result = await api('preview', 'POST', { enabled: true });
                if (!result || result.status !== 'ok') throw new Error((result && result.message) || 'Could not open camera preview');
                setPreviewVisible(true);
                status.textContent = 'Camera preview enabled. Pending automatic start cancelled. Start tracking when needed.';
                await updateStatus();
            } catch (error) {
                status.textContent = error.message;
                status.className = 'error';
            } finally {
                previewRequestPending = false;
                updatePreviewUI();
            }
        }

        // Camera controls are an explicit saved draft, independent of status polling.
        function cameraMessage(message, type = '') {
            const element = document.getElementById('camera-settings-status');
            element.textContent = message;
            element.className = 'camera-message' + (type ? ' ' + type : '');
        }

        function setStreamDimensions(width, height) {
            const root = document.documentElement;
            root.style.setProperty('--stream-width', width + 'px');
            root.style.setProperty('--stream-aspect', width + '/' + height);
            root.style.setProperty('--controls-width', width / height >= 1.6 ? '280px' : '300px');
        }

        function updateCameraResolutions(selected = '') {
            const aspect = document.getElementById('camera-aspect').value;
            const resolutions = aspect === '4:3'
                ? [['1280x960', '1280 × 960 · HD']]
                : [['1280x720', '1280 × 720 · HD'], ['1920x1080', '1920 × 1080 · Full HD']];
            const select = document.getElementById('camera-resolution');
            select.replaceChildren(...resolutions.map(([value, text]) => new Option(text, value)));
            if (selected) {
                if (!resolutions.some(([value]) => value === selected)) select.add(new Option(selected.replace('x', ' × ') + ' · current', selected));
                select.value = selected;
            }
        }

        function cameraLimit(name, bound, fallback) {
            const value = cameraLimits[name] && cameraLimits[name][bound];
            const values = Array.isArray(value) ? value : [value];
            return values.every(Number.isFinite) ? (bound === 'min' ? Math.max(...values) : Math.min(...values)) : fallback;
        }

        function updateCameraManualFields() {
            document.getElementById('camera-wb-manual').hidden = document.getElementById('camera-awb-mode').value !== 'manual';
            document.getElementById('camera-exposure-manual').hidden = document.getElementById('camera-exposure-mode').value !== 'manual';
            const exposure = document.getElementById('camera-exposure-ms');
            const fps = Number(document.getElementById('camera-framerate').value);
            exposure.min = Math.max(0.1, cameraLimit('ExposureTime', 'min', 100) / 1000);
            exposure.max = Math.min(1000 / fps, cameraLimit('ExposureTime', 'max', 40000) / 1000);
            document.getElementById('camera-exposure-hint').textContent =
                'Shutter must be ' + Number(exposure.min).toFixed(2) + '–' + Number(exposure.max).toFixed(2) + ' ms at ' + fps + ' FPS.';
            const gain = document.getElementById('camera-analogue-gain');
            gain.min = Math.max(1, cameraLimit('AnalogueGain', 'min', 1));
            gain.max = Math.min(16, cameraLimit('AnalogueGain', 'max', 16));
            for (const id of ['camera-red-gain', 'camera-blue-gain']) {
                document.getElementById(id).min = Math.max(0.1, cameraLimit('ColourGains', 'min', 0.1));
                document.getElementById(id).max = Math.min(8, cameraLimit('ColourGains', 'max', 8));
            }
            const manualExposure = document.getElementById('camera-exposure-mode').value === 'manual';
            for (const field of cameraImageFields) {
                const input = document.getElementById(cameraImageId(field.key));
                input.min = Math.max(field.min, cameraLimit(field.control, 'min', field.min));
                input.max = Math.min(field.max, cameraLimit(field.control, 'max', field.max));
                input.disabled = !hasCameraControl(field.control) || (field.key === 'exposure_value' && manualExposure);
                input.title = hasCameraControl(field.control) ? '' : 'Unsupported by this camera';
            }
            const noise = document.getElementById('camera-noise-reduction');
            noise.disabled = !hasCameraControl('NoiseReductionMode');
            noise.title = noise.disabled ? 'Unsupported by this camera' : '';
            for (const option of noise.options) {
                const value = noiseReductionModes.indexOf(option.value);
                option.disabled = value < cameraLimit('NoiseReductionMode', 'min', 0) || value > cameraLimit('NoiseReductionMode', 'max', 3);
            }
            document.getElementById('camera-ev-hint').textContent = manualExposure
                ? 'Exposure compensation is saved for Auto exposure; it has no effect in Manual mode. Unsupported controls are disabled.'
                : 'Exposure compensation applies to Auto exposure. Unsupported controls are disabled for this camera.';
        }

        function updateCameraAvailability() {
            const locked = isRecording || recordingStopPending;
            document.getElementById('camera-fields').disabled = cameraSettingsPending || !cameraSettings || locked;
            document.getElementById('camera-save').disabled = cameraSettingsPending || !cameraSettings || locked || isStopping || controlRequestPending;
            document.getElementById('camera-reload').disabled = cameraSettingsPending;
            document.getElementById('camera-save').textContent = cameraSettingsPending ? 'Applying…' : 'Save & apply';
            document.getElementById('camera-recording-note').hidden = !locked;
            document.getElementById('camera-settings-form').setAttribute('aria-busy', String(cameraSettingsPending));
        }

        function showCameraReadings(metadata = {}) {
            metadata = metadata || {};
            const readings = [];
            if (Number.isFinite(metadata.ExposureTime)) readings.push('Shutter ' + (metadata.ExposureTime / 1000).toFixed(1) + ' ms');
            if (Number.isFinite(metadata.AnalogueGain)) readings.push('gain ' + metadata.AnalogueGain.toFixed(2));
            if (Number.isFinite(metadata.ColourTemperature)) readings.push('white balance ' + Math.round(metadata.ColourTemperature) + ' K');
            if (Array.isArray(metadata.ColourGains) && metadata.ColourGains.length === 2 && metadata.ColourGains.every(Number.isFinite)) {
                readings.push('R/B ' + metadata.ColourGains.map(value => value.toFixed(2)).join(' / '));
            }
            if (Number.isFinite(metadata.FrameDuration) && metadata.FrameDuration > 0) readings.push((1000000 / metadata.FrameDuration).toFixed(1) + ' camera FPS');
            const element = document.getElementById('camera-current-readings');
            element.textContent = readings.length ? 'Measured: ' + readings.join(' · ') : '';
            element.hidden = !readings.length;
        }

        function showCameraSettings(payload) {
            const settings = payload.settings;
            if (!settings || !settings.resolution || !Number.isFinite(settings.resolution.width) || !Number.isFinite(settings.resolution.height)) {
                throw new Error('Invalid camera settings from server');
            }
            cameraSettings = settings;
            cameraLimits = payload.limits || {};
            const { width, height } = settings.resolution;
            document.getElementById('camera-aspect').value = width / height >= 1.6 ? '16:9' : '4:3';
            updateCameraResolutions(width + 'x' + height);
            const fps = document.getElementById('camera-framerate');
            if (!Array.from(fps.options).some(option => Number(option.value) === settings.framerate)) {
                fps.add(new Option(settings.framerate + ' FPS', settings.framerate));
            }
            fps.value = settings.framerate;
            document.getElementById('camera-awb-mode').value = settings.awb_mode;
            document.getElementById('camera-red-gain').value = (settings.colour_gains || [1.5, 1.5])[0];
            document.getElementById('camera-blue-gain').value = (settings.colour_gains || [1.5, 1.5])[1];
            document.getElementById('camera-exposure-mode').value = settings.exposure_mode;
            document.getElementById('camera-exposure-ms').value = (settings.exposure_time_us || 20000) / 1000;
            document.getElementById('camera-analogue-gain').value = settings.analogue_gain || 1;
            for (const field of cameraImageFields) {
                document.getElementById(cameraImageId(field.key)).value = settings[field.key] ?? field.default;
            }
            document.getElementById('camera-noise-reduction').value = settings.noise_reduction ?? 'fast';
            updateCameraManualFields();
            showCameraReadings(payload.metadata);
            setStreamDimensions(width, height);
        }

        async function savedSettingsRequest(endpoint, label, method = 'GET', settings) {
            const options = { method };
            if (settings) {
                options.headers = { 'Content-Type': 'application/json' };
                options.body = JSON.stringify(settings);
            }
            const response = await fetch('/api/' + endpoint, options);
            let payload;
            try {
                payload = await response.json();
            } catch (_) {
                throw new Error(label + ' are unavailable (' + response.status + ').');
            }
            if (!response.ok || !payload || (payload.status !== undefined && payload.status !== 'ok')) {
                throw new Error((payload && payload.message) || label + ' are unavailable (' + response.status + ').');
            }
            return payload;
        }

        async function loadCameraSettings() {
            if (cameraSettingsPending) return;
            cameraSettingsPending = true;
            updateCameraAvailability();
            cameraMessage('Loading camera settings…');
            try {
                showCameraSettings(await savedSettingsRequest('camera/settings', 'Camera settings'));
                cameraMessage('Saved camera settings loaded.');
            } catch (error) {
                cameraMessage(error.message + ' Use Reload saved to retry.', 'error');
            } finally {
                cameraSettingsPending = false;
                updateUI();
            }
        }

        function cameraNumber(id, label) {
            const input = document.getElementById(id);
            const value = input.valueAsNumber;
            if (!Number.isFinite(value) || value < Number(input.min) || value > Number(input.max)) {
                throw new Error(label + ' must be between ' + Number(input.min).toFixed(2) + ' and ' + Number(input.max).toFixed(2) + '.');
            }
            return value;
        }

        async function saveCameraSettings(event) {
            event.preventDefault();
            if (document.getElementById('camera-save').disabled) return;
            let settings;
            try {
                const [width, height] = document.getElementById('camera-resolution').value.split('x').map(Number);
                const framerate = Number(document.getElementById('camera-framerate').value);
                const awb_mode = document.getElementById('camera-awb-mode').value;
                const exposure_mode = document.getElementById('camera-exposure-mode').value;
                if (![width, height, framerate].every(Number.isFinite) || width < 1280 || height < 720 || framerate < 25 || framerate > 40) {
                    throw new Error('Choose at least HD resolution and 25–40 camera FPS.');
                }
                settings = {
                    resolution: { width, height }, framerate, awb_mode,
                    colour_gains: awb_mode === 'manual'
                        ? [cameraNumber('camera-red-gain', 'Red gain'), cameraNumber('camera-blue-gain', 'Blue gain')]
                        : cameraSettings.colour_gains,
                    exposure_mode,
                    exposure_time_us: exposure_mode === 'manual'
                        ? Math.round(cameraNumber('camera-exposure-ms', 'Exposure in ms') * 1000)
                        : Math.min(cameraSettings.exposure_time_us, Math.floor(1000000 / framerate)),
                    analogue_gain: exposure_mode === 'manual' ? cameraNumber('camera-analogue-gain', 'Analogue gain') : cameraSettings.analogue_gain,
                };
                for (const field of cameraImageFields) {
                    if (hasCameraControl(field.control)) settings[field.key] = cameraNumber(cameraImageId(field.key), field.label);
                }
                if (hasCameraControl('NoiseReductionMode')) {
                    const noise = document.getElementById('camera-noise-reduction');
                    if (!noiseReductionModes.includes(noise.value) || noise.selectedOptions[0].disabled) throw new Error('Choose a supported noise reduction mode.');
                    settings.noise_reduction = noise.value;
                }
            } catch (error) {
                cameraMessage(error.message, 'error');
                return;
            }
            cameraSettingsPending = true;
            updateUI();
            cameraMessage('Applying camera settings…');
            try {
                const payload = await savedSettingsRequest('camera/settings', 'Camera settings', 'POST', settings);
                showCameraSettings(payload);
                reconnectPreview();
                if (payload.warning) {
                    cameraMessage('Camera settings saved. ' + payload.warning, 'warning');
                    log('Camera settings saved. ' + payload.warning, 'error');
                } else {
                    cameraMessage('Camera settings saved and applied.', 'success');
                    log('Camera settings saved and applied', 'success');
                }
                await updateStatus();
            } catch (error) {
                cameraMessage(error.message, 'error');
            } finally {
                cameraSettingsPending = false;
                updateUI();
            }
        }

        function previewSettingsMessage(message, type = '') {
            const element = document.getElementById('preview-settings-status');
            element.textContent = message;
            element.className = 'camera-message' + (type ? ' ' + type : '');
        }

        function updatePreviewSettingsAvailability() {
            document.getElementById('preview-settings-fields').disabled = previewSettingsPending || !previewSettings;
            document.getElementById('preview-settings-save').disabled = previewSettingsPending || !previewSettings;
            document.getElementById('preview-settings-reload').disabled = previewSettingsPending;
            document.getElementById('preview-settings-save').textContent = previewSettingsPending ? 'Applying…' : 'Save & apply preview';
            document.getElementById('preview-settings-form').setAttribute('aria-busy', String(previewSettingsPending));
        }

        function showPreviewSettings(payload) {
            const settings = payload.settings;
            if (!settings || !Number.isInteger(settings.max_edge) || settings.max_edge < 0 || settings.max_edge > 16384 ||
                !Number.isInteger(settings.quality) || settings.quality < 1 || settings.quality > 100) {
                throw new Error('Invalid preview settings from server');
            }
            previewSettings = settings;
            const size = document.getElementById('preview-max-edge');
            if (!Array.from(size.options).some(option => Number(option.value) === settings.max_edge)) {
                size.add(new Option(settings.max_edge + ' px · saved', settings.max_edge));
            }
            size.value = settings.max_edge;
            document.getElementById('preview-quality').value = settings.quality;
        }

        async function loadPreviewSettings() {
            if (previewSettingsPending) return;
            previewSettingsPending = true;
            updatePreviewSettingsAvailability();
            previewSettingsMessage('Loading preview settings…');
            try {
                showPreviewSettings(await savedSettingsRequest('preview/settings', 'Preview settings'));
                previewSettingsMessage('Saved preview settings loaded.');
            } catch (error) {
                previewSettingsMessage(error.message + ' Use Reload saved to retry.', 'error');
            } finally {
                previewSettingsPending = false;
                updatePreviewSettingsAvailability();
            }
        }

        async function savePreviewSettings(event) {
            event.preventDefault();
            if (document.getElementById('preview-settings-save').disabled) return;
            let settings;
            try {
                const max_edge = Number(document.getElementById('preview-max-edge').value);
                const quality = cameraNumber('preview-quality', 'JPEG quality');
                if (!Number.isInteger(max_edge) || max_edge < 0 || max_edge > 16384) throw new Error('Choose a supported preview size.');
                if (!Number.isInteger(quality)) throw new Error('JPEG quality must be an integer.');
                settings = { max_edge, quality };
            } catch (error) {
                previewSettingsMessage(error.message, 'error');
                return;
            }
            previewSettingsPending = true;
            updatePreviewSettingsAvailability();
            previewSettingsMessage('Applying preview settings…');
            try {
                showPreviewSettings(await savedSettingsRequest('preview/settings', 'Preview settings', 'POST', settings));
                previewSettingsMessage('Preview settings saved and applied.', 'success');
            } catch (error) {
                previewSettingsMessage(error.message, 'error');
            } finally {
                previewSettingsPending = false;
                updatePreviewSettingsAvailability();
            }
        }

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
                    setPreviewVisible(true);
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
                reconnectPreview();
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
            document.getElementById('btn-start').disabled = isTracking || isStopping || recordingStopPending || controlRequestPending || cameraSettingsPending;
            document.getElementById('btn-stop').disabled = controlRequestPending;
            document.getElementById('btn-stop').textContent = recordingStopPending && !isTracking ? 'Retry camera stop' : 'Stop';
            document.getElementById('btn-record').disabled = !isTracking || isStopping || controlRequestPending || cameraSettingsPending;
            document.getElementById('btn-record').textContent = isRecording ? 'Stop Rec' : 'Record';
            document.getElementById('btn-record').className = isRecording ? 'danger' : '';
            document.getElementById('status-recording').className = 'dot ' + (isRecording ? 'red' : 'yellow');
            updateCameraAvailability();
            updatePreviewUI();
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
            if (res.width && res.height && !cameraSettings) setStreamDimensions(res.width, res.height);
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
        loadCameraSettings();
        loadPreviewSettings();
        log('Web interface loaded');
