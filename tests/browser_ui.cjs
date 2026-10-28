// Run against tests/web_ui_fixture.py, with Playwright available through
// PLAYWRIGHT_MODULE and optional PLAYWRIGHT_EXECUTABLE paths.
const assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const baseUrl = process.env.PISTREAM_UI_TEST_URL || 'http://127.0.0.1:5089';

async function main() {
    const browser = await chromium.launch({
        headless: true,
        ...(process.env.PLAYWRIGHT_EXECUTABLE ? { executablePath: process.env.PLAYWRIGHT_EXECUTABLE } : {}),
    });
    try {
        async function newPage(options) {
            const page = await browser.newPage(options);
            let settings = {
                resolution: { width: 1280, height: 960 }, framerate: 25,
                awb_mode: 'auto', colour_gains: [1.5, 1.6],
                exposure_mode: 'auto', exposure_time_us: 20000, analogue_gain: 1,
            };
            await page.route('**/api/camera/settings', route => {
                if (route.request().method() === 'POST') settings = route.request().postDataJSON();
                return route.fulfill({ json: {
                    status: 'ok', settings, saved: true,
                    limits: { AnalogueGain: { min: 1, max: 10.67, default: 1 } },
                    metadata: { ExposureTime: 12000, AnalogueGain: 1.2, ColourGains: [1.5, 1.6], ColourTemperature: 4500, FrameDuration: 40000 },
                } });
            });
            return page;
        }
        const page = await newPage({ viewport: { width: 390, height: 844 } });
        const errors = [];
        let cameraPosts = 0;
        page.on('pageerror', error => errors.push(error.message));
        page.on('request', request => {
            if (request.method() === 'POST' && request.url().endsWith('/api/camera/settings')) cameraPosts++;
        });
        await page.goto(baseUrl);
        await page.waitForFunction(() => document.getElementById('btn-stop').textContent === 'Retry camera stop');
        assert(await page.locator('#btn-start').isDisabled());
        assert(await page.locator('#btn-stop').isEnabled());
        assert(await page.locator('#btn-record').isDisabled());
        await page.waitForFunction(() => getComputedStyle(document.getElementById('btn-start')).backgroundColor === 'rgb(71, 85, 105)');
        assert(await page.locator('#recording-warning').isVisible());
        for (let i = 0; i < 2; i++) await page.waitForResponse(response => response.url().endsWith('/api/status'));
        assert.equal(await page.locator('#log-output .log-line').filter({ hasText: 'stop is unconfirmed' }).count(), 1);
        assert.equal((await (await page.request.get(baseUrl + '/__fixture/state')).json()).stop_calls, 0);

        const failedStop = page.waitForResponse(response => response.url().endsWith('/api/stop'));
        await page.locator('#btn-stop').click();
        assert.equal((await failedStop).status(), 503);
        await page.waitForFunction(() => !document.getElementById('btn-stop').disabled);
        assert(await page.locator('#btn-start').isDisabled());
        assert(await page.locator('#recording-warning').isVisible());
        assert.equal((await (await page.request.get(baseUrl + '/__fixture/state')).json()).stop_calls, 1);

        const successfulStop = page.waitForResponse(response => response.url().endsWith('/api/stop'));
        await page.locator('#btn-stop').click();
        assert.equal((await successfulStop).status(), 200);
        await page.waitForFunction(() => !document.getElementById('btn-start').disabled);
        assert(await page.locator('#btn-stop').isEnabled());
        assert(await page.locator('#btn-record').isDisabled());
        assert(await page.locator('#recording-warning').isHidden());
        assert.equal((await (await page.request.get(baseUrl + '/__fixture/state')).json()).stop_calls, 2);

        // Idle Stop must remain available to cancel a pending automatic start.
        let releaseIdleStop;
        let idleStopEntered;
        const idleStopGate = new Promise(resolve => { releaseIdleStop = resolve; });
        const enteredIdleStop = new Promise(resolve => { idleStopEntered = resolve; });
        await page.route('**/api/stop', async route => {
            idleStopEntered();
            await idleStopGate;
            await route.continue();
        });
        const idleStopResponse = page.waitForResponse(response => response.url().endsWith('/api/stop'));
        await page.locator('#btn-stop').click();
        await enteredIdleStop;
        assert(await page.locator('#btn-start').isDisabled());
        assert(await page.locator('#btn-stop').isDisabled());
        await page.waitForResponse(response => response.url().endsWith('/api/status'));
        assert(await page.locator('#btn-stop').isDisabled());
        releaseIdleStop();
        assert.equal((await idleStopResponse).status(), 200);
        await page.waitForFunction(() => !document.getElementById('btn-start').disabled);
        assert(await page.locator('#btn-stop').isEnabled());
        assert.equal((await (await page.request.get(baseUrl + '/__fixture/state')).json()).stop_calls, 3);
        await page.unroute('**/api/stop');

        let releaseStart;
        let startEntered;
        const startGate = new Promise(resolve => { releaseStart = resolve; });
        const enteredStart = new Promise(resolve => { startEntered = resolve; });
        await page.route('**/api/start', async route => {
            startEntered();
            await startGate;
            await route.fulfill({ json: { status: 'ok' } });
        });
        await page.locator('#btn-start').click();
        await enteredStart;
        assert(await page.locator('#btn-start').isDisabled());
        await page.waitForResponse(response => response.url().endsWith('/api/status'));
        assert(await page.locator('#btn-start').isDisabled());
        releaseStart();
        await page.waitForFunction(() => !document.getElementById('btn-start').disabled);

        for (const width of [320, 390, 768, 900, 1024, 1200]) {
            await page.setViewportSize({ width, height: 844 });
            const layout = await page.evaluate(() => {
                const video = document.querySelector('.video-container').getBoundingClientRect();
                const controls = document.querySelector('.controls').getBoundingClientRect();
                const settings = document.querySelector('.camera-settings').getBoundingClientRect();
                return { video: video.toJSON(), controls: controls.toJSON(), settings: settings.toJSON(), scrollWidth: document.documentElement.scrollWidth, innerWidth };
            });
            assert(layout.settings.top >= layout.video.bottom, `Camera settings must be below preview at ${width}px`);
            if (width <= 900) {
                assert(layout.controls.top >= layout.settings.bottom, `Controls must stack at ${width}px`);
                assert(layout.video.width >= width - 40, `Preview must fill mobile content at ${width}px`);
            } else {
                assert.equal(layout.controls.width, 300);
                assert.equal(layout.controls.top, layout.video.top);
            }
            assert(layout.scrollWidth <= layout.innerWidth, `Page must fit ${width}px viewport`);
        }

        // Camera edits stay local until Save, survive status polling, and are
        // restored from the saved server settings by Reload or a page reload.
        await page.locator('#camera-settings-form').waitFor();
        await page.waitForFunction(() => !document.getElementById('camera-save').disabled);
        assert.equal(await page.locator('#camera-aspect').inputValue(), '4:3');
        assert.equal(await page.locator('#camera-resolution').inputValue(), '1280x960');
        assert.match(await page.locator('#camera-current-readings').textContent(), /12\.0 ms.*4500 K.*25\.0 camera FPS/);
        assert.equal(await page.locator('#camera-analogue-gain').getAttribute('max'), '10.67');
        await page.locator('#camera-aspect').selectOption('16:9');
        await page.locator('#camera-resolution').selectOption('1920x1080');
        await page.locator('#camera-framerate').selectOption('30');
        await page.locator('#camera-awb-mode').selectOption('manual');
        await page.locator('#camera-red-gain').fill('2.2');
        await page.locator('#camera-blue-gain').fill('1.6');
        await page.locator('#camera-exposure-mode').selectOption('manual');
        await page.locator('#camera-exposure-ms').fill('10');
        await page.locator('#camera-analogue-gain').fill('2');
        await page.waitForResponse(response => response.url().endsWith('/api/status'));
        assert.equal(await page.locator('#camera-resolution').inputValue(), '1920x1080');
        assert.equal(await page.locator('#camera-exposure-ms').inputValue(), '10');
        assert.equal(cameraPosts, 0, 'Editing or polling must not save the camera draft');

        let releaseCameraSave;
        let enteredCameraSave;
        let submittedSettings;
        const cameraSaveGate = new Promise(resolve => { releaseCameraSave = resolve; });
        const cameraSaveEntered = new Promise(resolve => { enteredCameraSave = resolve; });
        await page.route('**/api/camera/settings', async route => {
            if (route.request().method() !== 'POST') return route.fallback();
            submittedSettings = route.request().postDataJSON();
            enteredCameraSave();
            await cameraSaveGate;
            return route.fallback();
        });
        await page.locator('#camera-save').click();
        await cameraSaveEntered;
        assert(await page.locator('#camera-save').isDisabled());
        assert(await page.locator('#camera-reload').isDisabled());
        await page.waitForResponse(response => response.url().endsWith('/api/status'));
        assert(await page.locator('#camera-save').isDisabled());
        releaseCameraSave();
        await page.waitForFunction(() => document.getElementById('camera-settings-status').textContent.includes('saved'));
        assert.deepEqual(submittedSettings, {
            resolution: { width: 1920, height: 1080 }, framerate: 30,
            awb_mode: 'manual', colour_gains: [2.2, 1.6],
            exposure_mode: 'manual', exposure_time_us: 10000, analogue_gain: 2,
        });
        assert.match(await page.locator('#video-feed').getAttribute('src'), /\/video_feed\?t=/);
        assert.equal(await page.evaluate(() => document.documentElement.style.getPropertyValue('--stream-aspect')), '1920/1080');
        await page.locator('#camera-red-gain').fill('3');
        await page.locator('#camera-reload').click();
        await page.waitForFunction(() => document.getElementById('camera-red-gain').value === '2.2');
        await page.reload();
        await page.waitForFunction(() => document.getElementById('camera-resolution').value === '1920x1080');
        assert.equal(await page.locator('#camera-exposure-ms').inputValue(), '10');
        await page.locator('#camera-exposure-ms').fill('40');
        await page.locator('#camera-save').click();
        assert.match(await page.locator('#camera-settings-status').textContent(), /exposure.*33\.33|33\.33.*exposure/i);
        assert.equal(await page.locator('#camera-exposure-ms').inputValue(), '40');
        await page.locator('#camera-exposure-ms').fill('12');
        await page.route('**/api/camera/settings', route => route.fulfill({ status: 503, json: { status: 'error', message: 'Camera unavailable' } }));
        await page.locator('#camera-save').click();
        await page.waitForFunction(() => document.getElementById('camera-settings-status').textContent.includes('Camera unavailable'));
        assert.equal(await page.locator('#camera-exposure-ms').inputValue(), '12');
        assert(await page.locator('#camera-save').isEnabled());
        await page.route('**/api/camera/settings', route => route.fulfill({ json: {
            status: 'ok', saved: true, settings: route.request().postDataJSON(),
            warning: 'Tracking restart failed. Press Start to retry.',
        } }));
        await page.locator('#camera-save').click();
        await page.waitForFunction(() => document.getElementById('camera-settings-status').textContent.includes('Tracking restart failed'));
        assert.match(await page.locator('#camera-settings-status').getAttribute('class'), /warning/);
        assert.match(await page.locator('#log-output').textContent(), /Tracking restart failed/);

        const stoppingPage = await newPage();
        stoppingPage.on('pageerror', error => errors.push(error.message));
        await stoppingPage.route('**/api/status', route => route.fulfill({ json: {
            running: false, recording: false, fps: 0, shift_x: null, shift_y: null,
            stopping: true, recording_stop_pending: false,
            warning: 'Tracker cleanup is unfinished. Press Stop again to retry.',
        } }));
        await stoppingPage.goto(baseUrl);
        await stoppingPage.waitForFunction(() => !document.getElementById('recording-warning').hidden);
        assert(await stoppingPage.locator('#btn-start').isDisabled());
        assert(await stoppingPage.locator('#btn-stop').isEnabled());
        assert(await stoppingPage.locator('#btn-record').isDisabled());

        const recordingPage = await newPage();
        recordingPage.on('pageerror', error => errors.push(error.message));
        await recordingPage.route('**/api/status', route => route.fulfill({ json: {
            running: true, recording: true, fps: 10, shift_x: null, shift_y: null,
            stopping: false, recording_stop_pending: true,
            warning: 'Camera recording stop is unconfirmed. Press Stop Rec to retry, or Stop to stop tracking.',
        } }));
        await recordingPage.goto(baseUrl);
        await recordingPage.waitForFunction(() => !document.getElementById('recording-warning').hidden);
        assert(await recordingPage.locator('#btn-record').isEnabled());
        assert.equal(await recordingPage.locator('#btn-record').textContent(), 'Stop Rec');
        assert.equal(await recordingPage.locator('#btn-stop').textContent(), 'Stop');
        assert(await recordingPage.locator('#camera-save').isDisabled());
        assert(await recordingPage.locator('#camera-recording-note').isVisible());

        const healthyPage = await newPage();
        healthyPage.on('pageerror', error => errors.push(error.message));
        await healthyPage.route('**/api/status', route => route.fulfill({ json: {
            running: true, recording: true, fps: 10, shift_x: null, shift_y: null,
            stopping: false, recording_stop_pending: false, warning: null,
        } }));
        await healthyPage.goto(baseUrl);
        await healthyPage.waitForFunction(() => !document.getElementById('btn-record').disabled);
        assert(await healthyPage.locator('#recording-warning').isHidden());

        const unavailablePage = await newPage();
        unavailablePage.on('pageerror', error => errors.push(error.message));
        let unavailableGets = 0;
        await unavailablePage.route('**/api/camera/settings', route => {
            unavailableGets++;
            return route.fulfill({ status: 503, json: { status: 'error', message: 'Camera offline' } });
        });
        await unavailablePage.goto(baseUrl);
        await unavailablePage.waitForFunction(() => document.getElementById('camera-settings-status').textContent.includes('Camera offline'));
        assert(await unavailablePage.locator('#camera-save').isDisabled());
        assert(await unavailablePage.locator('#camera-reload').isEnabled());
        for (let i = 0; i < 2; i++) await unavailablePage.waitForResponse(response => response.url().endsWith('/api/status'));
        assert.equal(unavailableGets, 1, 'Unavailable camera settings must not retry on every status poll');
        await unavailablePage.locator('#camera-reload').click();
        await unavailablePage.waitForFunction(() => !document.getElementById('camera-reload').disabled);
        assert.equal(unavailableGets, 2);

        const legacyPage = await newPage();
        legacyPage.on('pageerror', error => errors.push(error.message));
        await legacyPage.route('**/api/camera/settings', route => route.fulfill({ json: {
            status: 'ok', settings: {
                resolution: { width: 640, height: 480 }, framerate: 15,
                awb_mode: 'auto', colour_gains: [1.5, 1.5],
                exposure_mode: 'auto', exposure_time_us: 20000, analogue_gain: 1,
            },
        } }));
        await legacyPage.goto(baseUrl);
        await legacyPage.waitForFunction(() => !document.getElementById('camera-save').disabled);
        assert.equal(await legacyPage.locator('#camera-resolution').inputValue(), '640x480');
        assert.equal(await legacyPage.locator('#camera-framerate').inputValue(), '15');
        await legacyPage.locator('#camera-save').click();
        assert.match(await legacyPage.locator('#camera-settings-status').textContent(), /at least HD/);

        const servoPage = await newPage({ viewport: { width: 390, height: 844 } });
        servoPage.on('pageerror', error => errors.push(error.message));
        const motors = {
            backend: 'gpio_servo', connected: true, calibrated: false, mode: 'idle',
            error: 'pigpiod connection interrupted',
            pan: { commanded_angle: 2, target_angle: 4, min_angle: -30, max_angle: 30 },
            tilt: { commanded_angle: -1, target_angle: 0, min_angle: -15, max_angle: 15 },
        };
        await servoPage.route('**/api/config', route => route.fulfill({ json: {
            width: 640, height: 480, motors,
        } }));
        await servoPage.route('**/api/status', route => route.fulfill({ json: {
            running: true, recording: false, fps: 10, shift_x: null, shift_y: null,
            stopping: false, recording_stop_pending: false, warning: null, motors,
        } }));
        await servoPage.goto(baseUrl);
        await servoPage.waitForFunction(() => document.getElementById('motor-label').textContent === 'GPIO servos');
        assert(await servoPage.locator('#ev3-settings').isHidden());
        assert(await servoPage.locator('#servo-status').isVisible());
        assert.match(await servoPage.locator('#servo-pan').textContent(), /commanded 2.0°.*target 4.0°.*limits -30.0° … 30.0°/);
        assert.match(await servoPage.locator('#servo-calibration').textContent(), /automatic motion disabled/);
        assert.match(await servoPage.locator('#motor-error').textContent(), /pigpiod/);
        assert.equal(await servoPage.locator('#step-slider').inputValue(), '2');
        const manualRequest = servoPage.waitForRequest(request => request.url().endsWith('/api/motor_move'));
        await servoPage.route('**/api/motor_move', route => route.fulfill({ json: { status: 'ok' } }));
        await servoPage.getByTitle('Pan Right', { exact: true }).click();
        assert.deepEqual((await manualRequest).postDataJSON(), { direction: 'right', degrees: 2 });

        // A stopped session may still need operator action when daemon failure
        // prevents confirmation that the last PWM command was disabled.
        const stoppedMotors = {
            ...motors, connected: false, pulse_enabled: null,
            error: 'Unable to disable PWM: pigpiod connection lost',
        };
        await servoPage.route('**/api/status', route => route.fulfill({ json: {
            running: false, recording: false, fps: 0, shift_x: null, shift_y: null,
            stopping: false, recording_stop_pending: false, warning: null,
            motors: stoppedMotors,
        } }));
        await servoPage.evaluate(() => updateStatus());
        assert(await servoPage.locator('#motor-error').isVisible());
        assert.match(await servoPage.locator('#motor-error').textContent(), /Unable to disable PWM/);
        assert.equal(await servoPage.locator('#motor-connection').textContent(), 'Disconnected');
        assert(await servoPage.locator('#ev3-toggle').isDisabled());
        assert(await servoPage.locator('#btn-start').isEnabled());
        assert.deepEqual(errors, []);
        console.log('Browser UI checks passed: camera settings save/reload/persistence/validation/limits/offline/recording lock, idle Stop/autostart cancellation, recovery, recording, cleanup controls, GPIO controls, 320–1200px layout.');
    } finally {
        await browser.close();
    }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
