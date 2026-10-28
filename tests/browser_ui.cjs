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
        const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
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

        for (const width of [390, 900, 1200]) {
            await page.setViewportSize({ width, height: 844 });
            const layout = await page.evaluate(() => {
                const video = document.querySelector('.video-container').getBoundingClientRect();
                const controls = document.querySelector('.controls').getBoundingClientRect();
                return { video: video.toJSON(), controls: controls.toJSON(), scrollWidth: document.documentElement.scrollWidth, innerWidth };
            });
            if (width <= 900) {
                assert(layout.controls.top >= layout.video.bottom, `Controls must stack at ${width}px`);
                assert(layout.video.width >= width - 40, `Preview must fill mobile content at ${width}px`);
            } else {
                assert.equal(layout.controls.width, 300);
                assert.equal(layout.controls.top, layout.video.top);
            }
            assert(layout.scrollWidth <= layout.innerWidth, `Page must fit ${width}px viewport`);
        }

        const stoppingPage = await browser.newPage();
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

        const recordingPage = await browser.newPage();
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

        const healthyPage = await browser.newPage();
        healthyPage.on('pageerror', error => errors.push(error.message));
        await healthyPage.route('**/api/status', route => route.fulfill({ json: {
            running: true, recording: true, fps: 10, shift_x: null, shift_y: null,
            stopping: false, recording_stop_pending: false, warning: null,
        } }));
        await healthyPage.goto(baseUrl);
        await healthyPage.waitForFunction(() => !document.getElementById('btn-record').disabled);
        assert(await healthyPage.locator('#recording-warning').isHidden());

        const servoPage = await browser.newPage({ viewport: { width: 390, height: 844 } });
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
        console.log('Browser UI checks passed: idle Stop/autostart cancellation, recovery, recording, cleanup controls, GPIO status/calibration/errors/manual movement, 390/900/1200px layout.');
    } finally {
        await browser.close();
    }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
