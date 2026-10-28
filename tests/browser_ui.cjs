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
                brightness: 0, contrast: 1, saturation: 1, sharpness: 1,
                exposure_value: 0, noise_reduction: 'fast',
            };
            let previewSettings = { max_edge: 640, quality: 70 };
            await page.route('**/api/preview/settings', route => {
                if (route.request().method() === 'POST') previewSettings = route.request().postDataJSON();
                return route.fulfill({ json: { status: 'ok', settings: previewSettings, saved: true } });
            });
            await page.route('**/api/camera/settings', route => {
                if (route.request().method() === 'POST') settings = route.request().postDataJSON();
                return route.fulfill({ json: {
                    status: 'ok', settings, saved: true,
                    limits: {
                        AnalogueGain: { min: 1, max: 10.67, default: 1 },
                        Brightness: { min: -1, max: 1, default: 0 },
                        Contrast: { min: 0, max: 32, default: 1 },
                        Saturation: { min: 0, max: 32, default: 1 },
                        Sharpness: { min: 0, max: 16, default: 1 },
                        ExposureValue: { min: -8, max: 8, default: 0 },
                        NoiseReductionMode: { min: 0, max: 3, default: 1 },
                    },
                    metadata: { ExposureTime: 12000, AnalogueGain: 1.2, ColourGains: [1.5, 1.6], ColourTemperature: 4500, FrameDuration: 40000 },
                } });
            });
            await page.route('**/api/preview', route => route.fulfill({ json: {
                status: 'ok', enabled: route.request().postDataJSON().enabled,
            } }));
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

        assert.equal(await page.locator('#video-feed').getAttribute('src'), null);
        await page.locator('#btn-preview').click();
        await page.waitForFunction(() => document.getElementById('btn-preview').getAttribute('aria-pressed') === 'true');
        assert.match(await page.locator('#video-feed').getAttribute('src'), /\/video_feed\?t=/);
        assert(await page.locator('#overlay-toggle').isChecked());
        assert(await page.locator('#btn-start').isEnabled());
        await page.locator('#btn-preview').click();
        assert.equal(await page.locator('#video-feed').getAttribute('src'), null);
        assert(await page.locator('#preview-placeholder').isVisible());
        await page.locator('#btn-preview').click();
        await page.waitForFunction(() => document.getElementById('btn-preview').getAttribute('aria-pressed') === 'true');

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
        await page.locator('#camera-brightness').fill('0', { timeout: 5000 });
        await page.locator('#camera-contrast').fill('1.2');
        await page.locator('#camera-saturation').fill('0');
        await page.locator('#camera-sharpness').fill('2.5');
        await page.locator('#camera-exposure-value').fill('1.25');
        await page.locator('#camera-noise-reduction').selectOption('high_quality');
        await page.locator('#camera-exposure-mode').selectOption('manual');
        assert(await page.locator('#camera-exposure-value').isDisabled());
        assert.equal(await page.locator('#camera-exposure-value').inputValue(), '1.25');
        await page.locator('#camera-exposure-ms').fill('10');
        await page.locator('#camera-analogue-gain').fill('2');
        await page.locator('#camera-contrast').fill('33');
        await page.locator('#camera-save').click();
        assert.match(await page.locator('#camera-settings-status').textContent(), /Contrast.*32/);
        await page.locator('#camera-contrast').fill('1.2');
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
            brightness: 0, contrast: 1.2, saturation: 0, sharpness: 2.5,
            exposure_value: 1.25, noise_reduction: 'high_quality',
        });
        assert.match(await page.locator('#video-feed').getAttribute('src'), /\/video_feed\?t=/);
        assert.equal(await page.evaluate(() => document.documentElement.style.getPropertyValue('--stream-aspect')), '1920/1080');
        await page.locator('#camera-red-gain').fill('3');
        await page.locator('#camera-reload').click();
        await page.waitForFunction(() => document.getElementById('camera-red-gain').value === '2.2');
        await page.reload();
        await page.waitForFunction(() => document.getElementById('camera-resolution').value === '1920x1080');
        assert.equal(await page.locator('#camera-exposure-ms').inputValue(), '10');
        assert.equal(await page.locator('#camera-saturation').inputValue(), '0');
        assert.equal(await page.locator('#camera-sharpness').inputValue(), '2.5');
        assert.equal(await page.locator('#camera-noise-reduction').inputValue(), 'high_quality');
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

        const previewPage = await newPage();
        previewPage.on('pageerror', error => errors.push(error.message));
        let previewRunning = false;
        let previewOverlay = true;
        let previewStarts = 0;
        let previewStops = 0;
        let previewRequests = 0;
        let previewEnables = 0;
        previewPage.on('request', request => {
            if (request.url().includes('/video_feed')) previewRequests++;
            if (request.url().endsWith('/api/preview') && request.method() === 'POST') previewEnables++;
        });
        await previewPage.route('**/api/status', route => route.fulfill({ json: {
            running: previewRunning, recording: false, fps: previewRunning ? 10 : 0,
            shift_x: null, shift_y: null, overlay: previewOverlay,
            stopping: false, recording_stop_pending: false, warning: null,
        } }));
        await previewPage.route('**/api/start', route => {
            previewRunning = true;
            previewStarts++;
            return route.fulfill({ json: { status: 'ok' } });
        });
        await previewPage.route('**/api/stop', route => {
            previewRunning = false;
            previewStops++;
            return route.fulfill({ json: { status: 'ok' } });
        });
        await previewPage.route('**/api/settings', route => {
            const body = route.request().postDataJSON();
            if (typeof body.overlay === 'boolean') previewOverlay = body.overlay;
            return route.fulfill({ json: { status: 'ok' } });
        });
        await previewPage.goto(baseUrl);
        await previewPage.waitForFunction(() => !document.getElementById('camera-save').disabled);
        assert.equal(previewRequests, 0, 'An idle page must not consume a stream client');
        await previewPage.locator('#btn-preview').click();
        await previewPage.waitForFunction(() => document.getElementById('btn-preview').getAttribute('aria-pressed') === 'true');
        assert.equal(previewStarts, 0);
        assert.equal(previewStops, 0);
        assert.match(await previewPage.locator('#preview-status').textContent(), /automatic start.*cancelled/i);
        const rawSource = await previewPage.locator('#video-feed').getAttribute('src');
        await previewPage.locator('#btn-start').click();
        await previewPage.waitForFunction(() => document.getElementById('fps-display').textContent.includes('Tracking preview'));
        assert.notEqual(await previewPage.locator('#video-feed').getAttribute('src'), rawSource);
        await previewPage.locator('#btn-preview').click();
        await previewPage.waitForResponse(response => response.url().endsWith('/api/status'));
        assert.equal(await previewPage.locator('#video-feed').getAttribute('src'), null);
        const hiddenRequests = previewRequests;
        await previewPage.getByText('Tracking overlay', { exact: true }).click();
        await previewPage.locator('#camera-save').click();
        await previewPage.waitForFunction(() => document.getElementById('camera-settings-status').textContent.includes('saved and applied'));
        assert.equal(previewRequests, hiddenRequests, 'Saving settings and overlay changes must keep a hidden preview disconnected');
        await previewPage.waitForFunction(() => !document.getElementById('preview-settings-save').disabled);
        assert.equal(await previewPage.locator('#preview-max-edge').inputValue(), '640');
        assert.equal(await previewPage.locator('#preview-quality').inputValue(), '70');
        await previewPage.locator('#preview-max-edge').selectOption('960');
        await previewPage.locator('#preview-quality').fill('82');
        await previewPage.waitForResponse(response => response.url().endsWith('/api/status'));
        assert.equal(await previewPage.locator('#preview-quality').inputValue(), '82');
        const beforePreviewSaveEnables = previewEnables;
        let releasePreviewSave;
        let enteredPreviewSave;
        const previewSaveGate = new Promise(resolve => { releasePreviewSave = resolve; });
        const previewSaveEntered = new Promise(resolve => { enteredPreviewSave = resolve; });
        await previewPage.route('**/api/preview/settings', async route => {
            if (route.request().method() !== 'POST') return route.fallback();
            enteredPreviewSave();
            await previewSaveGate;
            return route.fallback();
        });
        const previewSaveRequest = previewPage.waitForRequest(request => request.url().endsWith('/api/preview/settings') && request.method() === 'POST');
        await previewPage.locator('#preview-settings-save').click();
        await previewSaveEntered;
        assert(await previewPage.locator('#preview-settings-save').isDisabled());
        assert(await previewPage.locator('#preview-settings-reload').isDisabled());
        await previewPage.waitForResponse(response => response.url().endsWith('/api/status'));
        assert(await previewPage.locator('#preview-settings-save').isDisabled());
        releasePreviewSave();
        await previewPage.waitForFunction(() => document.getElementById('preview-settings-status').textContent.includes('saved and applied'));
        assert.deepEqual((await previewSaveRequest).postDataJSON(), { max_edge: 960, quality: 82 });
        assert.equal(previewRequests, hiddenRequests, 'Preview settings must not reopen a hidden stream');
        assert.equal(previewStarts, 1);
        assert.equal(previewStops, 0);
        assert.equal(previewEnables, beforePreviewSaveEnables, 'Preview settings must not consume pending autostart through Show camera');
        await previewPage.locator('#preview-quality').fill('35');
        await previewPage.locator('#preview-settings-reload').click();
        await previewPage.waitForFunction(() => document.getElementById('preview-quality').value === '82');
        await previewPage.reload();
        await previewPage.waitForFunction(() => document.getElementById('preview-quality').value === '82');
        assert.equal(await previewPage.locator('#preview-max-edge').inputValue(), '960');
        assert.equal(await previewPage.locator('#video-feed').getAttribute('src'), null);
        await previewPage.locator('#preview-quality').fill('101');
        await previewPage.locator('#preview-settings-save').click();
        assert.match(await previewPage.locator('#preview-settings-status').textContent(), /1.*100/);
        await previewPage.locator('#preview-quality').fill('82.5');
        await previewPage.locator('#preview-settings-save').click();
        assert.match(await previewPage.locator('#preview-settings-status').textContent(), /integer/i);
        await previewPage.locator('#preview-quality').fill('91');
        await previewPage.route('**/api/preview/settings', route => route.fulfill({ status: 503, json: { status: 'error', message: 'Could not save preview' } }));
        await previewPage.locator('#preview-settings-save').click();
        await previewPage.waitForFunction(() => document.getElementById('preview-settings-status').textContent.includes('Could not save preview'));
        assert.equal(await previewPage.locator('#preview-quality').inputValue(), '91');
        assert(await previewPage.locator('#preview-settings-save').isEnabled());
        await previewPage.locator('#btn-preview').click();
        await previewPage.waitForFunction(() => document.getElementById('btn-preview').getAttribute('aria-pressed') === 'true');
        assert.equal(previewStarts, 1);
        assert.equal(previewStops, 0);
        await previewPage.locator('#btn-stop').click();
        await previewPage.waitForFunction(() => document.getElementById('fps-display').textContent.includes('Camera only'));
        assert(await previewPage.locator('#video-feed').isVisible());
        assert.equal(previewStops, 1);
        assert(!(await previewPage.locator('#overlay-toggle').isChecked()));
        await previewPage.locator('#btn-preview').click();
        await previewPage.route('**/api/preview', route => route.fulfill({ status: 503, json: { status: 'error', message: 'Preview unavailable' } }));
        await previewPage.locator('#btn-preview').click();
        await previewPage.waitForFunction(() => document.getElementById('preview-status').textContent.includes('Preview unavailable'));
        assert.equal(await previewPage.locator('#video-feed').getAttribute('src'), null);

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

        const unsupportedPage = await newPage();
        unsupportedPage.on('pageerror', error => errors.push(error.message));
        let unsupportedSettings = {
            resolution: { width: 1280, height: 960 }, framerate: 25,
            awb_mode: 'auto', colour_gains: [1.5, 1.5],
            exposure_mode: 'auto', exposure_time_us: 20000, analogue_gain: 1,
        };
        await unsupportedPage.route('**/api/camera/settings', route => {
            if (route.request().method() === 'POST') unsupportedSettings = route.request().postDataJSON();
            return route.fulfill({ json: { status: 'ok', settings: unsupportedSettings, limits: {} } });
        });
        await unsupportedPage.goto(baseUrl);
        await unsupportedPage.waitForFunction(() => !document.getElementById('camera-save').disabled);
        assert(await unsupportedPage.locator('#camera-brightness').isDisabled());
        assert(await unsupportedPage.locator('#camera-noise-reduction').isDisabled());
        await unsupportedPage.locator('#camera-save').click();
        await unsupportedPage.waitForFunction(() => document.getElementById('camera-settings-status').textContent.includes('saved and applied'));
        for (const name of ['brightness', 'contrast', 'saturation', 'sharpness', 'exposure_value', 'noise_reduction']) {
            assert(!Object.hasOwn(unsupportedSettings, name), `Unsupported ${name} must be omitted from POST`);
        }

        const legacyPage = await newPage();
        legacyPage.on('pageerror', error => errors.push(error.message));
        await legacyPage.route('**/api/camera/settings', route => route.fulfill({ json: {
            status: 'ok', settings: {
                resolution: { width: 640, height: 480 }, framerate: 15,
                awb_mode: 'auto', colour_gains: [1.5, 1.5],
                exposure_mode: 'auto', exposure_time_us: 20000, analogue_gain: 1,
            },
        } }));
        await legacyPage.route('**/api/preview/settings', route => route.fulfill({ json: { settings: { max_edge: 800, quality: 80 } } }));
        await legacyPage.goto(baseUrl);
        await legacyPage.waitForFunction(() => !document.getElementById('camera-save').disabled);
        assert.equal(await legacyPage.locator('#camera-resolution').inputValue(), '640x480');
        assert.equal(await legacyPage.locator('#camera-framerate').inputValue(), '15');
        await legacyPage.locator('#camera-save').click();
        assert.match(await legacyPage.locator('#camera-settings-status').textContent(), /at least HD/);
        await legacyPage.waitForFunction(() => !document.getElementById('preview-settings-save').disabled);
        assert.equal(await legacyPage.locator('#preview-max-edge').inputValue(), '800');
        assert.equal(await legacyPage.locator('#preview-quality').inputValue(), '80');

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
        console.log('Browser UI checks passed: image controls/unsupported limits/manual EV, saved preview size/quality/persistence/validation/isolation, camera settings, independent viewing, recovery, recording, GPIO controls, 320–1200px layout.');
    } finally {
        await browser.close();
    }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
