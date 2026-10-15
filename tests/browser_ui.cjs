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
        assert(await page.locator('#btn-stop').isDisabled());
        assert(await page.locator('#btn-record').isDisabled());
        assert(await page.locator('#recording-warning').isHidden());
        assert.equal((await (await page.request.get(baseUrl + '/__fixture/state')).json()).stop_calls, 2);

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
        assert.deepEqual(errors, []);
        console.log('Browser UI checks passed: explicit retry, warning deduplication, healthy/running recording, in-flight and cleanup controls, 390/900/1200px layout.');
    } finally {
        await browser.close();
    }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
