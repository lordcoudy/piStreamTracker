// Run against tests/preview_recovery_fixture.py. Uses the same Playwright
// module/executable environment variables as browser_ui.cjs.
const assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const baseUrl = process.env.PISTREAM_UI_TEST_URL || 'http://127.0.0.1:5099';

async function main() {
    const browser = await chromium.launch({
        headless: true,
        ...(process.env.PLAYWRIGHT_EXECUTABLE ? { executablePath: process.env.PLAYWRIGHT_EXECUTABLE } : {}),
    });
    try {
        const page = await browser.newPage();
        const errors = [];
        let viewerRequests = 0;
        page.on('pageerror', error => errors.push(error.message));
        page.on('request', request => {
            if (request.url().includes('/video_feed')) viewerRequests++;
        });
        const state = async () => (await page.request.get(baseUrl + '/__fixture/camera')).json();
        const power = online => page.request.post(baseUrl + '/__fixture/camera', { data: { online } });
        const imageHeight = height => page.waitForFunction(
            expected => document.getElementById('video-feed').naturalHeight === expected,
            height, { timeout: 15000 },
        );
        await page.goto(baseUrl);
        await page.locator('#btn-preview').click();
        await imageHeight(240);
        assert.equal(viewerRequests, 1);
        assert.equal((await state()).online, false);
        await power(true);
        await imageHeight(180);
        const originalSource = await page.locator('#video-feed').getAttribute('src');
        await power(false);
        await imageHeight(240);
        await power(true);
        await imageHeight(180);
        assert.equal(await page.locator('#video-feed').getAttribute('src'), originalSource);
        assert.equal(viewerRequests, 1, 'Recovery must keep one downstream browser stream');
        assert.equal((await state()).active, 1, 'A viewer must own only one upstream connection');
        await page.locator('#btn-preview').click();
        assert.equal(await page.locator('#video-feed').getAttribute('src'), null);
        for (let attempt = 0; attempt < 30 && (await state()).active; attempt++) {
            await page.waitForTimeout(100);
        }
        assert.equal((await state()).active, 0, 'Hide must close the upstream connection');
        const hiddenAttempts = (await state()).attempts;
        await page.waitForTimeout(3500);
        assert.equal((await state()).attempts, hiddenAttempts, 'A hidden viewer must not reconnect');
        const tracker = await (await page.request.get(baseUrl + '/api/status')).json();
        assert.equal(tracker.running, false);
        assert.equal(tracker.motor_connected, false);
        assert.deepEqual(errors, []);
        console.log('Preview recovery passed: initial offline, truncated-JPEG EOF, camera return, one upstream/viewer, Hide closes and stops retries.');
    } finally {
        await browser.close();
    }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
