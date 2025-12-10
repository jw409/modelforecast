const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    bypassCSP: true
  });
  const page = await context.newPage();

  // Disable cache
  await page.route('**/*', route => route.continue());

  // Capture console output
  page.on('console', msg => console.log('LOG:', msg.text()));
  page.on('pageerror', error => console.log('PAGE ERROR:', error.message));

  // Add cache-busting query param
  await page.goto('http://localhost:9999/?nocache=' + Date.now(), { waitUntil: 'networkidle' });
  await page.waitForTimeout(5000);

  // Start new game
  console.log('--- Starting new game ---');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(500);
  await page.keyboard.press('Enter');
  await page.waitForTimeout(500);
  await page.keyboard.press('Enter');
  await page.waitForTimeout(500);
  await page.keyboard.press('Enter');
  await page.waitForTimeout(3000);

  // Screenshot
  await page.screenshot({ path: '/tmp/doom_fresh.png' });
  console.log('--- Screenshot saved ---');

  await browser.close();
})();
