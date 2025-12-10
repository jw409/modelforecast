const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();

  page.on('console', msg => console.log('LOG:', msg.text()));

  await page.goto('http://localhost:9999/?t=' + Date.now(), { waitUntil: 'networkidle' });
  await page.waitForTimeout(4000);

  // Start game
  for (let i = 0; i < 4; i++) {
    await page.keyboard.press('Enter');
    await page.waitForTimeout(400);
  }
  await page.waitForTimeout(2000);

  // Test W key
  console.log('--- Pressing W ---');
  await page.keyboard.down('w');
  await page.waitForTimeout(1000);
  await page.keyboard.up('w');
  
  await page.screenshot({ path: '/tmp/doom_test_w.png' });

  await browser.close();
})();
