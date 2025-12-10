const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  // Capture console output
  page.on('console', msg => console.log('LOG:', msg.text()));
  page.on('pageerror', error => console.log('PAGE ERROR:', error.message));

  await page.goto('http://localhost:9999/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(5000);

  // Take screenshot before starting
  await page.screenshot({ path: '/tmp/doom_menu.png' });
  console.log('--- Menu screenshot saved ---');

  // Start new game - press Enter through menus
  console.log('--- Starting new game ---');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(500);
  await page.keyboard.press('Enter');
  await page.waitForTimeout(500);
  await page.keyboard.press('Enter');
  await page.waitForTimeout(500);
  await page.keyboard.press('Enter');
  await page.waitForTimeout(3000);

  // Take screenshot in game
  await page.screenshot({ path: '/tmp/doom_ingame2.png' });
  console.log('--- In-game screenshot saved ---');

  // Try moving forward
  console.log('--- Moving forward ---');
  await page.keyboard.down('ArrowUp');
  await page.waitForTimeout(2000);
  await page.keyboard.up('ArrowUp');

  // Final screenshot
  await page.screenshot({ path: '/tmp/doom_moved.png' });
  console.log('--- After movement screenshot saved ---');

  await browser.close();
})();
