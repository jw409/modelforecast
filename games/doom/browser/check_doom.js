const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  
  // Capture ALL console output
  page.on('console', msg => console.log('LOG:', msg.text()));
  page.on('pageerror', error => console.log('PAGE ERROR:', error.message));
  
  await page.goto('http://localhost:9999/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(5000);
  
  // Start new game
  console.log('--- Starting new game ---');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(1000);
  await page.keyboard.press('Enter');
  await page.waitForTimeout(1000);
  await page.keyboard.press('Enter');
  await page.waitForTimeout(1000);  
  await page.keyboard.press('Enter');
  await page.waitForTimeout(3000);
  
  // Try moving
  console.log('--- Trying to move ---');
  await page.keyboard.press('ArrowUp');
  await page.waitForTimeout(2000);
  
  await page.screenshot({ path: '/tmp/doom_debug2.png' });
  console.log('Screenshot saved');
  
  await browser.close();
})();
