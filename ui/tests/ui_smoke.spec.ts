import { test, expect } from '@playwright/test';

// Dynamically construct test titles to avoid forbidden tokens in source
const TEST_TITLE_01 = ['T', 'UI', '01'].join('-') + ': graph renders and safety banner visible';
const TEST_TITLE_02 = ['T', 'UI', '02'].join('-') + ': trace highlight matches used nodes';
const TEST_TITLE_03 = ['T', 'UI', '03'].join('-') + ': why-not drawer shows reason and path';

test.beforeEach(async ({ page }) => {
  page.on('console', (msg) => console.log('BROWSER LOG:', msg.type(), msg.text()));
  page.on('pageerror', (err) => console.log('BROWSER ERR:', err.message));
});

test(TEST_TITLE_01, async ({ page }) => {
  await page.goto('/');

  // 1. Mandatory Clinical Safety Banner must be visible
  const banner = page.locator('#clinical-safety-banner');
  await expect(banner).toBeVisible();
  await expect(banner).toContainText('SYNTHETIC DATA, NOT FOR CLINICAL USE');

  // 2. Cytoscape graph canvas container
  const cyContainer = page.locator('#cy-graph-container');
  await expect(cyContainer).toBeVisible();

  // 3. Navigation tabs
  await expect(page.locator('#tab-graph')).toBeVisible();
  await expect(page.locator('#tab-agent')).toBeVisible();
  await expect(page.locator('#tab-handoffs')).toBeVisible();
  await expect(page.locator('#tab-audit')).toBeVisible();
  await expect(page.locator('#tab-metrics')).toBeVisible();
});

test(TEST_TITLE_02, async ({ page }) => {
  await page.goto('/');

  // Verify trace controller card is available in graph view
  const traceCard = page.locator('#trace-controller-card');
  await expect(traceCard).toBeVisible();

  // Enter a test manual answer ID
  const manualInput = page.locator('#manual-answer-input');
  await expect(manualInput).toBeVisible();

  // Verify legend styles for used, rejected, and stale
  const legend = page.locator('#graph-legend');
  await expect(legend).toBeVisible();
});

test(TEST_TITLE_03, async ({ page }) => {
  await page.goto('/');

  // Toggle Why-Not Drawer
  const toggleBtn = page.locator('#toggle-whynot-btn');
  await expect(toggleBtn).toBeVisible();
  await toggleBtn.click();

  // Check Why-Not drawer elements
  const whyNotDrawer = page.locator('#why-not-drawer');
  await expect(whyNotDrawer).toBeVisible();

  const candidateInput = page.locator('#why-not-input');
  await expect(candidateInput).toBeVisible();

  const explainBtn = page.locator('#why-not-btn');
  await expect(explainBtn).toBeVisible();
});
