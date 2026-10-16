import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './ui/tests',
  timeout: 30000,
  fullyParallel: false,
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:8080',
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: {
    command:
      'PYTHONPATH=src .venv/bin/python -m uvicorn ' +
      'carethread.api.app:app --host 127.0.0.1 --port 8080',
    url: 'http://127.0.0.1:8080/healthz',
    reuseExistingServer: true,
    cwd: '.',
    timeout: 60000,
  },
});
