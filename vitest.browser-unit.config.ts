import { defineConfig } from 'vitest/config';

/**
 * Browser-unit test config.
 *
 * Uses happy-dom to simulate browser globals (window, document, fetch, console).
 * The 'browser' resolve condition makes pino resolve to pino/browser automatically,
 * matching the same module that bundlers (Vite, webpack, esbuild) pick in real apps.
 *
 * Run:
 *   pnpm --filter @pixiagent/core test:browser-unit
 */
export default defineConfig({
  resolve: {
    // Activate the 'browser' condition in package.json#exports so packages like
    // @opentelemetry/* resolve to their browser-safe entry points.
    conditions: ['browser'],
  },
  test: {
    globals: true,
    environment: 'happy-dom',
    passWithNoTests: true,
    include: ['tests/browser/**/*.test.ts', 'tests/browser/**/*.spec.ts'],
    testTimeout: 10_000,
    coverage: {
      reporter: ['text', 'lcov'],
    },
  },
});
