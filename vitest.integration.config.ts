import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    passWithNoTests: true,
    include: ['tests/integration/**/*.test.ts', 'tests/integration/**/*.spec.ts'],
    setupFiles: ['tests/integration/setup.ts'],
    testTimeout: 30000,
    coverage: {
      reporter: ['text', 'lcov']
    }
  }
});