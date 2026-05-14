import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    passWithNoTests: true,
    include: ['src/**/*.test.ts', 'src/**/*.spec.ts'],
    exclude: ['**/*.integration.test.ts', 'tests/integration/**'],
    coverage: {
      reporter: ['text', 'lcov']
    }
  }
});
