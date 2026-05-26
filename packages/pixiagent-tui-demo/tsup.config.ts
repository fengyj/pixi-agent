import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/cli.tsx'],
  platform: 'node',
  target: 'node22',
  format: ['esm'],
  outDir: 'dist',
  clean: true,
  sourcemap: true,
});
