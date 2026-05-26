import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/index.ts'],  // 只以 index 为入口
  format: ['esm'],
  outDir: 'dist',
  clean: true,
  bundle: true,             // 改成 true
  dts: false,
  sourcemap: true,
  target: 'node22',
  platform: 'node',
  outExtension: () => ({
    js: '.js',
    dts: '.d.ts',
  }),
});