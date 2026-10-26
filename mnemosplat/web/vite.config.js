import { defineConfig } from 'vite';

// The splat sorter uses SharedArrayBuffer when the page is cross-origin isolated.
// These headers make `crossOriginIsolated === true` in dev and preview. The FastAPI
// server sets the same headers in production (see mns/api.py).
const isolation = {
  name: 'mnemosplat-cross-origin-isolation',
  configureServer(server) {
    server.middlewares.use((_req, res, next) => {
      res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
      res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
      res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
      next();
    });
  },
  configurePreviewServer(server) {
    server.middlewares.use((_req, res, next) => {
      res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
      res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
      res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
      next();
    });
  },
};

export default defineConfig({
  base: '/',
  plugins: [isolation],
  server: {
    port: 5173,
    proxy: {
      // `mnemosplat serve` owns the assets and the API; Vite is only the dev shell.
      '/assets': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    // The demo's splats live at /assets, so Vite's hashed bundles must not collide.
    assetsDir: 'static',
    sourcemap: false,
    target: 'es2020',
  },
});
