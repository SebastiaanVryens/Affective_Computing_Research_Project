import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    port: 5173,
    // getUserMedia requires a secure context. localhost counts as one, so no
    // certificate is needed — but if you open the app from another device on
    // the network, the camera will silently fail without HTTPS.
    host: 'localhost',
  },
  build: {
    target: 'es2022',
    // three + face-api + tfjs is genuinely large; the default 500kB warning
    // fires on every build and trains you to ignore it.
    chunkSizeWarningLimit: 2400,
    rollupOptions: {
      output: {
        // Rolldown (Vite 8's bundler) only accepts the function form here; the
        // object shorthand throws at config time.
        manualChunks(id: string) {
          if (id.includes('node_modules/three')) return 'three';
          if (id.includes('@vladmandic/face-api')) return 'faceapi';
          return undefined;
        },
      },
    },
  },
});
