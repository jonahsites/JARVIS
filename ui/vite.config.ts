import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Bound to localhost only. Nothing here should be reachable off the machine.
    host: '127.0.0.1',
    strictPort: true,
    open: false,
  },
});
