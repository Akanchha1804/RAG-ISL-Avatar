import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

// Where the backend lives. Defaults to the documented port; override with
// ISL_BACKEND_URL when 8000 is occupied by another service.
const BACKEND = process.env.ISL_BACKEND_URL || 'http://localhost:8000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: BACKEND,
        changeOrigin: true,
        // Required for ws://<dev-host>/api/pipeline/ws; without it the dev
        // server answers the upgrade with 404 and the socket never opens.
        ws: true,
      },
      '/landmarks': {
        target: BACKEND,
        changeOrigin: true,
      },
      '/clips': {
        target: BACKEND,
        changeOrigin: true,
      },
      '/avatar': {
        target: BACKEND,
        changeOrigin: true,
      }
    }
  }
})
