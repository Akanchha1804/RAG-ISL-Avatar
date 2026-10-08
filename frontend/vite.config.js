import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        // Required for ws://<dev-host>/api/pipeline/ws; without it the dev
        // server answers the upgrade with 404 and the socket never opens.
        ws: true,
      },
      '/landmarks': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/clips': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/avatar': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  }
})
