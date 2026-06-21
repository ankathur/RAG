import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In dev the SPA uses relative URLs (/health, /ingest, /ask) and Vite proxies
// them to the FastAPI backend, so there is no CORS to deal with locally.
// Override the backend location with VITE_API_TARGET if it isn't on :8000.
const target = process.env.VITE_API_TARGET || 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/health': target,
      '/ingest': target,
      '/ask': target,
    },
  },
})
