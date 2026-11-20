import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The dev server proxies /api to the FastAPI backend, so no CORS setup is needed in dev.
// Set VITE_API_URL to call a backend on another origin instead.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    rollupOptions: {
      output: { manualChunks: { charts: ["recharts"], vendor: ["react", "react-dom", "react-router-dom", "axios"] } },
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": { target: process.env.VITE_PROXY_TARGET || "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
