import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const apiTarget = process.env.SWARMGPT_API_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    watch: {
      usePolling: true,
      interval: 1000
    },
    proxy: {
      "/api": {
        target: apiTarget,
        changeOrigin: true,
        ws: true
      }
    }
  }
});
