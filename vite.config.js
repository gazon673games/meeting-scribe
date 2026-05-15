import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { DEFAULT_RENDERER_HOST, DEFAULT_RENDERER_PORT, parsePort } = require("./frontend/electron/renderer-url.cjs");

const devServerHost = process.env.VITE_DEV_SERVER_HOST || DEFAULT_RENDERER_HOST;
const devServerPort = parsePort(process.env.VITE_DEV_SERVER_PORT, DEFAULT_RENDERER_PORT);

export default defineConfig({
  root: "frontend/renderer",
  plugins: [react()],
  server: {
    host: devServerHost,
    port: devServerPort,
    strictPort: true
  },
  build: {
    outDir: "../../build/electron-renderer",
    emptyOutDir: true
  },
  test: {
    include: ["src/**/*.test.{js,jsx}", "../electron/**/*.test.js"],
    setupFiles: ["src/test/setup.js"],
    coverage: {
      provider: "v8",
      allowExternal: true,
      reporter: ["text", "html"],
      reportsDirectory: "../../coverage/frontend",
      include: ["src/**/*.{js,jsx}", "../electron/**/*.{cjs,js}", "frontend/electron/**/*.{cjs,js}"],
      exclude: [
        "src/**/*.test.{js,jsx}",
        "src/test/**",
        "../electron/**/__tests__/**",
        "frontend/electron/**/__tests__/**",
        "../electron/assets/**",
        "frontend/electron/assets/**",
        "../electron/run-electron.cjs"
      ]
    }
  }
});
