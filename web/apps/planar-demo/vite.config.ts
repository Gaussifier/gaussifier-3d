import { defineConfig } from "vite";

// onnxruntime-web stays outside the bundle: index.html maps "onnxruntime-web/webgpu" to ./ort/ort.webgpu.mjs
// through an import map, and scripts/prepare-assets.mjs copies the runtime files into public/ort/.
export default defineConfig({
  base: "./",
  build: { outDir: "dist", target: "es2022", chunkSizeWarningLimit: 4096, rollupOptions: { external: ["onnxruntime-web/webgpu", "onnxruntime-web"] } },
  optimizeDeps: { exclude: ["onnxruntime-web", "@playcanvas/splat-transform"] },
  server: { headers: { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" } },
});
