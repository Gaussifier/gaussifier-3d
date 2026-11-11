import { defineConfig } from "vite";

// onnxruntime-web stays outside the bundle: index.html maps "onnxruntime-web/webgpu" to ./ort/ort.webgpu.mjs
// through an import map, and scripts/prepare-assets.mjs copies the runtime files into public/ort/.
// Bundling it would also drag every ORT wasm variant into dist/assets.
export default defineConfig({
  base: "./",
  build: { outDir: "dist", target: "es2022", rollupOptions: { external: ["onnxruntime-web/webgpu", "onnxruntime-web"] } },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  server: { headers: { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" } },
});
