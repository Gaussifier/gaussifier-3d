import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";

// Single-file ES module: `import "./gaussifier-viewer.js"` registers <gaussifier-viewer> and exports the API.
export default defineConfig({
  build: {
    lib: { entry: fileURLToPath(new URL("./src/bundle.ts", import.meta.url)), formats: ["es"], fileName: () => "gaussifier-viewer.js" },
    outDir: "bundle",
    emptyOutDir: true,
    target: "es2022",
    minify: false,
    sourcemap: true,
    rollupOptions: { external: [] },
  },
});
