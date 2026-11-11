// Copy the runtime assets the demo loads at runtime into public/ (all gitignored):
//   public/ort/              ONNX Runtime Web wasm + glue
//   public/oversampler/      prebuilt Emscripten oversampler glue + wasm
//   public/samples/kodak/    the Kodak sample images (shared with the 2D demo)
// public/models/{fast,quality}/ hold the web bundles of the release (planar_*_web).
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const demo = path.resolve(here, "..");
const web = path.resolve(demo, "../..");
const copy = (src, dst) => { if (!fs.existsSync(src)) { console.warn(`[prepare-assets] missing ${src}`); return; } fs.mkdirSync(path.dirname(dst), { recursive: true }); fs.copyFileSync(src, dst); };

const ortDist = path.join(web, "node_modules/onnxruntime-web/dist");
for (const f of ["ort.webgpu.mjs", "ort.webgpu.min.mjs"]) copy(path.join(ortDist, f), path.join(demo, "public/ort", f));
for (const f of fs.readdirSync(ortDist).filter((f) => f.startsWith("ort-wasm-simd-threaded") && /\.(mjs|wasm)$/.test(f))) copy(path.join(ortDist, f), path.join(demo, "public/ort", f));
// The runtime the page downloads itself (for its progress bar) and hands to ONNX Runtime, with its
// size: a server that compresses the download sends no usable length.
const runtimeWasm = "ort-wasm-simd-threaded.asyncify.wasm";
fs.writeFileSync(path.join(demo, "public/ort/runtime.json"), JSON.stringify({ wasm: runtimeWasm, bytes: fs.statSync(path.join(ortDist, runtimeWasm)).size }) + "\n");
for (const f of ["oversampler.mjs", "oversampler.wasm"]) copy(path.join(web, "packages/wasm-oversampler/prebuilt", f), path.join(demo, "public/oversampler", f));
const kodak = path.join(web, "apps/demo/public/samples/kodak");
if (fs.existsSync(kodak)) {
  for (const f of fs.readdirSync(kodak).filter((f) => /\.(png|md)$/.test(f))) copy(path.join(kodak, f), path.join(demo, "public/samples/kodak", f));
  for (const f of fs.readdirSync(path.join(kodak, "thumbs"))) copy(path.join(kodak, "thumbs", f), path.join(demo, "public/samples/kodak/thumbs", f));
}
else console.warn(`[prepare-assets] missing ${kodak}`);
for (const kind of ["fast", "quality"]) {
  if (!fs.existsSync(path.join(demo, "public/models", kind, "planar_bundle.json"))) console.warn(`[prepare-assets] public/models/${kind}/planar_bundle.json missing: unpack the release's planar_${kind}_web bundle there`);
}
console.log("[prepare-assets] done");
