// Copy runtime assets the demo loads at runtime into public/ (all gitignored):
//   public/ort/          ONNX Runtime Web wasm + glue
//   public/oversampler/  prebuilt Emscripten oversampler glue + wasm
//   public/samples/      the sample .splat2d asset for embed.html, written from the bench_input_512
//                        goldens' final state (the Kodak images under samples/kodak are committed)
//   public/lib/          the single-file viewer module for embed.html
// public/models/ is written by scripts/export_web_bundle.py.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const demo = path.resolve(here, "..");
const web = path.resolve(demo, "../..");
const repo = path.resolve(web, "..");
const copy = (src, dst) => { if (!fs.existsSync(src)) { console.warn(`[prepare-assets] missing ${src}`); return; } fs.mkdirSync(path.dirname(dst), { recursive: true }); fs.copyFileSync(src, dst); };

const ortDist = path.join(web, "node_modules/onnxruntime-web/dist");
for (const f of ["ort.webgpu.mjs", "ort.webgpu.min.mjs"]) copy(path.join(ortDist, f), path.join(demo, "public/ort", f));
for (const f of fs.readdirSync(ortDist).filter((f) => f.startsWith("ort-wasm-simd-threaded") && /\.(mjs|wasm)$/.test(f))) copy(path.join(ortDist, f), path.join(demo, "public/ort", f));
for (const f of ["oversampler.mjs", "oversampler.wasm"]) copy(path.join(web, "packages/wasm-oversampler/prebuilt", f), path.join(demo, "public/oversampler", f));
copy(path.join(web, "packages/viewer/bundle/gaussifier-viewer.js"), path.join(demo, "public/lib/gaussifier-viewer.js"));
copy(path.join(web, "packages/viewer/bundle/gaussifier-viewer.js.map"), path.join(demo, "public/lib/gaussifier-viewer.js.map"));

/** The sample asset for embed.html: the final golden state of bench_input_512 in the Splat2D format. */
async function writeSampleAsset() {
  const golden = path.join(web, "fixtures/bench_input_512");
  const out = path.join(demo, "public/samples/bench_input_512.splat2d");
  const bin = (name) => { const b = fs.readFileSync(path.join(golden, name)); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
  if (!fs.existsSync(path.join(golden, "shapes.json"))) { console.warn(`[prepare-assets] ${golden} missing: run scripts/export_web_goldens.py; embed.html has no sample asset`); return; }
  const shapes = JSON.parse(fs.readFileSync(path.join(golden, "shapes.json"), "utf8"));
  const last = Math.max(...Object.keys(shapes).map((k) => /^state_(\d+)_xy$/.exec(k)).filter(Boolean).map((m) => Number(m[1])));
  // shapes.json is tracked; the .bin arrays next to it are not (scripts/export_web_goldens.py writes them).
  if (!fs.existsSync(path.join(golden, `state_${last}_xy.bin`))) { console.warn(`[prepare-assets] ${golden} has no .bin goldens: run scripts/export_web_goldens.py; embed.html has no sample asset`); return; }
  const [, H, W] = shapes[`state_${last}_rendered`].shape;
  let viewer;
  try { viewer = await import("@gaussifier/viewer"); } catch { console.warn("[prepare-assets] @gaussifier/viewer is not built (run `npm run build` first); embed.html has no sample asset"); return; }
  const xy = bin(`state_${last}_xy.bin`), cov = bin(`state_${last}_cov.bin`), color = bin(`state_${last}_color.bin`);
  const n = xy.length / 2;
  const { scale, rotation } = viewer.covToScaleRotation(cov, n);
  const blob = viewer.sceneToSplat2d({ width: W, height: H, states: [{ xy, cov, scale, rotation, color, label: `state ${last}` }] });
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, new Uint8Array(await blob.arrayBuffer()));
  console.log(`[prepare-assets] wrote ${path.relative(demo, out)} (${n} Gaussians, ${W}x${H}, state ${last})`);
}
await writeSampleAsset();
if (!fs.existsSync(path.join(demo, "public/models/web_bundle.json"))) console.warn("[prepare-assets] public/models/web_bundle.json missing: run scripts/export_web_bundle.py --out-dir web/apps/demo/public/models");
console.log("[prepare-assets] done");
