/** Engines (one per model, created on first use), their loading progress, and the image preparation. */
import * as ort from "onnxruntime-web/webgpu";
import { internals } from "@gaussifier/inference";
import { createOversampler } from "@gaussifier/wasm-oversampler";
import { FrontalFasterGsRasterizer, PlanarEngine, fetchBytes, loadPlanarBundle } from "@gaussifier/planar";

export type ModelKind = "fast" | "quality";

/** Loading progress: a fraction of the bytes, or null while the downloaded model is prepared. */
export type LoadProgress = (fraction: number | null, loadedBytes: number, totalBytes: number) => void;

const base = new URL("./", location.href);
const engines = new Map<ModelKind, Promise<PlanarEngine>>();

/** True when the page ships a bundle for this model. */
export async function hasModel(kind: ModelKind): Promise<boolean> {
  try {
    const r = await fetch(new URL(`models/${kind}/planar_bundle.json`, base));
    return r.ok;
  } catch {
    return false;
  }
}

/**
 * The ONNX Runtime wasm (27 MB), downloaded once by the page so its progress shows, then handed
 * to ONNX Runtime. Shared by both models.
 */
let runtime: { bytes: number; received: number; promise: Promise<Uint8Array> } | null = null;
function loadRuntime(onBytes: () => void): { bytes: number; received: () => number; promise: Promise<Uint8Array> } {
  if (!runtime) {
    const state = { bytes: 27_000_000, received: 0, promise: null as unknown as Promise<Uint8Array> };
    state.promise = (async () => {
      const info = await (await fetch(new URL("ort/runtime.json", base))).json() as { wasm: string; bytes: number };
      state.bytes = info.bytes;
      return fetchBytes(new URL(`ort/${info.wasm}`, base).href, (n) => { state.received = n; onBytes(); });
    })();
    runtime = state;
  }
  const r = runtime;
  return { bytes: r.bytes, received: () => r.received, promise: r.promise };
}

export function getEngine(kind: ModelKind, onProgress?: LoadProgress): Promise<PlanarEngine> {
  let engine = engines.get(kind);
  if (!engine) {
    engine = (async () => {
      if (!("gpu" in navigator)) throw new Error("WebGPU unavailable");
      internals.configureOrt({ wasmPaths: new URL("ort/", base).href, logLevel: "error" });
      let bundleLoaded = 0, bundleTotal = 13_000_000;
      const rt = loadRuntime(() => report());
      const runtimePending = !ort.env.wasm.wasmBinary;
      function report(): void {
        const loaded = (runtimePending ? Math.min(rt.received(), rt.bytes) : 0) + bundleLoaded;
        const total = (runtimePending ? rt.bytes : 0) + bundleTotal;
        onProgress?.(Math.min(1, loaded / Math.max(1, total)), loaded, total);
      }
      report();
      const [wasm, bundle] = await Promise.all([
        rt.promise,
        loadPlanarBundle(new URL(`models/${kind}/`, base).href, (loaded, total) => { bundleLoaded = loaded; bundleTotal = total; report(); }),
      ]);
      if (!ort.env.wasm.wasmBinary) ort.env.wasm.wasmBinary = wasm;
      onProgress?.(null, 0, 0);
      const oversampler = await createOversampler({
        moduleUrl: new URL("oversampler/oversampler.mjs", base).href,
        wasmUrl: new URL("oversampler/oversampler.wasm", base).href,
      });
      return PlanarEngine.create(bundle, oversampler, (device) => new FrontalFasterGsRasterizer(device));
    })();
    engine.catch(() => engines.delete(kind));
    engines.set(kind, engine);
  }
  return engine;
}

/**
 * The model's input size: the short side SHORT_SIDE (the scale the model was trained at), the long
 * side at most MAX_SIDE (very wide pictures end up with a shorter short side), both sides multiples
 * of SIZE_MULTIPLE.
 */
export const SHORT_SIDE = 512;
export const MAX_SIDE = 1024;
export const SIZE_MULTIPLE = 16;

/** The size a picture runs at, and the size it is scaled to before the centre trim. */
export function modelSize(width: number, height: number): { width: number; height: number; scaledWidth: number; scaledHeight: number; scale: number } {
  const s = Math.min(SHORT_SIDE / Math.min(width, height), MAX_SIDE / Math.max(width, height));
  const scaledWidth = Math.max(SIZE_MULTIPLE, Math.round(width * s));
  const scaledHeight = Math.max(SIZE_MULTIPLE, Math.round(height * s));
  return {
    width: scaledWidth - (scaledWidth % SIZE_MULTIPLE),
    height: scaledHeight - (scaledHeight % SIZE_MULTIPLE),
    scaledWidth, scaledHeight, scale: s,
  };
}

export interface PreparedImage {
  /** CHW f32 in [0, 1]. */
  chw: Float32Array;
  width: number;
  height: number;
  sourceWidth: number;
  sourceHeight: number;
  resized: boolean;
  cropped: boolean;
}

/**
 * The picture at its own aspect ratio, scaled to `modelSize` (short side SHORT_SIDE, long side at
 * most MAX_SIDE), then centre-cropped by at most SIZE_MULTIPLE - 1 pixels per side so both sides
 * divide by SIZE_MULTIPLE.
 */
export async function prepareImage(blob: Blob): Promise<PreparedImage> {
  const bitmap = await createImageBitmap(blob);
  const sw = bitmap.width, sh = bitmap.height;
  const size = modelSize(sw, sh);
  const { width, height } = size;
  const rw = size.scaledWidth, rh = size.scaledHeight, s = size.scale;
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("no 2D canvas");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, width, height);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  // Scale to rw x rh and keep the centre width x height.
  ctx.drawImage(bitmap, 0, 0, sw, sh, -Math.floor((rw - width) / 2), -Math.floor((rh - height) / 2), rw, rh);
  bitmap.close();
  const rgba = ctx.getImageData(0, 0, width, height).data;
  const hw = width * height;
  const chw = new Float32Array(3 * hw);
  for (let i = 0; i < hw; i++) {
    chw[i] = rgba[4 * i] / 255;
    chw[hw + i] = rgba[4 * i + 1] / 255;
    chw[2 * hw + i] = rgba[4 * i + 2] / 255;
  }
  return { chw, width, height, sourceWidth: sw, sourceHeight: sh, resized: s !== 1, cropped: rw !== width || rh !== height };
}
