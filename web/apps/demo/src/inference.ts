import { createEngine, prepareInput, type Engine, type PreparedInput, type RunResult } from "@gaussifier/inference";
import { createOversampler } from "@gaussifier/wasm-oversampler";

const base = (rel: string) => new URL(rel, document.baseURI).href;
let enginePromise: Promise<Engine> | null = null;

/**
 * The engine is created once per page; the bundle and runtime files live next to index.html.
 * The warmup compiles every pipeline at the sample size, so the first click does not pay for it.
 */
export function getEngine(): Promise<Engine> {
  enginePromise ??= createEngine({
    bundleUrl: base("models/"),
    ortWasmPaths: base("ort/"),
    oversampler: () => createOversampler({ moduleUrl: base("oversampler/oversampler.mjs"), wasmUrl: base("oversampler/oversampler.wasm") }),
    warmup: { width: 512, height: 512 },
  });
  return enginePromise;
}

/**
 * What the adapter ORT chose means for run time. Chrome reports vendor and architecture, not the
 * model: "fast" is a GPU that runs a 512² image well under a second, "slow" an integrated Intel
 * GPU (a few seconds), "software" a CPU renderer such as SwiftShader (minutes).
 */
export type GpuSpeed = "fast" | "slow" | "software" | "unknown";
export function gpuSpeed(adapter: string): GpuSpeed {
  const a = adapter.toLowerCase();
  if (/swiftshader|llvmpipe|lavapipe|software|\bwarp\b|cpu/.test(a)) return "software";
  if (/nvidia|amd|radeon|apple|qualcomm|adreno|\barm\b|mali/.test(a)) return "fast";
  if (/intel|integrated|iris|uhd|gen-?\d/.test(a)) return "slow";
  return "unknown";
}

export interface InferenceResult { run: RunResult; adapter: string }

/**
 * The longest side the demo feeds the engine. Larger photos are downscaled on decode: the model
 * was trained on crops of this scale, and a 12-megapixel phone picture would otherwise take
 * many seconds even on a discrete GPU and exhaust integrated ones.
 */
export const MAX_IMAGE_SIDE = 768;

/** The size a picked image is decoded at: unchanged up to MAX_IMAGE_SIDE, else scaled to fit it. */
export function decodedSize(width: number, height: number): { width: number; height: number; resized: boolean } {
  const longest = Math.max(width, height);
  if (longest <= MAX_IMAGE_SIDE) return { width, height, resized: false };
  const s = MAX_IMAGE_SIDE / longest;
  return { width: Math.max(1, Math.round(width * s)), height: Math.max(1, Math.round(height * s)), resized: true };
}

/** Decode a picked file, downscaling to MAX_IMAGE_SIDE when it is larger. The caller closes the bitmap. */
export async function decodeImage(file: File): Promise<ImageBitmap> {
  const probe = await createImageBitmap(file);
  const size = decodedSize(probe.width, probe.height);
  if (!size.resized) return probe;
  try {
    return await createImageBitmap(probe, { resizeWidth: size.width, resizeHeight: size.height, resizeQuality: "high" });
  } finally {
    probe.close();
  }
}

/** Convert a picked image once: planar float32, edge padded to the bundle's multiple. Reused across runs. */
export async function prepareImage(image: ImageBitmap): Promise<PreparedInput> {
  const engine = await getEngine();
  return prepareInput(image, engine.manifest.input.pad_multiple);
}

/**
 * Run the production profile on a prepared image with K rendered states. Every state after the
 * initializer is kept for the scrubber; they are copied out inside the loop and read back once
 * at the end, so the GPU never waits on the page. `countScale` multiplies the adaptive
 * trunc(sum(rate)); 1 is the model's own count.
 */
export async function runInference(prepared: PreparedInput, states: number, countScale = 1): Promise<InferenceResult> {
  const engine = await getEngine();
  const run = await engine.run({ data: prepared.planar, width: prepared.width, height: prepared.height, prepared: true }, { states, countScale, includeRendered: false, includeInit: false, keepStates: true, includeLatent: false });
  return { run, adapter: engine.adapter };
}
