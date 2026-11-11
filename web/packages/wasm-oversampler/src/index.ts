/**
 * WebAssembly build of the production oversampler (tile-stratified error
 * diffusion). The C++ source is generated from the packaged CPU sampler by
 * tools/transform.py with the CUDA seed-hash convention, so the point set is
 * identical to the CUDA kernel's (see the tests for the measured parity).
 * The compiled module ships in prebuilt/ so a checkout runs without Emscripten;
 * build.sh regenerates it and records the input hashes that the tests verify.
 */
import type { Oversampler, OversamplerOptions } from "./types.js";

export type { Oversampler, OversamplerOptions } from "./types.js";

/** Default tile cap; mirrors the CUDA shared-memory limit of 32x32 cells. */
export const DEFAULT_TILE_CAP = 32;

interface EmscriptenModule {
  _gs_density_to_points(density: number, height: bigint, width: bigint, count: bigint, seed: bigint, tileCap: bigint, out: number): bigint;
  _malloc(bytes: number): number;
  _free(ptr: number): void;
  HEAPF32: Float32Array;
}

type ModuleFactory = (init?: { locateFile?: (path: string, prefix: string) => string }) => Promise<EmscriptenModule>;

const MAX_SEED = 0x7fffffff;

function validate(density: Float32Array, width: number, height: number, count: number, seed: number, tileCap: number): void {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) {
    throw new RangeError(`oversampler: width and height must be positive integers, got ${width}x${height}`);
  }
  if (density.length !== width * height) {
    throw new RangeError(`oversampler: density length ${density.length} != ${width}*${height}`);
  }
  if (!Number.isInteger(count) || count < 1) {
    throw new RangeError(`oversampler: count must be a positive integer, got ${count}`);
  }
  if (!Number.isInteger(seed) || seed < 0 || seed > MAX_SEED) {
    throw new RangeError(`oversampler: seed must be an integer in [0, 2^31-1], got ${seed}`);
  }
  if (!Number.isInteger(tileCap) || tileCap < 0) {
    throw new RangeError(`oversampler: tileCap must be a non-negative integer, got ${tileCap}`);
  }
}

export async function createOversampler(opts: OversamplerOptions = {}): Promise<Oversampler> {
  const moduleUrl = opts.moduleUrl ?? new URL("../prebuilt/oversampler.mjs", import.meta.url).href;
  const factory = (await import(/* @vite-ignore */ moduleUrl)).default as ModuleFactory;
  const init: Parameters<ModuleFactory>[0] = {};
  if (opts.wasmUrl) {
    const wasmUrl = opts.wasmUrl;
    init.locateFile = (path: string, prefix: string) => (path.endsWith(".wasm") ? wasmUrl : prefix + path);
  }
  const mod = await factory(init);
  let disposed = false;

  return {
    sample(density, width, height, count, seed, tileCap = DEFAULT_TILE_CAP): Float32Array {
      if (disposed) throw new Error("oversampler: disposed");
      validate(density, width, height, count, seed, tileCap);
      const densityPtr = mod._malloc(density.length * 4);
      const outPtr = mod._malloc(count * 2 * 4);
      try {
        // HEAPF32 must be re-read after every call that may grow memory.
        mod.HEAPF32.set(density, densityPtr >>> 2);
        const written = mod._gs_density_to_points(densityPtr, BigInt(height), BigInt(width), BigInt(count), BigInt(seed), BigInt(tileCap), outPtr);
        if (written !== BigInt(count)) {
          throw new Error(`oversampler: sampler returned ${written}, expected ${count}`);
        }
        return mod.HEAPF32.slice(outPtr >>> 2, (outPtr >>> 2) + count * 2);
      } finally {
        mod._free(outPtr);
        mod._free(densityPtr);
      }
    },
    dispose() {
      disposed = true;
    },
  };
}
