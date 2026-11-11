/** Oversampler contract: bit-exact port of the CUDA tile-stratified sampler. */
export interface Oversampler {
  /**
   * Sample `count` points from a row-major float32 density of size width*height.
   * Returns normalized xy, length count*2, in the CPU sampler's tile order.
   * `seed` is a 31-bit non-negative integer; `tileCap` defaults to 32 (CUDA rule).
   */
  sample(density: Float32Array, width: number, height: number, count: number, seed: number, tileCap?: number): Float32Array;
  dispose(): void;
}

export interface OversamplerOptions {
  /** URL or path of the .wasm; defaults to the file next to the glue module. */
  wasmUrl?: string;
  /** URL of the Emscripten glue module; defaults to ../prebuilt/oversampler.mjs relative to this module. */
  moduleUrl?: string;
}
