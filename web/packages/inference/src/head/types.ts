/** Correction-head backend contract: ORT session or the custom WGSL kernels. */
export interface HeadBackend {
  readonly kind: "ort" | "wgsl";
  /** Allocate or reuse per-size resources. */
  prepare(width: number, height: number): void;
  /**
   * Record the head into the caller's encoder, reading the [1,17,H,W] features buffer and
   * returning the [1,8,H,W] delta buffer, valid after the submit. Only the WGSL backend supports this.
   */
  record?(encoder: GPUCommandEncoder, features: GPUBuffer, width: number, height: number): GPUBuffer;
  /** Run the head and resolve with the delta buffer. The buffer stays valid until release(). */
  run(features: GPUBuffer, width: number, height: number): Promise<GPUBuffer>;
  /** Free the outputs of the last run(). */
  release(): void;
  dispose(): void;
}
