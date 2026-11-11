/** Public contract of the inference engine: inputs, options, results, the bundle manifest. */
import type { Placement } from "./placement/types.js";

export interface PlanarImage {
  /** RGB planar float32 in [0,1], length 3 * width * height. */
  data: Float32Array;
  width: number;
  height: number;
  /**
   * True when `data` came from `prepareInput` for this engine's pad multiple: already padded and
   * clamped, so `run` uses it without copying. A prepared input is typically reused across runs.
   */
  prepared?: boolean;
}

export interface EngineOptions {
  /** Directory URL holding web_bundle.json and the ONNX files. */
  bundleUrl: string;
  /** Warm the sessions and pipelines at this size so the first user run is fast. */
  warmup?: { width: number; height: number };
  /** Adapter power preference handed to ORT, which owns the device. Default "high-performance". */
  powerPreference?: GPUPowerPreference;
  /** URL prefix of the ORT wasm files. Default "/node_modules/onnxruntime-web/dist/". */
  ortWasmPaths?: string;
  ortLogLevel?: "verbose" | "info" | "warning" | "error" | "fatal";
  /** Override the oversampler (default: @gaussifier/wasm-oversampler). */
  oversampler?: () => Promise<OversamplerLike> | OversamplerLike;
  /** Correction-head backend: "wgsl" uses the custom kernels when the bundle ships weights, "ort" the ONNX session. Default "auto" = wgsl when available. */
  headBackend?: "ort" | "wgsl" | "auto";
  /** Override the placement stage (default: ./placement/placement.js). */
  placement?: (device: GPUDevice) => Promise<Placement> | Placement;
}

export interface Engine {
  readonly device: GPUDevice;
  /** Human-readable adapter behind `device` (vendor and architecture, or the model when the browser exposes it). */
  readonly adapter: string;
  readonly manifest: BundleManifest;
  /** Head backends available in this engine; the first is the default. */
  readonly headKinds: Array<"ort" | "wgsl">;
  run(image: ImageBitmap | ImageData | PlanarImage, options?: RunOptions): Promise<RunResult>;
  dispose(): void;
}

/** Structural mirror of @gaussifier/wasm-oversampler's Oversampler. */
export interface OversamplerLike {
  sample(density: Float32Array, width: number, height: number, count: number, seed: number, tileCap?: number): Float32Array;
  dispose?(): void;
}

export interface RunOptions {
  /** K rendered states. Default from the bundle (7). Minimum 1. */
  states?: number;
  /** Overrides trunc(sum(rate)). */
  count?: number;
  /** Scales the adaptive count: N = trunc(sum(rate) * countScale). Ignored when `count` is given. Default 1. */
  countScale?: number;
  /** Default 12345. */
  seed?: number;
  /** Called after each rendered state. Forces a snapshot readback per state. */
  onState?: (k: number, snapshot: StateSnapshot) => void;
  /** Include the rendered image in snapshots. */
  includeRendered?: boolean;
  signal?: AbortSignal;
  /** Wait for the GPU after every stage so timings are completion times, not submission times. */
  profile?: boolean;
  /** Per-run head backend override. */
  headBackend?: "ort" | "wgsl";
  /** Read back the initializer state (state 0). Default true; false saves one snapshot readback. */
  includeInit?: boolean;
  /**
   * Keep every state: each one is copied to a staging buffer inside the loop without waiting and
   * read back once at the end, so the GPU never stalls between states (unlike `onState`, which
   * needs a round trip per state). The result's `snapshots` then holds states 1..K-1 (0..K-1
   * with includeInit), the last one with its rendered planes.
   */
  keepStates?: boolean;
  /** Read back the latent RGB with every snapshot. Default true; nothing that only displays needs it. */
  includeLatent?: boolean;
}

export interface StateSnapshot {
  k: number;
  /** Normalized xy, length 2N. */
  xy: Float32Array;
  /** Log-covariance channels [xx, xy, yy], length 3N. */
  cov: Float32Array;
  /** Renderer-ready weighted color, length 3N. */
  color: Float32Array;
  /** Latent (unweighted) RGB, length 3N; absent when the run was made with includeLatent: false. */
  rgbLatent?: Float32Array;
  /** Planar [3, H, W] rendered image, present when requested. */
  rendered?: Float32Array;
}

export interface RunResult {
  /** Padded frame size the coordinates refer to. */
  width: number;
  height: number;
  /** Original image size before edge padding. */
  crop: { w: number; h: number };
  /** Gaussians placed: the adaptive count, or `count`, or the scaled adaptive count. */
  n: number;
  /** The model's own count for this image, trunc(sum(rate)), whatever `n` was. */
  adaptiveCount: number;
  seed: number;
  states: number;
  density: Float32Array;
  weight: Float32Array;
  /** Initializer state, absent when the run was made with includeInit: false. */
  init?: StateSnapshot;
  final: StateSnapshot;
  /** Every kept state in order, present with `keepStates`; the last entry is `final`. */
  snapshots?: StateSnapshot[];
  timings: Record<string, number>;
}

export interface BundleManifest {
  schema_version: number;
  runtime_version: string;
  checkpoint_sha256: string;
  model: {
    correction_iterations: number;
    correction_predict_xy: boolean;
    correction_output_channels: number;
    correction_input_channels: number;
    correction_xy_step_pixels: number;
    decoder_density_aware: boolean;
  };
  files: Array<{ path: string; sha256: string; bytes: number; role: "forward_map" | "correction_head" | "correction_head_wgsl" | "correction_head_wgsl_layout"; precision: "f32"; shape: "dynamic" | "static512" | "any" }>;
  io: {
    forward_map: { input: string; outputs: string[] };
    correction_head: { input: string; output: string };
  };
  sampler: { oversample: number; merge_rounds: number; per_round_fraction: number; knn_k: number; lloyd_iters: number; tile_cap: number; seed_default: number };
  renderer: { tile: number; large_tile_threshold: number; alpha_cutoff: number; capacity: string };
  input: { pad_multiple: number; pad_mode: "edge" };
}

export class WebGpuUnavailableError extends Error {}
export class BundleMismatchError extends Error {}
export class CountContractError extends Error {}
export class AbortedError extends Error {}
