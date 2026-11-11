/**
 * WebGPU port of the Faster-GS rasterizer forward pass; see shaders.ts for the
 * kernels and the one documented difference (the order of primitives at bit-identical depth).
 *
 * `FasterGsRasterizer.record` records the whole forward pass into a caller-supplied command
 * encoder: one buffer clear and one compute pass of forty dispatches, with no CPU readback.
 * Instance-level dispatches are sized from the instance capacity and their kernels stop at the
 * count computed on the GPU (indirect dispatches cost about 0.13 ms each in Chromium, which
 * validates their arguments with extra passes). The instance list has a fixed capacity; when a
 * render needs more, the overflow is recorded in the counters buffer and `readStats` reports it
 * (`render` retries with a larger capacity).
 *
 * `recordBackward` records the backward pass of the last colour render (backward_shaders.ts):
 * gradients of the five splat tensors from dL/d(image), written or accumulated. Each instance
 * owns all of its buffers; instances share only the compiled pipelines (`FasterGsKernels`).
 */
import { Kernel, createStorageBuffer, readBuffer, workgroups, type KernelSpec } from "@gaussifier/wgsl";
import {
  BLEND_THREADS, COLORS_WGSL, COUNTER, CREATE_INSTANCES_WGSL, OFFSETS_REDUCE_WGSL, OFFSETS_SCAN_WGSL,
  PREPROCESS_WGSL, PROJ_BYTES, RADIX_COUNT_WGSL, RADIX_SCAN_WGSL, RANGES_WGSL, SCAN_BLOCK, SLOT_N,
  SORT_BLOCK, SORT_RADIX, TILE_H, TILE_W, blendWgsl, packRasterParams, radixScatterWgsl,
} from "./shaders.js";
import {
  ACC_STRIDE, BLEND_BACKWARD_WGSL, BUCKET_DISPATCH_X, BUCKET_INFO_WGSL, BUCKET_SCAN_WGSL, FINALIZE_BACKWARD_WGSL,
  PREPROCESS_BACKWARD_WGSL, SNAPSHOT_WGSL,
} from "./backward_shaders.js";

const RO = "read-only-storage" as const;
const RW = "storage" as const;
const U = "uniform" as const;

const SPECS = {
  preprocess: { code: PREPROCESS_WGSL, entryPoint: "preprocess", bindings: [U, RO, RO, RO, RO, RW, RW, RW], label: "fgs_preprocess" },
  colors: { code: COLORS_WGSL, entryPoint: "colors_main", bindings: [U, RO, RW], label: "fgs_colors" },
  radixCount: { code: RADIX_COUNT_WGSL, entryPoint: "radix_count", bindings: [U, U, RO, RO, RW], label: "fgs_radix_count" },
  radixScan: { code: RADIX_SCAN_WGSL, entryPoint: "radix_scan", bindings: [U, U, RO, RW], label: "fgs_radix_scan" },
  radixScatterKV: { code: radixScatterWgsl(true), entryPoint: "radix_scatter", bindings: [U, U, RO, RO, RO, RO, RW, RW], label: "fgs_radix_scatter_kv" },
  radixScatterK: { code: radixScatterWgsl(false), entryPoint: "radix_scatter", bindings: [U, U, RO, RO, RO, RW], label: "fgs_radix_scatter_k" },
  offsetsReduce: { code: OFFSETS_REDUCE_WGSL, entryPoint: "offsets_reduce", bindings: [U, RO, RO, RW], label: "fgs_offsets_reduce" },
  offsetsScan: { code: OFFSETS_SCAN_WGSL, entryPoint: "offsets_scan", bindings: [U, RW, RW], label: "fgs_offsets_scan" },
  createInstances: { code: CREATE_INSTANCES_WGSL, entryPoint: "create_instances", bindings: [U, RO, RO, RO, RO, RW], label: "fgs_create_instances" },
  ranges: { code: RANGES_WGSL, entryPoint: "tile_ranges", bindings: [U, RO, RO, RO, RW, RW], label: "fgs_ranges" },
  blend: { code: blendWgsl(false), entryPoint: "blend", bindings: [U, RO, RO, RO, RO, RW, RW, RW], label: "fgs_blend" },
  blendTieMask: { code: blendWgsl(true), entryPoint: "blend", bindings: [U, RO, RO, RO, RO, RO, RO, RW, RW], label: "fgs_blend_tiemask" },
  bucketScan: { code: BUCKET_SCAN_WGSL, entryPoint: "bucket_scan", bindings: [U, U, RO, RW, RW, RW], label: "fgs_bucket_scan" },
  bucketInfo: { code: BUCKET_INFO_WGSL, entryPoint: "bucket_info", bindings: [U, U, RO, RO, RW], label: "fgs_bucket_info" },
  snapshot: { code: SNAPSHOT_WGSL, entryPoint: "snapshot", bindings: [U, U, RO, RO, RO, RO, RO, RW, RW], label: "fgs_snapshot" },
  blendBackward: { code: BLEND_BACKWARD_WGSL, entryPoint: "blend_backward", bindings: [U, U, RO, RO, RO, RO, RO, RO, RO, RW], label: "fgs_blend_backward" },
  preprocessBackward: { code: PREPROCESS_BACKWARD_WGSL, entryPoint: "preprocess_backward", bindings: [U, U, RO, RO, RO, RO, RO, RW, RW, RW], label: "fgs_preprocess_backward" },
  finalizeBackward: { code: FINALIZE_BACKWARD_WGSL, entryPoint: "finalize_backward", bindings: [U, U, RO, RO, RW, RW], label: "fgs_finalize_backward" },
} satisfies Record<string, KernelSpec>;

const BACKWARD_KERNELS: KernelName[] = ["bucketScan", "bucketInfo", "snapshot", "blendBackward", "preprocessBackward", "finalizeBackward"];

type KernelName = keyof typeof SPECS;

/** Compiled pipelines, independent of sizes. Create once per device and share between rasterizers. */
export class FasterGsKernels {
  readonly k: Record<KernelName, Kernel>;
  constructor(readonly device: GPUDevice) {
    const k = {} as Record<KernelName, Kernel>;
    for (const [name, spec] of Object.entries(SPECS)) k[name as KernelName] = Kernel.create(device, spec);
    this.k = k;
  }
}

/** Pinhole camera in the Faster-GS argument convention. */
export interface FasterGsCamera {
  /** World-to-camera matrix, 16 floats, row major; rows 0..2 are used. */
  w2c: ArrayLike<number>;
  /** Camera centre in world space (used only by SH degree > 0, which this port does not implement). */
  position: ArrayLike<number>;
  width: number;
  height: number;
  focalX: number;
  focalY: number;
  centerX: number;
  centerY: number;
  near: number;
  far: number;
}

/** The camera settings of a planar bundle (`meta.frontal`, the Python `CameraSettings`). */
export interface CameraSettingsJson {
  w2c: number[];
  position: number[];
  width: number;
  height: number;
  focal_x: number;
  focal_y: number;
  center_x: number;
  center_y: number;
  near: number;
  far: number;
}

export function cameraFromSettings(s: CameraSettingsJson): FasterGsCamera {
  return {
    w2c: s.w2c, position: s.position, width: s.width, height: s.height,
    focalX: s.focal_x, focalY: s.focal_y, centerX: s.center_x, centerY: s.center_y, near: s.near, far: s.far,
  };
}

/**
 * The five splat tensors, float32, contiguous, in the layouts of the Python call:
 * means [N, 3]; logScales [N, 3] (log of the standard deviations, exponentiated in the kernel);
 * quats [N, 4] as (w, x, y, z), normalised in the kernel; opacityLogits [N] or [N, 1] (pre-sigmoid);
 * sh0 [N, 1, 3] (degree-0 SH coefficients).
 */
export interface SplatBuffers {
  means: GPUBuffer;
  logScales: GPUBuffer;
  quats: GPUBuffer;
  opacityLogits: GPUBuffer;
  sh0: GPUBuffer;
}

/**
 * Gradient outputs of `recordBackward`, float32, in the layouts of the Faster-GS `_C.backward`
 * outputs: means (N, 3); scales (N, 3), w.r.t. the log scales; quats (N, 4), w.r.t. the raw
 * (unnormalised) quaternions; opacity (N, 1), w.r.t. the logits; sh0 (N, 1, 3).
 */
export interface GradientBuffers {
  means: GPUBuffer;
  scales: GPUBuffer;
  quats: GPUBuffer;
  opacity: GPUBuffer;
  sh0: GPUBuffer;
}

export interface BackwardOptions {
  /** Add into the existing contents of the gradient buffers instead of overwriting them. */
  accumulate: boolean;
}

export interface RasterStats {
  /** Tile instances the last render needed. */
  instances: number;
  /** Instance capacity of the buffers. */
  capacity: number;
  /**
   * True when instances > capacity, or when a backward pass ran out of bucket space: that render
   * or gradient is incomplete. `ensureInstanceCapacity(1.25 * instances)` grows both.
   */
  overflow: boolean;
}

/**
 * Forward intermediates of the last `record`, kept for a backward pass. Primitive-indexed arrays
 * have N entries; rank-indexed ones are in depth order; instance arrays have `instances` entries.
 */
export interface ForwardIntermediates {
  /** Proj[N] (32 bytes): mean2d vec2f, bounds vec2u (x0 | x1 << 16, y0 | y1 << 16), conic.xx, conic.xy, conic.yy, opacity. Valid where touched > 0. */
  proj: GPUBuffer;
  /** vec4f[N]: 0.5 + C0 * sh0, unclamped (blending uses max(c, 0)), w unused. */
  colors: GPUBuffer;
  /** u32[N]: tiles each primitive contributes to; 0 for culled primitives. */
  touched: GPUBuffer;
  /** u32[N] by depth rank: depth as float bits (0xffffffff for culled primitives, which sort last). */
  depthKeysSorted: GPUBuffer;
  /** u32[N] by depth rank: primitive index. */
  sortedIndices: GPUBuffer;
  /** u32[instances]: tile << rankBits | depth rank, sorted by tile then depth. */
  instanceKeys: GPUBuffer;
  /** u32[instances]: primitive index of each sorted instance (Faster-GS instance_primitive_indices). */
  instancePrimitives: GPUBuffer;
  /** u32[2 * tiles]: [start, end) of every tile's instances, tiles row major. */
  tileRanges: GPUBuffer;
  /** f32[H * W]: final transmittance per pixel. */
  finalTransmittance: GPUBuffer;
  /** u32[H * W]: 1 + index (within the tile's range) of the last blended primitive, 0 if none. */
  nProcessed: GPUBuffer;
  /** u32[4]: instances, instances clamped to the capacity, overflow flag, unused. */
  counters: GPUBuffer;
  rankBits: number;
  gridW: number;
  gridH: number;
}

export interface FasterGsRasterizerOptions {
  kernels?: FasterGsKernels;
  /** Initial instance capacity per primitive (default 16; Faster-GS scenes here use about 4). */
  instancesPerPoint?: number;
  /** Floor of the instance capacity (default 65536). */
  minInstanceCapacity?: number;
  /** Tile instances per primitive assumed when the backward buffers are first sized (default 5). */
  backwardInstancesPerPoint?: number;
}

export class FasterGsRasterizer {
  readonly kernels: FasterGsKernels;
  private readonly k: Record<KernelName, Kernel>;
  private readonly instancesPerPoint: number;
  private readonly minInstanceCapacity: number;
  private readonly backwardInstancesPerPoint: number;
  private nCap = 0;
  private mCap = 0;
  private pixelCap = 0;
  private tileCap = 0;
  private buf: Record<string, GPUBuffer> = {};
  private readonly counters: GPUBuffer;
  private readonly depthPass: GPUBuffer[] = [];
  private readonly tilePass: GPUBuffer[] = [];
  private readonly paramCache = new Map<string, GPUBuffer>();
  private last: { params: GPUBuffer; n: number; width: number; height: number; gridW: number; gridH: number; rankBits: number; instanceKeys: GPUBuffer; splats: SplatBuffers } | null = null;
  // backward pass: allocated on first use, sized by primitive, tile and pixel counts and buckets
  private bw: { acc: GPUBuffer; tileBuckets: GPUBuffer; bucketInfo: GPUBuffer; snapshots: GPUBuffer; pixState: GPUBuffer; params: GPUBuffer[] } | null = null;
  private bwCaps = { n: 0, tiles: 0, pixels: 0, buckets: 0 };

  constructor(readonly device: GPUDevice, opts: FasterGsRasterizerOptions = {}) {
    this.kernels = opts.kernels ?? new FasterGsKernels(device);
    this.k = this.kernels.k;
    this.instancesPerPoint = opts.instancesPerPoint ?? 16;
    this.minInstanceCapacity = opts.minInstanceCapacity ?? 65536;
    this.backwardInstancesPerPoint = opts.backwardInstancesPerPoint ?? 5;
    this.counters = createStorageBuffer(device, 16, "fgs-counters");
    const sortParams = (pass: number, slot: number, rankBase: number, implicit: number) => {
      const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true, label: "fgs-sort-params" });
      new Uint32Array(b.getMappedRange()).set([pass, slot, rankBase, implicit]);
      b.unmap();
      return b;
    };
    for (let p = 0; p < 8; p++) this.depthPass.push(sortParams(p, SLOT_N, 0, p === 0 ? 1 : 0));
    for (let p = 0; p < 8; p++) this.tilePass.push(sortParams(p, COUNTER.sortCount, 1, 0));
  }

  /** Current instance capacity. */
  get instanceCapacity(): number { return this.mCap; }

  /**
   * Allocate (or grow) the per-primitive buffers for n primitives, and the instance buffers to
   * `instancesPerPoint` per primitive. Growing replaces buffers, so it must not happen between two
   * renders recorded into the same command buffer; call this up front when the counts vary.
   */
  ensurePrimitiveCapacity(n: number): void {
    if (n <= this.nCap && this.nCap > 0) return;
    const cap = Math.max(1024, Math.ceil(n * 1.25));
    for (const name of ["proj", "colors", "colorsAlt", "touched", "keysA", "keysB", "valsA", "valsB", "blockSums"]) this.buf[name]?.destroy();
    const d = this.device;
    this.buf.proj = createStorageBuffer(d, cap * PROJ_BYTES, "fgs-proj");
    this.buf.colors = createStorageBuffer(d, cap * 16, "fgs-colors");
    this.buf.colorsAlt = createStorageBuffer(d, cap * 16, "fgs-colors-alt");
    this.buf.touched = createStorageBuffer(d, cap * 4, "fgs-touched");
    this.buf.keysA = createStorageBuffer(d, cap * 4, "fgs-depth-keys-a");
    this.buf.keysB = createStorageBuffer(d, cap * 4, "fgs-depth-keys-b");
    this.buf.valsA = createStorageBuffer(d, cap * 4, "fgs-depth-vals-a");
    this.buf.valsB = createStorageBuffer(d, cap * 4, "fgs-depth-vals-b");
    this.buf.blockSums = createStorageBuffer(d, Math.ceil(cap / SCAN_BLOCK) * 4, "fgs-block-sums");
    this.nCap = cap;
    this.resizeBlockCounts();
    this.ensureInstanceCapacity(Math.min(Math.max(this.minInstanceCapacity, this.instancesPerPoint * cap), MAX_WORKGROUPS * 256));
    this.invalidate();
  }

  /** Allocate (or grow) the instance buffers to hold at least m tile instances. */
  ensureInstanceCapacity(m: number): void {
    if (this.bw) this.ensureBackward(this.bwCaps.n, this.bwCaps.tiles, this.bwCaps.pixels, Math.ceil(m / 32) + this.bwCaps.tiles);
    if (m <= this.mCap) return;
    const cap = Math.min(Math.max(this.minInstanceCapacity, Math.ceil(m)), MAX_WORKGROUPS * 256);
    if (cap < m) throw new Error(`FasterGsRasterizer: ${m} instances exceed the supported maximum ${cap}`);
    for (const name of ["instKeysA", "instKeysB", "instPrim"]) this.buf[name]?.destroy();
    this.buf.instKeysA = createStorageBuffer(this.device, cap * 4, "fgs-inst-keys-a");
    this.buf.instKeysB = createStorageBuffer(this.device, cap * 4, "fgs-inst-keys-b");
    this.buf.instPrim = createStorageBuffer(this.device, cap * 4, "fgs-inst-prims");
    this.mCap = cap;
    this.resizeBlockCounts();
    this.invalidate();
  }

  private ensureImage(width: number, height: number, tiles: number): void {
    if (width * height > this.pixelCap) {
      this.buf.finalT?.destroy(); this.buf.nProcessed?.destroy();
      this.pixelCap = width * height;
      this.buf.finalT = createStorageBuffer(this.device, this.pixelCap * 4, "fgs-final-t");
      this.buf.nProcessed = createStorageBuffer(this.device, this.pixelCap * 4, "fgs-n-processed");
      this.invalidate();
    }
    if (tiles > this.tileCap) {
      this.buf.ranges?.destroy();
      this.tileCap = tiles;
      this.buf.ranges = createStorageBuffer(this.device, tiles * 8, "fgs-tile-ranges");
      this.invalidate();
    }
  }

  private resizeBlockCounts(): void {
    const blocks = Math.max(Math.ceil(this.nCap / SORT_BLOCK), Math.ceil(this.mCap / SORT_BLOCK), 1);
    this.buf.blockCounts?.destroy();
    this.buf.blockCounts = createStorageBuffer(this.device, SORT_RADIX * blocks * 4, "fgs-block-counts");
  }

  private invalidate(): void {
    for (const kernel of Object.values(this.k)) kernel.clearCache();
    for (const b of this.paramCache.values()) b.destroy();
    this.paramCache.clear();
    this.last = null;
  }

  private paramsFor(camera: FasterGsCamera, n: number, background: ArrayLike<number>, gridW: number, gridH: number, rankBits: number): GPUBuffer {
    const data = packRasterParams({
      w2c: camera.w2c, position: camera.position, background, n, width: camera.width, height: camera.height,
      gridW, gridH, rankBits, capacity: this.mCap, focalX: camera.focalX, focalY: camera.focalY,
      centerX: camera.centerX, centerY: camera.centerY, near: camera.near, far: camera.far,
    });
    const key = Array.from(new Uint32Array(data)).join(",");
    let b = this.paramCache.get(key);
    if (b) { this.paramCache.delete(key); this.paramCache.set(key, b); return b; }
    // One uniform buffer per distinct setting, so renders with different cameras or counts can be
    // recorded into one encoder; the least recently used entries beyond 32 are dropped.
    b = this.device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true, label: "fgs-params" });
    new Uint8Array(b.getMappedRange()).set(new Uint8Array(data));
    b.unmap();
    this.paramCache.set(key, b);
    if (this.paramCache.size > 32) { const [oldKey, old] = this.paramCache.entries().next().value!; this.paramCache.delete(oldKey); old.destroy(); }
    return b;
  }

  /**
   * Record the forward pass: renders `n` splats into `output`, f32 CHW [3, height, width],
   * without clamping (the `_C.forward` image and the `_C.inference` image with to_chw = true,
   * clamp = false are the same function of their inputs). Allocates on first use and grows when
   * n or the image outgrow the buffers (see `ensurePrimitiveCapacity`); an instance overflow is
   * only visible through `readStats` after the command buffer ran.
   */
  record(encoder: GPUCommandEncoder, splats: SplatBuffers, n: number, camera: FasterGsCamera, output: GPUBuffer, background: ArrayLike<number> = [0, 0, 0]): void {
    const { width, height } = camera;
    if (output.size < 3 * width * height * 4) throw new Error("FasterGsRasterizer: output buffer too small");
    const gridW = Math.ceil(width / TILE_W);
    const gridH = Math.ceil(height / TILE_H);
    const tiles = gridW * gridH;
    const rankBits = n > 1 ? 32 - Math.clz32(n - 1) : 1;
    const tileBits = 32 - Math.clz32(tiles);
    if (rankBits + tileBits > 32) throw new Error(`FasterGsRasterizer: ${n} primitives on ${tiles} tiles exceed the 32-bit instance key`);
    if (gridW * TILE_W > 0xffff || gridH * TILE_H > 0xffff) throw new Error("FasterGsRasterizer: image too large for 16-bit bounds");
    this.ensurePrimitiveCapacity(n);
    this.ensureImage(width, height, tiles);
    const params = this.paramsFor(camera, n, background, gridW, gridH, rankBits);
    const b = this.buf;
    const k = this.k;
    encoder.clearBuffer(b.ranges, 0, tiles * 8);
    const pass = encoder.beginComputePass({ label: "fgs-forward" });
    k.preprocess.dispatch(pass, [params, splats.means, splats.logScales, splats.quats, splats.opacityLogits, b.proj, b.keysA, b.touched], workgroups(n, 128));
    k.colors.dispatch(pass, [params, splats.sh0, b.colors], workgroups(n, 256));
    // depth sort: stable LSD radix sort of the 32-bit depth keys, values start as the index
    const depthBlocks = workgroups(n, SORT_BLOCK);
    for (let p = 0; p < 8; p++) {
      const [kin, vin, kout, vout] = p % 2 === 0 ? [b.keysA, b.valsA, b.keysB, b.valsB] : [b.keysB, b.valsB, b.keysA, b.valsA];
      const sp = this.depthPass[p];
      k.radixCount.dispatch(pass, [params, sp, kin, this.counters, b.blockCounts], depthBlocks);
      k.radixScan.dispatch(pass, [params, sp, this.counters, b.blockCounts], 1);
      k.radixScatterKV.dispatch(pass, [params, sp, kin, vin, this.counters, b.blockCounts, kout, vout], depthBlocks);
    }
    // instance offsets in depth order, instance creation
    k.offsetsReduce.dispatch(pass, [params, b.valsA, b.touched, b.blockSums], workgroups(n, SCAN_BLOCK));
    k.offsetsScan.dispatch(pass, [params, b.blockSums, this.counters], 1);
    k.createInstances.dispatch(pass, [params, b.valsA, b.touched, b.blockSums, b.proj, b.instKeysA], workgroups(n, SCAN_BLOCK));
    // stable sort by tile (the bits above rankBits); depth order survives inside each tile
    const tilePasses = Math.ceil(tileBits / 4);
    const instanceBlocks = workgroups(this.mCap, SORT_BLOCK);
    let keys = b.instKeysA;
    for (let p = 0; p < tilePasses; p++) {
      const out = keys === b.instKeysA ? b.instKeysB : b.instKeysA;
      const sp = this.tilePass[p];
      k.radixCount.dispatch(pass, [params, sp, keys, this.counters, b.blockCounts], instanceBlocks);
      k.radixScan.dispatch(pass, [params, sp, this.counters, b.blockCounts], 1);
      k.radixScatterK.dispatch(pass, [params, sp, keys, this.counters, b.blockCounts, out], instanceBlocks);
      keys = out;
    }
    k.ranges.dispatch(pass, [params, this.counters, keys, b.valsA, b.ranges, b.instPrim], workgroups(this.mCap, 256));
    k.blend.dispatch(pass, [params, b.ranges, b.instPrim, b.proj, b.colors, output, b.finalT, b.nProcessed], gridW, gridH);
    pass.end();
    this.last = { params, n, width, height, gridW, gridH, rankBits, instanceKeys: keys, splats };
  }

  /**
   * Re-blend the geometry of the last `record` with other degree-0 SH colours, for example
   * `depth_sh0`: the `_C.inference` image (CHW, no clamp) for the same splats with `sh0` swapped.
   * Must follow a `record` of the same splats (in this or an earlier submitted encoder).
   */
  recordColors(encoder: GPUCommandEncoder, sh0: GPUBuffer, output: GPUBuffer): void {
    const last = this.requireLast();
    if (output.size < 3 * last.width * last.height * 4) throw new Error("FasterGsRasterizer: output buffer too small");
    const b = this.buf;
    const pass = encoder.beginComputePass({ label: "fgs-recolor" });
    this.k.colors.dispatch(pass, [last.params, sh0, b.colorsAlt], workgroups(last.n, 256));
    this.k.blend.dispatch(pass, [last.params, b.ranges, b.instPrim, b.proj, b.colorsAlt, output, b.finalT, b.nProcessed], last.gridW, last.gridH);
    pass.end();
  }

  /**
   * Diagnostic: re-blend the last `record` into `image` and write `mask` (u32 [H * W]), 1 where
   * two contributing primitives of bit-identical depth from different 32-index groups meet in a
   * pixel. Those pixels depend on the CUDA scheduling order, so they bound what parity can mean.
   */
  recordTieMask(encoder: GPUCommandEncoder, image: GPUBuffer, mask: GPUBuffer): void {
    const last = this.requireLast();
    const b = this.buf;
    const pass = encoder.beginComputePass({ label: "fgs-tie-mask" });
    this.k.blendTieMask.dispatch(pass, [last.params, b.ranges, b.instPrim, b.proj, b.colors, last.instanceKeys, b.keysA, image, mask], last.gridW, last.gridH);
    pass.end();
  }

  /**
   * Record the backward pass of the last colour `record`: `gradImage` is dL/d(image), f32 CHW
   * [3, H, W]; writes (or with `accumulate`, adds) dL/d(means, log scales, raw quaternions,
   * opacity logits, sh0) into `out`. Uses that record's intermediates and splat buffers, which
   * must be unchanged; does not change what `recordColors` re-blends. One compute pass of six
   * dispatches after a clear of the accumulators.
   */
  recordBackward(encoder: GPUCommandEncoder, gradImage: GPUBuffer, out: GradientBuffers, options: BackwardOptions = { accumulate: false }): void {
    const last = this.requireLast();
    const { n, width, height, gridW, gridH } = last;
    const tiles = gridW * gridH;
    if (gradImage.size < 3 * width * height * 4) throw new Error("FasterGsRasterizer: gradient image too small");
    if (out.means.size < 12 * n || out.scales.size < 12 * n || out.quats.size < 16 * n || out.opacity.size < 4 * n || out.sh0.size < 12 * n) {
      throw new Error("FasterGsRasterizer: gradient buffers too small");
    }
    const bw = this.ensureBackward(n, tiles, width * height, this.bwCaps.buckets || this.defaultBuckets(tiles));
    const bp = bw.params[options.accumulate ? 1 : 0];
    const b = this.buf;
    const k = this.k;
    const P = last.params;
    encoder.clearBuffer(bw.acc, 0, n * ACC_STRIDE * 4);
    const pass = encoder.beginComputePass({ label: "fgs-backward" });
    k.bucketScan.dispatch(pass, [P, bp, b.ranges, bw.tileBuckets, bw.bucketInfo, this.counters], 1);
    k.bucketInfo.dispatch(pass, [P, bp, b.ranges, bw.tileBuckets, bw.bucketInfo], workgroups(tiles, 256));
    k.snapshot.dispatch(pass, [P, bp, b.ranges, b.instPrim, b.proj, b.colors, bw.tileBuckets, bw.snapshots, bw.pixState], gridW, gridH);
    const buckets = this.bwCaps.buckets;
    k.blendBackward.dispatch(pass, [P, bp, b.instPrim, b.proj, b.colors, bw.bucketInfo, bw.snapshots, bw.pixState, gradImage, bw.acc],
      Math.min(buckets, BUCKET_DISPATCH_X), Math.ceil(buckets / BUCKET_DISPATCH_X));
    const s = last.splats;
    k.preprocessBackward.dispatch(pass, [P, bp, s.means, s.logScales, s.quats, b.touched, bw.acc, out.means, out.scales, out.quats], workgroups(n, 128));
    k.finalizeBackward.dispatch(pass, [P, bp, b.touched, bw.acc, out.opacity, out.sh0], workgroups(n, 256));
    pass.end();
  }

  /** Bucket capacity before any measurement: `backwardInstancesPerPoint` instances per primitive, plus a partial bucket per tile. */
  private defaultBuckets(tiles: number): number {
    return Math.ceil((this.backwardInstancesPerPoint * this.nCap) / 32) + tiles;
  }

  /**
   * Allocate or grow the backward buffers. Growing replaces buffers, so like the forward buffers
   * it must not happen between two recorded uses in one command buffer. Does not invalidate the
   * last forward record.
   */
  private ensureBackward(n: number, tiles: number, pixels: number, buckets: number) {
    const maxBuckets = Math.floor(Math.min(this.device.limits.maxStorageBufferBindingSize, this.device.limits.maxBufferSize) / (BLEND_THREADS * 16)) - 1;
    const caps = this.bwCaps;
    const want = {
      n: Math.max(caps.n, this.nCap, n), tiles: Math.max(caps.tiles, this.tileCap, tiles),
      pixels: Math.max(caps.pixels, this.pixelCap, pixels), buckets: Math.min(Math.max(caps.buckets, buckets), maxBuckets),
    };
    if (this.bw && want.n === caps.n && want.tiles === caps.tiles && want.pixels === caps.pixels && want.buckets === caps.buckets) return this.bw;
    this.destroyBackward();
    const d = this.device;
    const params = [0, 1].map((accumulate) => {
      const u = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true, label: "fgs-backward-params" });
      new Uint32Array(u.getMappedRange()).set([want.buckets, accumulate, 0, 0]);
      u.unmap();
      return u;
    });
    this.bw = {
      acc: createStorageBuffer(d, want.n * ACC_STRIDE * 4, "fgs-grad-acc"),
      tileBuckets: createStorageBuffer(d, want.tiles * 4, "fgs-tile-buckets"),
      bucketInfo: createStorageBuffer(d, (want.buckets + 1) * 16, "fgs-bucket-info"),
      snapshots: createStorageBuffer(d, want.buckets * BLEND_THREADS * 16, "fgs-bucket-snapshots"),
      pixState: createStorageBuffer(d, want.pixels * 8, "fgs-pixel-state"),
      params,
    };
    this.bwCaps = want;
    for (const name of BACKWARD_KERNELS) this.k[name].clearCache();
    return this.bw;
  }

  private destroyBackward(): void {
    if (!this.bw) return;
    const { params, ...bufs } = this.bw;
    for (const b of [...Object.values(bufs), ...params]) b.destroy();
    this.bw = null;
  }

  /** Read the counters of the last submitted render. */
  async readStats(): Promise<RasterStats> {
    const c = new Uint32Array(await readBuffer(this.device, this.counters, 16));
    return { instances: c[COUNTER.instances], capacity: this.mCap, overflow: c[COUNTER.overflow] !== 0 || c[COUNTER.bucketOverflow] !== 0 };
  }

  /** Record, submit, and check for overflow; on overflow grow the instance buffers and render again. */
  async render(splats: SplatBuffers, n: number, camera: FasterGsCamera, output: GPUBuffer, background: ArrayLike<number> = [0, 0, 0]): Promise<RasterStats> {
    for (let attempt = 0; ; attempt++) {
      const encoder = this.device.createCommandEncoder();
      this.record(encoder, splats, n, camera, output, background);
      this.device.queue.submit([encoder.finish()]);
      const stats = await this.readStats();
      if (!stats.overflow || attempt > 0) return stats;
      this.ensureInstanceCapacity(Math.ceil(stats.instances * 1.25));
    }
  }

  /** Buffers of the last `record`; see `ForwardIntermediates`. */
  get intermediates(): ForwardIntermediates {
    const last = this.requireLast();
    const b = this.buf;
    return {
      proj: b.proj, colors: b.colors, touched: b.touched, depthKeysSorted: b.keysA, sortedIndices: b.valsA,
      instanceKeys: last.instanceKeys, instancePrimitives: b.instPrim, tileRanges: b.ranges,
      finalTransmittance: b.finalT, nProcessed: b.nProcessed, counters: this.counters,
      rankBits: last.rankBits, gridW: last.gridW, gridH: last.gridH,
    };
  }

  private requireLast() {
    if (!this.last) throw new Error("FasterGsRasterizer: no render recorded since the buffers were (re)allocated");
    return this.last;
  }

  dispose(): void {
    for (const b of Object.values(this.buf)) b.destroy();
    this.destroyBackward();
    for (const b of this.paramCache.values()) b.destroy();
    for (const b of [...this.depthPass, ...this.tilePass, this.counters]) b.destroy();
    this.buf = {};
    this.paramCache.clear();
    this.last = null;
  }
}

/** Workgroup-count limit per dimension guaranteed by WebGPU. */
const MAX_WORKGROUPS = 65535;

/** SplatBuffers with their count, as the planar engine passes them. */
export interface CountedSplatBuffers extends SplatBuffers {
  n: number;
}

/**
 * The planar engine's rasterizer shape: `record(encoder, splats, cameraSettings, out, colours?)`
 * on a black background. A call with `colours` (an [N, 1, 3] SH0 buffer, for example depth_sh0)
 * that directly follows a colour render of the same geometry buffers, count, and camera re-blends
 * that geometry with the new colours (`recordColors`) instead of running the whole pipeline again;
 * otherwise it runs the whole pipeline with `colours` in place of the splats' SH0. The geometry
 * buffers must not be rewritten between the two calls.
 */
export class FrontalFasterGsRasterizer {
  readonly raster: FasterGsRasterizer;
  private last: { splats: CountedSplatBuffers; cameraKey: string } | null = null;

  constructor(readonly device: GPUDevice, opts: FasterGsRasterizerOptions = {}) {
    this.raster = new FasterGsRasterizer(device, opts);
  }

  record(encoder: GPUCommandEncoder, splats: CountedSplatBuffers, camera: CameraSettingsJson, out: GPUBuffer, colours?: GPUBuffer): void {
    const cameraKey = JSON.stringify(camera);
    const l = this.last;
    if (colours && l && l.cameraKey === cameraKey && l.splats.n === splats.n && l.splats.means === splats.means
      && l.splats.logScales === splats.logScales && l.splats.quats === splats.quats && l.splats.opacityLogits === splats.opacityLogits) {
      this.raster.recordColors(encoder, colours, out);
      return;
    }
    this.raster.record(encoder, colours ? { ...splats, sh0: colours } : splats, splats.n, cameraFromSettings(camera), out, [0, 0, 0]);
    this.last = colours ? null : { splats: { ...splats }, cameraKey };
  }

  /**
   * Backward pass of this instance's last colour render; see `FasterGsRasterizer.recordBackward`.
   * Leaves the re-blend state alone, so a depth `record(..., colours)` can follow.
   */
  recordBackward(encoder: GPUCommandEncoder, gradImage: GPUBuffer, out: GradientBuffers, options: BackwardOptions): void {
    this.raster.recordBackward(encoder, gradImage, out, options);
  }

  /** Overflow check of the last submitted render; see `FasterGsRasterizer.readStats`. */
  readStats(): Promise<RasterStats> { return this.raster.readStats(); }

  /** Grow the tile-instance buffers; see `FasterGsRasterizer.ensureInstanceCapacity`. */
  ensureInstanceCapacity(instances: number): void { this.raster.ensureInstanceCapacity(instances); this.last = null; }

  dispose(): void { this.raster.dispose(); this.last = null; }
}
