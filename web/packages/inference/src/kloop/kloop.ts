/**
 * K-state loop buffers and kernels. The features buffer is the persistent
 * 17-channel NCHW head input; the splatter writes rendered and diff into it
 * directly, so no pack kernel runs per state.
 */
import { Kernel, createStorageBuffer, createUniformBuffer, workgroups, type KernelSpec } from "@gaussifier/wgsl";
import { TileScan } from "./scan.js";
import {
  FEATURES, TILE, LARGE_TILE_THRESHOLD,
  HARD_COUNT_WGSL, HARD_TO_F32_WGSL, INIT_ATTRS_WGSL, PACK_STATIC_WGSL, POST_HEAD_WGSL, PRE_RENDER_WGSL, PROJECT_WGSL,
  RASTER_WGSL, SCATTER_WGSL, packParams,
} from "./shaders.js";
import type { StateSnapshot } from "../types.js";

/** A state copied to a staging buffer inside the loop, read back later by `KLoop.readDeferred`. */
export interface DeferredSnapshot {
  k: number;
  staging: GPUBuffer;
  offs: number[];
  bytes: number[];
  includeRendered: boolean;
  includeLatent: boolean;
}

const RO = "read-only-storage" as const;
const RW = "storage" as const;
const U = "uniform" as const;

const SPECS: Record<string, KernelSpec> = {
  packStatic: { code: PACK_STATIC_WGSL, entryPoint: "pack_static", bindings: [U, RO, RO, RO, RO, RW], label: "pack_static" },
  initAttrs: { code: INIT_ATTRS_WGSL, entryPoint: "init_attrs", bindings: [U, RO, RO, RW, RW], label: "init_attrs" },
  preRender: { code: PRE_RENDER_WGSL, entryPoint: "pre_render", bindings: [U, RO, RO, RO, RW, RW, RW], label: "pre_render" },
  project: { code: PROJECT_WGSL, entryPoint: "project", bindings: [U, RO, RO, RO, RW, RW, RW, RW], label: "project" },
  scatter: { code: SCATTER_WGSL, entryPoint: "scatter", bindings: [U, RO, RO, RW, RW, RW], label: "scatter" },
  raster: { code: RASTER_WGSL, entryPoint: "raster", bindings: [U, RO, RO, RO, RO, RO, RO, RW], label: "raster" },
  hardCount: { code: HARD_COUNT_WGSL, entryPoint: "hard_count", bindings: [U, RO, RW], label: "hard_count" },
  hardToF32: { code: HARD_TO_F32_WGSL, entryPoint: "hard_to_f32", bindings: [U, RO, RW], label: "hard_to_f32" },
  postHead: { code: POST_HEAD_WGSL, entryPoint: "post_head", bindings: [U, RO, RW, RW, RW], label: "post_head" },
};

/** Compiled pipelines, independent of sizes. Create once per device. */
export class KLoopKernels {
  readonly kernels: Record<string, Kernel> = {};
  constructor(readonly device: GPUDevice) {
    for (const [name, spec] of Object.entries(SPECS)) this.kernels[name] = Kernel.create(device, spec);
  }
}

export interface KLoopConfig {
  width: number;
  height: number;
  n: number;
  /** Defaults to 2.56 / max(width, height). */
  xyStep?: number;
  /** Added to the log-covariance diagonal when packing; see PACK_STATIC_WGSL. Defaults to 0. */
  covShift?: number;
}


export function listCapacity(n: number): number {
  return Math.max(8 * n, 4096);
}

/** Per-run buffers and recording helpers. */
export class KLoop {
  readonly width: number;
  readonly height: number;
  readonly n: number;
  readonly hw: number;
  readonly tilesX: number;
  readonly tilesY: number;
  readonly capacity: number;
  readonly xyStep: number;
  readonly params: GPUBuffer;
  /** f32 [FEATURES.count * H * W], NCHW head input; see FEATURES for the channel layout. */
  readonly features: GPUBuffer;
  readonly xy: GPUBuffer;
  readonly cov: GPUBuffer;
  readonly rgb: GPUBuffer;
  readonly weight: GPUBuffer;
  readonly scale: GPUBuffer;
  readonly rot: GPUBuffer;
  readonly color: GPUBuffer;
  readonly proj: GPUBuffer;
  readonly counts: GPUBuffer;
  readonly offsets: GPUBuffer;
  readonly cursors: GPUBuffer;
  readonly list: GPUBuffer;
  readonly large: GPUBuffer;
  readonly flags: GPUBuffer;
  readonly hardScratch: GPUBuffer;
  readonly delta: GPUBuffer;
  private scan: TileScan;
  private k: Record<string, Kernel>;

  constructor(readonly device: GPUDevice, kernels: KLoopKernels, cfg: KLoopConfig) {
    this.k = kernels.kernels;
    this.width = cfg.width; this.height = cfg.height; this.n = cfg.n;
    this.hw = cfg.width * cfg.height;
    this.tilesX = Math.ceil(cfg.width / TILE); this.tilesY = Math.ceil(cfg.height / TILE);
    this.capacity = listCapacity(cfg.n);
    this.xyStep = cfg.xyStep ?? 2.56 / Math.max(cfg.width, cfg.height);
    const tiles = this.tilesX * this.tilesY;
    this.params = createUniformBuffer(device, packParams({ W: cfg.width, H: cfg.height, N: cfg.n, tilesX: this.tilesX, tilesY: this.tilesY, capacity: this.capacity, xyStep: this.xyStep, covShift: cfg.covShift }), "kloop-params");
    const n = cfg.n;
    this.features = createStorageBuffer(device, FEATURES.count * this.hw * 4, "features");
    this.xy = createStorageBuffer(device, 2 * n * 4, "xy");
    this.cov = createStorageBuffer(device, 3 * n * 4, "cov");
    this.rgb = createStorageBuffer(device, 3 * n * 4, "rgb");
    this.weight = createStorageBuffer(device, n * 4, "weight");
    this.scale = createStorageBuffer(device, 2 * n * 4, "scale");
    this.rot = createStorageBuffer(device, n * 4, "rot");
    this.color = createStorageBuffer(device, 3 * n * 4, "color");
    this.proj = createStorageBuffer(device, 8 * n * 4, "proj");
    this.counts = createStorageBuffer(device, tiles * 4, "tile-counts");
    this.offsets = createStorageBuffer(device, (tiles + 1) * 4, "tile-offsets");
    this.cursors = createStorageBuffer(device, tiles * 4, "tile-cursors");
    this.list = createStorageBuffer(device, this.capacity * 4, "tile-list");
    this.large = createStorageBuffer(device, n * 4, "large-list");
    this.flags = createStorageBuffer(device, 16, "flags");
    this.hardScratch = createStorageBuffer(device, this.hw * 4, "hard-scratch");
    this.delta = createStorageBuffer(device, 8 * this.hw * 4, "delta");
    this.scan = new TileScan(device, tiles);
  }

  /** Upload per-point state directly (the engine test page drives the loop from goldens this way). */
  setPoints(xy: Float32Array, cov: Float32Array | null, rgb: Float32Array | null, weight: Float32Array): void {
    const q = this.device.queue;
    q.writeBuffer(this.xy, 0, xy.buffer, xy.byteOffset, 2 * this.n * 4);
    if (cov) q.writeBuffer(this.cov, 0, cov.buffer, cov.byteOffset, 3 * this.n * 4);
    if (rgb) q.writeBuffer(this.rgb, 0, rgb.buffer, rgb.byteOffset, 3 * this.n * 4);
    q.writeBuffer(this.weight, 0, weight.buffer, weight.byteOffset, this.n * 4);
  }

  /** Channels [0:10] from the forward-map outputs. */
  recordPackStatic(pass: GPUComputePassEncoder, image: GPUBuffer, density: GPUBuffer, logcov: GPUBuffer, rgbMap: GPUBuffer): void {
    this.k.packStatic.dispatch(pass, [this.params, image, density, logcov, rgbMap, this.features], workgroups(this.hw, 256));
  }

  /** cov and rgb from the dense maps at the current xy. */
  recordInitAttrs(pass: GPUComputePassEncoder): void {
    this.k.initAttrs.dispatch(pass, [this.params, this.xy, this.features, this.cov, this.rgb], workgroups(this.n, 256));
  }

  recordPreRender(pass: GPUComputePassEncoder): void {
    this.k.preRender.dispatch(pass, [this.params, this.cov, this.rgb, this.weight, this.scale, this.rot, this.color], workgroups(this.n, 256));
  }

  /** Clear the tile scratch buffers; must precede recordSplat, and clears are encoder-level so they sit outside the pass. */
  recordSplatClears(encoder: GPUCommandEncoder): void {
    encoder.clearBuffer(this.counts);
    encoder.clearBuffer(this.cursors);
    encoder.clearBuffer(this.flags);
  }

  recordSplat(pass: GPUComputePassEncoder): void {
    const nWg = workgroups(this.n, 256);
    this.k.project.dispatch(pass, [this.params, this.xy, this.scale, this.rot, this.proj, this.counts, this.large, this.flags], nWg);
    this.scan.record(pass, this.counts, this.offsets);
    this.k.scatter.dispatch(pass, [this.params, this.proj, this.offsets, this.cursors, this.list, this.flags], nWg);
    this.k.raster.dispatch(pass, [this.params, this.proj, this.color, this.list, this.offsets, this.large, this.flags, this.features], this.tilesX, this.tilesY);
  }

  /** Pre-render plus splat in one encoder. */
  recordRender(encoder: GPUCommandEncoder): void {
    this.recordSplatClears(encoder);
    const pass = encoder.beginComputePass({ label: "render" });
    this.recordPreRender(pass);
    this.recordSplat(pass);
    pass.end();
  }

  recordHardCount(encoder: GPUCommandEncoder): void {
    encoder.clearBuffer(this.hardScratch);
    const pass = encoder.beginComputePass({ label: "hard-count" });
    this.k.hardCount.dispatch(pass, [this.params, this.xy, this.hardScratch], workgroups(this.n, 256));
    this.k.hardToF32.dispatch(pass, [this.params, this.hardScratch, this.features], workgroups(this.hw, 256));
    pass.end();
  }

  /** Copy the head output into the owned delta buffer, then apply it. */
  recordPostHead(encoder: GPUCommandEncoder, headOutput: GPUBuffer): void {
    encoder.copyBufferToBuffer(headOutput, 0, this.delta, 0, 8 * this.hw * 4);
    const pass = encoder.beginComputePass({ label: "post-head" });
    this.k.postHead.dispatch(pass, [this.params, this.delta, this.xy, this.cov, this.rgb], workgroups(this.n, 256));
    pass.end();
  }

  /** Segments in order: xy, cov, color, then the optional latent rgb and rendered planes. */
  private snapshotSegments(includeRendered: boolean, includeLatent: boolean): { src: GPUBuffer; bytes: number; offset: number }[] {
    const n = this.n, hw = this.hw;
    const segs = [
      { src: this.xy, bytes: 2 * n * 4, offset: 0 },
      { src: this.cov, bytes: 3 * n * 4, offset: 0 },
      { src: this.color, bytes: 3 * n * 4, offset: 0 },
    ];
    if (includeLatent) segs.push({ src: this.rgb, bytes: 3 * n * 4, offset: 0 });
    if (includeRendered) segs.push({ src: this.features, bytes: 3 * hw * 4, offset: FEATURES.rendered * hw * 4 });
    return segs;
  }

  /**
   * Record copies of the current state into a fresh staging buffer, without waiting. The state
   * buffers are overwritten by the next step, so this is how a run keeps every state at no
   * synchronization cost; `readDeferred` maps the staging buffers once at the end.
   */
  recordSnapshot(enc: GPUCommandEncoder, k: number, includeRendered: boolean, includeLatent = true): DeferredSnapshot {
    const segs = this.snapshotSegments(includeRendered, includeLatent);
    let total = 0;
    const offs = segs.map((s) => { const o = total; total += Math.ceil(s.bytes / 4) * 4; return o; });
    const staging = this.device.createBuffer({ size: total, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, label: `snapshot ${k}` });
    segs.forEach((s, i) => enc.copyBufferToBuffer(s.src, s.offset, staging, offs[i], s.bytes));
    return { k, staging, offs, bytes: segs.map((s) => s.bytes), includeRendered, includeLatent };
  }

  /** Map a deferred snapshot's staging buffer and unpack it; the buffer is destroyed afterwards. */
  async readDeferred(d: DeferredSnapshot): Promise<StateSnapshot> {
    await d.staging.mapAsync(GPUMapMode.READ);
    const all = d.staging.getMappedRange();
    const take = (i: number) => new Float32Array(all.slice(d.offs[i], d.offs[i] + d.bytes[i]));
    const snap: StateSnapshot = { k: d.k, xy: take(0), cov: take(1), color: take(2) };
    let next = 3;
    if (d.includeLatent) snap.rgbLatent = take(next++);
    if (d.includeRendered) snap.rendered = take(next);
    d.staging.unmap();
    d.staging.destroy();
    return snap;
  }

  /** Read back the per-point state and optionally the render, in one staging buffer and one map: a GPU round trip. */
  async snapshot(k: number, includeRendered: boolean, includeLatent = true): Promise<StateSnapshot> {
    const enc = this.device.createCommandEncoder();
    const d = this.recordSnapshot(enc, k, includeRendered, includeLatent);
    this.device.queue.submit([enc.finish()]);
    return this.readDeferred(d);
  }

  dispose(): void {
    for (const b of [this.params, this.features, this.xy, this.cov, this.rgb, this.weight, this.scale, this.rot, this.color, this.proj, this.counts, this.offsets, this.cursors, this.list, this.large, this.flags, this.hardScratch, this.delta]) b.destroy();
    this.scan.dispose();
  }
}
