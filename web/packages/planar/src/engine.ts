/**
 * The planar runtime contract in the browser (the call sequence of `gaussifier_planar_infer`):
 *
 *   rate_sum, density, maps = forward_map(image)                       ONNX
 *   points, weights = EMV(density, count)                              WASM oversampler + WGSL placement
 *   state, log_weights, footprint, constant, extra = setup(...)        ONNX
 *   for k in 0..K-1:
 *     splats, depth_sh0 = pre_render(state, log_weights)               ONNX
 *     render = raster(splats, frontal)                                 WGSL (Faster-GS port)
 *     if k == K-1: break
 *     depth_render = raster(splats with depth_sh0 colours, frontal)    WGSL
 *     hard_count = count(state xy)                                     WGSL
 *     features = head(constant, extra, image, render, depth_render, hard_count)   ONNX
 *     state = update(features, hard_count, state, footprint)           ONNX
 *   splats = last pre_render + the depth-tie break                     CPU
 *
 * A model with gradient input (the quality recipe) also backpropagates two residuals through the
 * rasterizer each iteration, the frontal one and the 30-degree view's against the plane-warped
 * target, and the update graph reads their summed gradients. Every tensor stays on the GPU;
 * ONNX Runtime and the WGSL kernels share its device.
 */
import * as ort from "onnxruntime-web/webgpu";
import { internals as inf } from "@gaussifier/inference";
import type { Placement } from "@gaussifier/inference";
import type { Oversampler } from "@gaussifier/wasm-oversampler";
import { createStorageBuffer, readBuffer, uploadStorage } from "@gaussifier/wgsl";
import { GRAPH_NAMES, camerasFor, halfThickness, pointCount, type GraphName, type PlanarBundle, type PlanarCamera, type PlanarManifest } from "./bundle.js";
import { PointCounter } from "./count.js";
import { psnr, tieFreeMeans } from "./final.js";
import { Residual } from "./residual.js";
import { PlaneWarp } from "./warp.js";
import type { CountedSplatBuffers } from "./raster/index.js";

/** What the engine needs from the rasterizer: a Faster-GS render recorded into an encoder. */
export interface FrontalRasterizer {
  /**
   * Record a render of `splats` seen from `camera` on a black background into `out` (CHW f32,
   * 3 x H x W). `colours` (an (N,1,3) SH0 buffer) replaces the splats' own SH0, unclamped, as
   * Faster-GS `inference` does for the depth-as-colour maps.
   */
  record(encoder: GPUCommandEncoder, splats: CountedSplatBuffers, camera: PlanarCamera, out: GPUBuffer, colours?: GPUBuffer): void;
  /** Instance count and overflow of the last submitted render. */
  readStats?(): Promise<{ instances: number; overflow: boolean }>;
  /** Grow the tile-instance buffers to hold `instances`. */
  ensureInstanceCapacity?(instances: number): void;
  /**
   * Record the gradient of the last colour render for dL/d(image) = `gradImage` into `out`
   * (Faster-GS `backward`, the five raw-parameter gradients), adding into it with `accumulate`.
   * Needed for models with gradient input.
   */
  recordBackward?(encoder: GPUCommandEncoder, gradImage: GPUBuffer, out: GradientBuffers, options: { accumulate: boolean }): void;
}

/** Faster-GS gradients: (N,3) means, (N,3) log-scales, (N,4) quats, (N,1) opacity logits, (N,1,3) SH0. */
export interface GradientBuffers {
  means: GPUBuffer;
  scales: GPUBuffer;
  quats: GPUBuffer;
  opacity: GPUBuffer;
  sh0: GPUBuffer;
}

/** A CHW f32 image in [0, 1]. */
export interface PlanarImage {
  chw: Float32Array;
  width: number;
  height: number;
}

/** Final Gaussians on the CPU, in the runtime's world frame (OpenCV axes, image height 1). */
export interface PlanarSplats {
  n: number;
  means: Float32Array;
  logScales: Float32Array;
  quats: Float32Array;
  opacityLogits: Float32Array;
  sh0: Float32Array;
}

export interface PlanarRunOptions {
  seed?: number;
  /** Multiplies the model's own point count (1 = as trained). */
  countScale?: number;
  /** Read back every render to report the frontal PSNR per iteration (one GPU sync each). */
  psnrPerIteration?: boolean;
  /** Rendered states: the initializer plus states - 1 correction steps (default: as trained). */
  states?: number;
  /** Keep every state's Gaussians and frontal render (one readback per state). */
  keepStates?: boolean;
  onProgress?: (stage: string, iteration?: number) => void;
  signal?: AbortSignal;
}

export interface PlanarRunResult {
  splats: PlanarSplats;
  /** Model count (from the forward map) and the count used. */
  adaptiveCount: number;
  count: number;
  /** Frontal PSNR of the last render, and of every render with `psnrPerIteration`. */
  psnr: number;
  psnrPerIteration: number[];
  /** The last frontal render, CHW f32. */
  render: Float32Array;
  /** Every state with `keepStates` (state 0 is the initializer), else the last one. */
  states: PlanarState[];
  /** The forward map's density (H x W), for display. */
  density: Float32Array;
  width: number;
  height: number;
  timings: Record<string, number>;
}

/** One rendered state of the loop: its Gaussians, frontal render (CHW f32), and frontal PSNR. */
export interface PlanarState {
  k: number;
  splats: PlanarSplats;
  render: Float32Array;
  psnr: number;
}

type Outputs = Map<string, { buffer: GPUBuffer; dims: number[]; tensor: ort.Tensor }>;

async function run(session: ort.InferenceSession, feeds: Record<string, { buffer: GPUBuffer; dims: number[] }>): Promise<Outputs> {
  const ortFeeds: Record<string, ort.Tensor> = {};
  for (const name of session.inputNames) {
    const feed = feeds[name];
    if (!feed) throw new Error(`missing input ${name}`);
    ortFeeds[name] = inf.gpuTensor(feed.buffer, feed.dims);
  }
  const outputs = await session.run(ortFeeds);
  const map: Outputs = new Map();
  for (const [name, tensor] of Object.entries(outputs)) {
    if (tensor.location !== "gpu-buffer") throw new Error(`output ${name} is not on the GPU (${tensor.location})`);
    map.set(name, { buffer: tensor.gpuBuffer as GPUBuffer, dims: [...tensor.dims], tensor });
  }
  return map;
}

function release(outputs: Outputs | null): void {
  if (outputs) for (const o of outputs.values()) o.tensor.dispose();
}

function get(outputs: Outputs, name: string) {
  const o = outputs.get(name);
  if (!o) throw new Error(`graph output ${name} missing`);
  return o;
}

const bytesOf = (dims: number[]) => 4 * dims.reduce((a, b) => a * b, 1);

export class AbortedError extends Error {
  override name = "AbortedError";
}

export class PlanarEngine {
  private constructor(
    readonly device: GPUDevice,
    readonly manifest: PlanarManifest,
    private readonly sessions: Map<GraphName, ort.InferenceSession>,
    private readonly placement: Placement,
    private readonly oversampler: Oversampler,
    private readonly raster: FrontalRasterizer,
    /** The 30-degree view's rasterizer (gradient models only): its own forward intermediates. */
    private readonly raster30: FrontalRasterizer | null,
  ) {}

  /**
   * Sessions first: ORT owns the WebGPU device, and every other pipeline must be built on it, so
   * the rasterizer comes from a factory that receives that device.
   */
  static async create(bundle: PlanarBundle, oversampler: Oversampler, rasterFactory: (device: GPUDevice) => FrontalRasterizer): Promise<PlanarEngine> {
    const sessions = new Map<GraphName, ort.InferenceSession>();
    for (const name of GRAPH_NAMES) {
      const bytes = bundle.models.get(name);
      if (!bytes) throw new Error(`bundle has no ${name} graph`);
      sessions.set(name, await inf.createSession(bytes));
    }
    const device = await inf.ortDevice();
    const raster = rasterFactory(device);
    const raster30 = bundle.manifest.gradient_input ? rasterFactory(device) : null;
    if (raster30 && !(raster.recordBackward && raster30.recordBackward)) throw new Error("a model with gradient input needs a rasterizer with a backward pass");
    return new PlanarEngine(device, bundle.manifest, sessions, inf.createPlacement(device), oversampler, raster, raster30);
  }

  private warp: PlaneWarp | null = null;

  private session(name: GraphName): ort.InferenceSession {
    const s = this.sessions.get(name);
    if (!s) throw new Error(`no ${name} session`);
    return s;
  }

  /** Run the model on a CHW f32 image in [0, 1] whose sides divide by `geometry.size_multiple`. */
  async run(input: PlanarImage, options: PlanarRunOptions = {}): Promise<PlanarRunResult> {
    const m = this.manifest;
    const { width: W, height: H, chw: image } = input;
    const HW = H * W;
    const mult = m.geometry.size_multiple;
    if (H % mult || W % mult || H < mult || W < mult) throw new Error(`image sides must divide by ${mult}, got ${W} x ${H}`);
    if (image.length !== 3 * HW) throw new Error(`image must be 3 x ${H} x ${W}`);
    const cameras = camerasFor(m, H, W);
    const device = this.device;
    const t0 = performance.now();
    let tLast = t0;
    const timings: Record<string, number> = {};
    const mark = (name: string) => { const now = performance.now(); timings[name] = +(now - tLast).toFixed(1); tLast = now; };
    const check = () => { if (options.signal?.aborted) throw new AbortedError("run aborted"); };
    const owned: GPUBuffer[] = [];
    const own = (bytes: number, label: string) => { const b = createStorageBuffer(device, bytes, label); owned.push(b); return b; };
    let fm: Outputs | null = null;
    let counter: PointCounter | null = null;
    let residual: Residual | null = null;
    const uniforms: GPUBuffer[] = [];
    try {
      options.onProgress?.("forward map");
      const imageBuf = uploadStorage(device, image, "planar-image");
      owned.push(imageBuf);
      // The graphs read the size and the slab thickness: [height, width, thickness_px].
      const geom = uploadStorage(device, new Float32Array([H, W, m.geometry.thickness_px]), "planar-geom");
      owned.push(geom);
      fm = await run(this.session("forward_map"), { image: { buffer: imageBuf, dims: [1, 3, H, W] } });
      const [rateBytes, densityBytes] = await Promise.all([
        readBuffer(device, get(fm, "rate_sum").buffer, 4),
        readBuffer(device, get(fm, "density").buffer, HW * 4),
      ]);
      const rateSum = new Float32Array(rateBytes)[0];
      const density = new Float32Array(densityBytes);
      mark("forward_map");
      check();

      options.onProgress?.("placement");
      const adaptiveCount = pointCount(m, rateSum, H, W);
      const n = Math.max(1, Math.min(adaptiveCount * 8, Math.round(adaptiveCount * (options.countScale ?? 1))));
      const seed = options.seed ?? m.placement.seed;
      const s = m.sampler;
      const nMax = Math.max(n, Math.ceil(s.oversample * n));
      const oversampled = this.oversampler.sample(density, W, H, nMax, seed, s.tile_cap);
      const pointsIn = uploadStorage(device, oversampled, "planar-oversampled");
      owned.push(pointsIn);
      this.placement.prepare({ width: W, height: H, n, nMax, mergeRounds: s.merge_rounds, perRoundFraction: s.per_round_fraction, knnK: s.knn_k, lloydIters: s.lloyd_iters });
      const densityUnit = own(HW * 4, "planar-density");
      const points = own(n * 2 * 4, "planar-points");
      const weights = own(n * 4, "planar-weights");
      {
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(get(fm, "density").buffer, 0, densityUnit, 0, HW * 4);
        const placed = this.placement.record(enc, { density: densityUnit, points: pointsIn });
        enc.copyBufferToBuffer(placed.points, 0, points, 0, n * 2 * 4);
        enc.copyBufferToBuffer(placed.weights, 0, weights, 0, n * 4);
        device.queue.submit([enc.finish()]);
      }
      mark("placement");
      check();

      options.onProgress?.("setup");
      const feedOf = (o: Outputs, name: string) => ({ buffer: get(o, name).buffer, dims: get(o, name).dims });
      const setup = await run(this.session("setup"), {
        image: { buffer: imageBuf, dims: [3, H, W] },
        density: feedOf(fm, "density"),
        log_cov: feedOf(fm, "log_cov"),
        rgb: feedOf(fm, "rgb"),
        depth: feedOf(fm, "depth"),
        opacity: feedOf(fm, "opacity"),
        normal: feedOf(fm, "normal"),
        points: { buffer: points, dims: [n, 2] },
        weights: { buffer: weights, dims: [n] },
      });
      const stateDims = get(setup, "state").dims;
      const stride = stateDims[1];
      const state = own(bytesOf(stateDims), "planar-state");
      const logWeights = own(n * 4, "planar-log-weights");
      const footprint = own(n * 4, "planar-footprint");
      const constant = own(bytesOf(get(setup, "constant").dims), "planar-constant");
      const extra = own(bytesOf(get(setup, "extra").dims), "planar-extra");
      const constantDims = get(setup, "constant").dims, extraDims = get(setup, "extra").dims;
      {
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(get(setup, "state").buffer, 0, state, 0, bytesOf(stateDims));
        enc.copyBufferToBuffer(get(setup, "log_weights").buffer, 0, logWeights, 0, n * 4);
        enc.copyBufferToBuffer(get(setup, "footprint").buffer, 0, footprint, 0, n * 4);
        enc.copyBufferToBuffer(get(setup, "constant").buffer, 0, constant, 0, bytesOf(constantDims));
        enc.copyBufferToBuffer(get(setup, "extra").buffer, 0, extra, 0, bytesOf(extraDims));
        device.queue.submit([enc.finish()]);
      }
      release(setup);
      release(fm);
      fm = null;
      mark("setup");
      check();

      const K = Math.max(1, Math.floor(options.states ?? m.meta.iterations));
      const render = own(3 * HW * 4, "planar-render");
      const depthRender = own(3 * HW * 4, "planar-depth-render");
      const hardCount = own(HW * 4, "planar-hard-count");
      counter = new PointCounter(device, n, W, H, stride, 6);
      // Gradient models: the 30-degree target (once per image) and the gradient buffers.
      const gradient = m.gradient_input && this.raster30 ? (() => {
        const target30 = own(3 * HW * 4, "planar-target30");
        const enc = device.createCommandEncoder();
        this.warp ??= new PlaneWarp(device);
        uniforms.push(this.warp.record(enc, imageBuf, W, H, cameras.gradientView, target30, 4));
        device.queue.submit([enc.finish()]);
        residual = new Residual(device, 3 * HW);
        return {
          target30,
          render30: own(3 * HW * 4, "planar-render30"),
          resid: own(3 * HW * 4, "planar-residual"),
          grads: {
            means: own(n * 3 * 4, "planar-g-means"),
            scales: own(n * 3 * 4, "planar-g-scales"),
            quats: own(n * 4 * 4, "planar-g-quats"),
            opacity: own(n * 4, "planar-g-opacity"),
            sh0: own(n * 3 * 4, "planar-g-sh0"),
          } satisfies GradientBuffers,
        };
      })() : null;
      const psnrs: number[] = [];
      const states: PlanarState[] = [];
      let finalSplats: PlanarSplats | null = null;
      let lastRender: Float32Array | null = null;
      for (let k = 0; k < K; k++) {
        options.onProgress?.("refine", k);
        const pre = await run(this.session("pre_render"), {
          geom: { buffer: geom, dims: [3] },
          state: { buffer: state, dims: stateDims },
          log_weights: { buffer: logWeights, dims: [n] },
        });
        const splats: CountedSplatBuffers = {
          n,
          means: get(pre, "means").buffer,
          logScales: get(pre, "log_scales").buffer,
          quats: get(pre, "quats").buffer,
          opacityLogits: get(pre, "opacity_logits").buffer,
          sh0: get(pre, "sh0").buffer,
        };
        const last = k === K - 1;
        // A render that outgrows the tile-instance buffers flags it; grow them and render again.
        for (let attempt = 0; ; attempt++) {
          const enc = device.createCommandEncoder();
          this.raster.record(enc, splats, cameras.frontal, render);
          if (!last && gradient && residual && this.raster30) {
            // Same order as the reference: frontal backward, 30-degree forward and backward, then the
            // depth maps (a re-blend of the frontal instance's last geometry).
            residual.record(enc, render, imageBuf, gradient.resid);
            this.raster.recordBackward!(enc, gradient.resid, gradient.grads, { accumulate: false });
            this.raster30.record(enc, splats, cameras.gradientView, gradient.render30);
            residual.record(enc, gradient.render30, gradient.target30, gradient.resid);
            this.raster30.recordBackward!(enc, gradient.resid, gradient.grads, { accumulate: true });
          }
          if (!last) {
            this.raster.record(enc, splats, cameras.frontal, depthRender, get(pre, "depth_sh0").buffer);
            counter.record(enc, state, hardCount);
          }
          device.queue.submit([enc.finish()]);
          let overflow = false;
          for (const r of gradient && !last ? [this.raster, this.raster30!] : [this.raster]) {
            const stats = r.readStats ? await r.readStats() : null;
            if (!stats?.overflow) continue;
            overflow = true;
            if (attempt > 0 || !r.ensureInstanceCapacity) throw new Error(`rasterizer overflow at ${stats.instances} tile instances`);
            r.ensureInstanceCapacity(Math.ceil(stats.instances * 1.25));
          }
          if (!overflow) break;
        }
        const keep = last || !!options.keepStates;
        if (options.psnrPerIteration || keep) {
          const r = new Float32Array(await readBuffer(device, render, 3 * HW * 4));
          psnrs.push(psnr(r, image));
          if (last) lastRender = r;
          if (keep) {
            const read = (name: string, count: number) => readBuffer(device, get(pre, name).buffer, count * 4).then((b) => new Float32Array(b));
            const [means, logScales, quats, opacityLogits, sh0] = await Promise.all([
              read("means", 3 * n), read("log_scales", 3 * n), read("quats", 4 * n), read("opacity_logits", n), read("sh0", 3 * n),
            ]);
            tieFreeMeans(means, halfThickness(m, H), m.final.tie_break_rel_step);
            const splats = { n, means, logScales, quats, opacityLogits, sh0 };
            states.push({ k, splats, render: r, psnr: psnrs[psnrs.length - 1] });
            if (last) finalSplats = splats;
          }
        }
        if (last) {
          release(pre);
          mark(`iteration_${k}`);
          break;
        }
        const head = await run(this.session("head"), {
          constant: { buffer: constant, dims: constantDims },
          extra: { buffer: extra, dims: extraDims },
          image: { buffer: imageBuf, dims: [3, H, W] },
          render: { buffer: render, dims: [3, H, W] },
          depth_render: { buffer: depthRender, dims: [3, H, W] },
          hard_count: { buffer: hardCount, dims: [1, 1, H, W] },
        });
        const update = await run(this.session("update"), {
          geom: { buffer: geom, dims: [3] },
          features: feedOf(head, "features"),
          hard_count: { buffer: hardCount, dims: [1, 1, H, W] },
          state: { buffer: state, dims: stateDims },
          footprint: { buffer: footprint, dims: [n] },
          ...(gradient ? {
            g_means: { buffer: gradient.grads.means, dims: [n, 3] },
            g_scales: { buffer: gradient.grads.scales, dims: [n, 3] },
            g_quats: { buffer: gradient.grads.quats, dims: [n, 4] },
            g_opacity: { buffer: gradient.grads.opacity, dims: [n, 1] },
            g_sh0: { buffer: gradient.grads.sh0, dims: [n, 1, 3] },
          } : {}),
        });
        const enc2 = device.createCommandEncoder();
        enc2.copyBufferToBuffer(get(update, "state_next").buffer, 0, state, 0, bytesOf(stateDims));
        device.queue.submit([enc2.finish()]);
        release(update);
        release(head);
        release(pre);
        mark(`iteration_${k}`);
        check();
      }
      if (!finalSplats || !lastRender) throw new Error("internal: the loop ended without a final state");
      timings.total = +(performance.now() - t0).toFixed(1);
      return {
        splats: finalSplats, adaptiveCount, count: n,
        psnr: psnrs[psnrs.length - 1], psnrPerIteration: psnrs, render: lastRender, states, density, width: W, height: H, timings,
      };
    } finally {
      release(fm);
      counter?.dispose();
      (residual as Residual | null)?.dispose();
      for (const b of [...owned, ...uniforms]) b.destroy();
    }
  }

  dispose(): void {
    this.placement.dispose();
    this.oversampler.dispose();
    for (const s of this.sessions.values()) void s.release();
  }
}
