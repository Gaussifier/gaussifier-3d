import { createStorageBuffer, readBuffer, uploadStorage } from "@gaussifier/wgsl";
import { loadBundle, type LoadedBundle } from "./bundle.js";
import { prepareInput } from "./image.js";
import { KLoop, KLoopKernels, type DeferredSnapshot } from "./kloop/kloop.js";
import { LARGE_TILE_THRESHOLD, TILE } from "./kloop/shaders.js";
import { configureOrt, createSession, describeAdapter, disposeOutputs, ortDevice, runGpu, type OrtSession } from "./ort.js";
import { OrtHead } from "./head/ort_head.js";
import { WgslHead, parseHeadWeights } from "./head/wgsl_head.js";
import type { HeadBackend } from "./head/types.js";
import type { Placement, PlacementParams } from "./placement/types.js";
import { createPlacement } from "./placement/placement.js";
import { createOversampler } from "@gaussifier/wasm-oversampler";
import {
  AbortedError, BundleMismatchError, CountContractError, WebGpuUnavailableError,
  type BundleManifest, type Engine, type EngineOptions, type OversamplerLike, type PlanarImage, type RunOptions, type RunResult, type StateSnapshot,
} from "./types.js";

interface SizeBuffers {
  width: number;
  height: number;
  image: GPUBuffer;
  density: GPUBuffer;
  logcov: GPUBuffer;
  rgb: GPUBuffer;
  rate: GPUBuffer;
}

/** Per-run bookkeeping: stage timings, abort checks, and the optional GPU sync between stages. */
class RunClock {
  readonly marks: Record<string, number> = {};
  private readonly start = performance.now();
  private last = this.start;
  constructor(private readonly device: GPUDevice, private readonly options: RunOptions) {}
  /** Record the time since the previous mark. Submission time unless profiling, which waits for the GPU first. */
  async mark(name: string): Promise<void> {
    if (this.options.profile) await this.device.queue.onSubmittedWorkDone();
    const now = performance.now();
    this.marks[name] = +(now - this.last).toFixed(2);
    this.last = now;
  }
  check(): void {
    if (this.options.signal?.aborted) throw new AbortedError("run aborted");
  }
  finish(): Record<string, number> {
    this.marks.total = +(performance.now() - this.start).toFixed(2);
    return this.marks;
  }
}

/** What the forward map produced, on the GPU (in the size buffers) and the two maps the CPU needs. */
interface ForwardMaps { density: Float32Array; rateSum: number }

interface KeptStates { init: StateSnapshot | null; final: StateSnapshot; snapshots?: StateSnapshot[] }

class EngineImpl implements Engine {
  private sizeBuffers: SizeBuffers | null = null;
  private kernels: KLoopKernels;
  readonly adapter: string;
  constructor(
    readonly device: GPUDevice,
    readonly bundle: LoadedBundle,
    private sessions: { fmDynamic: OrtSession; fmStatic: OrtSession | null },
    private heads: { ort: OrtHead; wgsl: WgslHead | null },
    private defaultHead: "ort" | "wgsl",
    private oversampler: OversamplerLike,
    private placement: Placement | null,
  ) {
    this.kernels = new KLoopKernels(device);
    this.adapter = describeAdapter(device);
  }

  get manifest(): BundleManifest { return this.bundle.manifest; }

  get headKinds(): Array<"ort" | "wgsl"> {
    const kinds: Array<"ort" | "wgsl"> = [this.defaultHead];
    if (this.defaultHead === "wgsl") kinds.push("ort"); else if (this.heads.wgsl) kinds.push("wgsl");
    return kinds;
  }

  private selectHead(kind: "ort" | "wgsl" | undefined): HeadBackend {
    const k = kind ?? this.defaultHead;
    if (k === "wgsl") { if (!this.heads.wgsl) throw new Error("wgsl head unavailable: the bundle has no correction_head_wgsl weights"); return this.heads.wgsl; }
    return this.heads.ort;
  }

  /** The image and forward-map buffers for one frame size, reallocated when the size changes. */
  private buffersFor(width: number, height: number): SizeBuffers {
    const b = this.sizeBuffers;
    if (b && b.width === width && b.height === height) return b;
    if (b) for (const x of [b.image, b.density, b.logcov, b.rgb, b.rate]) x.destroy();
    const hw = width * height;
    const nb: SizeBuffers = {
      width, height,
      image: createStorageBuffer(this.device, 3 * hw * 4, "image"),
      density: createStorageBuffer(this.device, hw * 4, "density"),
      logcov: createStorageBuffer(this.device, 3 * hw * 4, "logcov"),
      rgb: createStorageBuffer(this.device, 3 * hw * 4, "rgbmap"),
      rate: createStorageBuffer(this.device, hw * 4, "rate"),
    };
    this.sizeBuffers = nb;
    return nb;
  }

  /**
   * The production profile, stage by stage: upload the padded image, run the forward map, choose
   * the count and oversample on the CPU, place the points on the GPU, then render K states with
   * the correction head between them.
   */
  async run(image: ImageBitmap | ImageData | PlanarImage, options: RunOptions = {}): Promise<RunResult> {
    const clock = new RunClock(this.device, options);
    const m = this.bundle.manifest;
    const K = Math.max(1, Math.floor(options.states ?? m.model.correction_iterations));
    const seed = options.seed ?? m.sampler.seed_default;

    const prep = prepareInput(image, m.input.pad_multiple);
    const { width: W, height: H } = prep;
    const buf = this.buffersFor(W, H);
    this.device.queue.writeBuffer(buf.image, 0, prep.planar.buffer, prep.planar.byteOffset, prep.planar.byteLength);
    await clock.mark("input");
    clock.check();

    const maps = await this.forwardMap(buf, clock);
    clock.check();

    const adaptiveCount = Math.max(1, Math.trunc(maps.rateSum));
    const N = Math.max(1, Math.trunc(options.count !== undefined ? options.count : maps.rateSum * (options.countScale ?? 1)));
    const nMax = Math.max(N, Math.ceil(m.sampler.oversample * N));
    const oversampled = this.oversampler.sample(maps.density, W, H, nMax, seed, m.sampler.tile_cap);
    if (oversampled.length !== 2 * nMax) throw new CountContractError(`oversampler returned ${oversampled.length / 2} points, expected ${nMax}`);
    await clock.mark("oversample");
    clock.check();

    // A count above the adaptive one keeps its splatted mass: Sigma *= adaptiveCount / N. Below it
    // the loop recovers the darker start on its own, so the map is left alone.
    const kloop = this.place(buf, oversampled, N, nMax, Math.min(0, Math.log(adaptiveCount / N)));
    await clock.mark("placement_submit");
    clock.check();

    const head = this.selectHead(options.headBackend);
    head.prepare(W, H);
    const weightRead = readBuffer(this.device, kloop.weight, N * 4);
    const states = await this.refine(kloop, head, K, options, clock);
    const weight = new Float32Array(await weightRead);
    kloop.dispose();
    return {
      width: W, height: H, crop: prep.crop, n: N, adaptiveCount, seed, states: K, density: maps.density, weight,
      ...(states.init ? { init: states.init } : {}), final: states.final, ...(states.snapshots ? { snapshots: states.snapshots } : {}),
      timings: clock.finish(),
    };
  }

  /** Run the UNet, keep its four maps on the GPU, and read back density and rate for the CPU stages. */
  private async forwardMap(buf: SizeBuffers, clock: RunClock): Promise<ForwardMaps> {
    const m = this.bundle.manifest;
    const { width: W, height: H } = buf;
    const hw = W * H;
    const session = (W === 512 && H === 512 && this.sessions.fmStatic) || this.sessions.fmDynamic;
    const outputs = await runGpu(session, m.io.forward_map.input, buf.image, [1, 3, H, W]);
    const byName = new Map(outputs.map((o) => [o.name, o]));
    const output = (name: string) => { const o = byName.get(name); if (!o) throw new Error(`forward map output ${name} missing`); return o.buffer; };
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(output("density"), 0, buf.density, 0, hw * 4);
    enc.copyBufferToBuffer(output("log_cov"), 0, buf.logcov, 0, 3 * hw * 4);
    enc.copyBufferToBuffer(output("rgb"), 0, buf.rgb, 0, 3 * hw * 4);
    enc.copyBufferToBuffer(output("rate"), 0, buf.rate, 0, hw * 4);
    this.device.queue.submit([enc.finish()]);
    disposeOutputs(outputs);
    await clock.mark("forward_map");
    const [densityBytes, rateBytes] = await Promise.all([readBuffer(this.device, buf.density, hw * 4), readBuffer(this.device, buf.rate, hw * 4)]);
    const rate = new Float32Array(rateBytes);
    let rateSum = 0;
    for (let i = 0; i < rate.length; i++) rateSum += rate[i];
    await clock.mark("readback_maps");
    return { density: new Float32Array(densityBytes), rateSum };
  }

  /** Placement plus the initial attributes, recorded into one submit; returns the K-loop holding the state. */
  private place(buf: SizeBuffers, oversampled: Float32Array, N: number, nMax: number, covShift: number): KLoop {
    if (!this.placement) throw new Error("placement stage unavailable: no Placement implementation was provided");
    const m = this.bundle.manifest;
    const { width: W, height: H } = buf;
    const params: PlacementParams = {
      width: W, height: H, n: N, nMax,
      mergeRounds: m.sampler.merge_rounds, perRoundFraction: m.sampler.per_round_fraction, knnK: m.sampler.knn_k, lloydIters: m.sampler.lloyd_iters,
    };
    this.placement.prepare(params);
    const kloop = new KLoop(this.device, this.kernels, { width: W, height: H, n: N, xyStep: m.model.correction_xy_step_pixels / Math.max(W, H), covShift });
    const pointsIn = uploadStorage(this.device, oversampled, "oversampled");
    const enc = this.device.createCommandEncoder();
    const out = this.placement.record(enc, { density: buf.density, points: pointsIn });
    enc.copyBufferToBuffer(out.points, 0, kloop.xy, 0, 2 * N * 4);
    enc.copyBufferToBuffer(out.weights, 0, kloop.weight, 0, N * 4);
    const pass = enc.beginComputePass({ label: "pack-init" });
    kloop.recordPackStatic(pass, buf.image, buf.density, buf.logcov, buf.rgb);
    kloop.recordInitAttrs(pass);
    pass.end();
    this.device.queue.submit([enc.finish()]);
    pointsIn.destroy();
    return kloop;
  }

  /**
   * K rendered states with the correction chain between them. Kept states ride along in the
   * render's command buffer and are read once after the loop; otherwise each wanted state is read
   * back on the spot, one GPU round trip per state.
   */
  private async refine(kloop: KLoop, head: HeadBackend, K: number, options: RunOptions, clock: RunClock): Promise<KeptStates> {
    const { width: W, height: H } = kloop;
    const includeLatent = options.includeLatent !== false;
    const deferAll = !!options.keepStates && !options.onState;
    const deferred: DeferredSnapshot[] = [];
    let init: StateSnapshot | null = null;
    let final: StateSnapshot | null = null;
    for (let k = 0; k < K; k++) {
      const last = k === K - 1;
      const wantInit = k === 0 && options.includeInit !== false;
      const withRender = !!options.includeRendered || last;
      const enc = this.device.createCommandEncoder();
      kloop.recordRender(enc);
      // The final state is always kept: at K=1 it is also the initializer the caller may not want.
      if (deferAll && (k > 0 || wantInit || last)) deferred.push(kloop.recordSnapshot(enc, k, withRender, includeLatent));
      this.device.queue.submit([enc.finish()]);
      if (!deferAll && (wantInit || last || options.onState)) {
        const snap = await kloop.snapshot(k, withRender, includeLatent);
        if (wantInit) init = snap;
        if (last) final = snap;
        options.onState?.(k, snap);
      }
      await clock.mark(`state_${k}`);
      clock.check();
      if (last) break;
      await this.applyHead(kloop, head, W, H, k, options, clock);
      clock.check();
    }
    let snapshots: StateSnapshot[] | undefined;
    if (deferred.length > 0) {
      snapshots = await Promise.all(deferred.map((d) => kloop.readDeferred(d)));
      if (options.includeInit !== false && snapshots[0]?.k === 0) init = snapshots[0];
      final = snapshots[snapshots.length - 1];
      await clock.mark("readback_states");
    }
    if (!final) throw new Error("internal: missing final snapshot");
    return { init, final, snapshots };
  }

  /**
   * One correction step: hard count, head, post-head. The WGSL head records into the same command
   * buffer; the ORT head runs through its own session, so what precedes it is submitted first.
   * When profiling, each piece is submitted and timed on its own.
   */
  private async applyHead(kloop: KLoop, head: HeadBackend, W: number, H: number, k: number, options: RunOptions, clock: RunClock): Promise<void> {
    const cut = async (enc: GPUCommandEncoder, name: string): Promise<GPUCommandEncoder> => {
      if (!options.profile) return enc;
      this.device.queue.submit([enc.finish()]);
      await clock.mark(name);
      return this.device.createCommandEncoder();
    };
    let chain = this.device.createCommandEncoder();
    kloop.recordHardCount(chain);
    chain = await cut(chain, `hardcount_${k}`);
    let delta: GPUBuffer;
    if (head.record) {
      delta = head.record(chain, kloop.features, W, H);
      chain = await cut(chain, `head_run_${k}`);
    } else {
      this.device.queue.submit([chain.finish()]);
      delta = await head.run(kloop.features, W, H);
      if (options.profile) await clock.mark(`head_run_${k}`);
      chain = this.device.createCommandEncoder();
    }
    kloop.recordPostHead(chain, delta);
    this.device.queue.submit([chain.finish()]);
    if (!head.record) head.release();
    await clock.mark(`head_${k}`);
  }

  dispose(): void {
    const b = this.sizeBuffers;
    if (b) for (const x of [b.image, b.density, b.logcov, b.rgb, b.rate]) x.destroy();
    this.placement?.dispose();
    this.oversampler.dispose?.();
    void this.sessions.fmDynamic.release();
    void this.sessions.fmStatic?.release();
    this.heads.ort.dispose();
    this.heads.wgsl?.dispose();
  }
}

/** Create sessions first, take ORT's device, then build every pipeline on it. */
export async function createEngine(options: EngineOptions): Promise<Engine> {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) throw new WebGpuUnavailableError("navigator.gpu is unavailable");
  // A browser can expose navigator.gpu and still have no adapter (WebGPU disabled, blocklisted
  // GPU); probing here keeps ORT's backend-selection error out of the user-facing message.
  if (!(await navigator.gpu.requestAdapter({ powerPreference: options.powerPreference }))) throw new WebGpuUnavailableError("no WebGPU adapter");
  configureOrt({ wasmPaths: options.ortWasmPaths, logLevel: options.ortLogLevel, powerPreference: options.powerPreference });
  const bundle = await loadBundle(options.bundleUrl);
  const r = bundle.manifest.renderer;
  if (r.tile !== TILE || r.large_tile_threshold !== LARGE_TILE_THRESHOLD) throw new BundleMismatchError(`bundle renderer rules (tile ${r.tile}, large ${r.large_tile_threshold}) differ from the kernels (${TILE}, ${LARGE_TILE_THRESHOLD})`);
  const get = (key: string) => bundle.models.get(key);
  const fmDynamic = await createSession(get("forward_map:dynamic")!);
  const headDynamic = await createSession(get("correction_head:dynamic")!);
  const fmStaticBytes = get("forward_map:static512");
  const headStaticBytes = get("correction_head:static512");
  const fmStatic = fmStaticBytes ? await createSession(fmStaticBytes, { graphCapture: true }) : null;
  const headStatic = headStaticBytes ? await createSession(headStaticBytes, { graphCapture: true }) : null;
  const device = await ortDevice();
  const oversampler = options.oversampler ? await options.oversampler() : await createOversampler();
  const placement = options.placement ? await options.placement(device) : createPlacement(device);
  const ortHead = new OrtHead(headDynamic, headStatic, bundle.manifest.io.correction_head.input, bundle.manifest.io.correction_head.output);
  let wgslHead: WgslHead | null = null;
  const wgslBytes = get("correction_head_wgsl:any"), layoutBytes = get("correction_head_wgsl_layout:any");
  if (options.headBackend !== "ort" && wgslBytes && layoutBytes) {
    const { layout, weights } = parseHeadWeights(layoutBytes, wgslBytes);
    wgslHead = new WgslHead(device, layout, weights);
  } else if (options.headBackend === "wgsl") {
    throw new Error("headBackend wgsl requested but the bundle has no correction_head_wgsl files");
  }
  const defaultHead: "ort" | "wgsl" = wgslHead ? "wgsl" : "ort";
  const engine = new EngineImpl(device, bundle, { fmDynamic, fmStatic }, { ort: ortHead, wgsl: wgslHead }, defaultHead, oversampler, placement);
  if (options.warmup) {
    const { width, height } = options.warmup;
    const planar = new Float32Array(3 * width * height).fill(0.5);
    try { await engine.run({ data: planar, width, height }, { states: 2, count: 64 }); } catch { /* placement may be unavailable; sessions are still warm */ }
  }
  return engine;
}
