/**
 * GPU placement stage: equal-mass Voronoi with the capturable native semantics.
 * All work is recorded into the caller's encoder; no readbacks; static sizes.
 */
import {
  Kernel,
  createStorageBuffer,
  createUniformBuffer,
  workgroups,
  PLACEMENT_KERNELS,
  ADJ_K,
  SCAN_BLOCK,
  SORT_BLOCK,
} from "@gaussifier/wgsl";
import type { Placement, PlacementInputs, PlacementOutputs, PlacementParams } from "./types.js";
import { jfaSchedule, mergePlan } from "./schedule.js";

type KernelName = keyof typeof PLACEMENT_KERNELS;
type Kernels = Record<KernelName, Kernel>;

const NEG_ONE = 0xffffffff;

class Uniforms {
  private readonly cache = new Map<string, GPUBuffer>();
  constructor(private readonly device: GPUDevice, private readonly width: number, private readonly height: number) {}
  get(n: number, step = 0, aux = 0): GPUBuffer {
    const key = `${n},${step},${aux}`;
    let buffer = this.cache.get(key);
    if (!buffer) {
      buffer = createUniformBuffer(this.device, new Uint32Array([this.width, this.height, n >>> 0, step >>> 0, aux >>> 0, 0, 0, 0]), `u:${key}`);
      this.cache.set(key, buffer);
    }
    return buffer;
  }
  dispose(): void {
    for (const b of this.cache.values()) b.destroy();
    this.cache.clear();
  }
}

/** Exclusive prefix sum over u32 arrays, multi-level, scratch allocated for a maximum length. */
class Scanner {
  private readonly levels: { sums: GPUBuffer; scanned: GPUBuffer }[] = [];
  constructor(device: GPUDevice, private readonly k: Kernels, private readonly u: Uniforms, maxLength: number) {
    let len = maxLength;
    for (;;) {
      const blocks = Math.ceil(len / SCAN_BLOCK);
      this.levels.push({ sums: createStorageBuffer(device, blocks * 4, "scan.sums"), scanned: createStorageBuffer(device, blocks * 4, "scan.scanned") });
      if (blocks <= 1) break;
      len = blocks;
    }
  }
  record(pass: GPUComputePassEncoder, input: GPUBuffer, output: GPUBuffer, n: number, level = 0): void {
    const blocks = Math.ceil(n / SCAN_BLOCK);
    const lv = this.levels[level];
    if (!lv) throw new Error("scan length exceeds prepared scratch");
    this.k.scan_local.dispatch(pass, [this.u.get(n), input, output, lv.sums], blocks);
    if (blocks > 1) {
      this.record(pass, lv.sums, lv.scanned, blocks, level + 1);
      this.k.scan_add.dispatch(pass, [this.u.get(n), output, lv.scanned], workgroups(n, 256));
    }
  }
  dispose(): void {
    for (const lv of this.levels) { lv.sums.destroy(); lv.scanned.destroy(); }
  }
}

/** Stable LSD radix sort (4 x 8 bits) of non-negative float keys with index values. */
class RadixSorter {
  private readonly keysA: GPUBuffer; private readonly valsA: GPUBuffer;
  private readonly keysB: GPUBuffer; private readonly valsB: GPUBuffer;
  private readonly hist: GPUBuffer; private readonly histScan: GPUBuffer;
  private readonly scanner: Scanner;
  readonly rank: GPUBuffer; readonly idxOfRank: GPUBuffer;
  constructor(device: GPUDevice, private readonly k: Kernels, private readonly u: Uniforms, nMax: number) {
    const blocks = Math.ceil(nMax / SORT_BLOCK);
    this.keysA = createStorageBuffer(device, nMax * 4, "sort.keysA"); this.valsA = createStorageBuffer(device, nMax * 4, "sort.valsA");
    this.keysB = createStorageBuffer(device, nMax * 4, "sort.keysB"); this.valsB = createStorageBuffer(device, nMax * 4, "sort.valsB");
    this.hist = createStorageBuffer(device, 256 * blocks * 4, "sort.hist"); this.histScan = createStorageBuffer(device, 256 * blocks * 4, "sort.histScan");
    this.scanner = new Scanner(device, k, u, 256 * blocks);
    this.rank = createStorageBuffer(device, nMax * 4, "rank"); this.idxOfRank = createStorageBuffer(device, nMax * 4, "idxOfRank");
  }
  /** Sorts `mass[0:n]`; leaves rank and idxOfRank valid for n entries. */
  record(pass: GPUComputePassEncoder, mass: GPUBuffer, n: number): void {
    const blocks = Math.ceil(n / SORT_BLOCK);
    this.k.sort_init.dispatch(pass, [this.u.get(n), mass, this.keysA, this.valsA], workgroups(n, 256));
    let kin = this.keysA, vin = this.valsA, kout = this.keysB, vout = this.valsB;
    for (let pass8 = 0; pass8 < 4; pass8++) {
      const shift = pass8 * 8;
      this.k.sort_hist.dispatch(pass, [this.u.get(n, shift, blocks), kin, this.hist], blocks);
      this.scanner.record(pass, this.hist, this.histScan, 256 * blocks);
      this.k.sort_scatter.dispatch(pass, [this.u.get(n, shift, blocks), kin, vin, this.histScan, kout, vout], blocks);
      [kin, kout] = [kout, kin]; [vin, vout] = [vout, vin];
    }
    // four passes: result is back in A
    this.k.rank_from_sorted.dispatch(pass, [this.u.get(n), vin, this.rank, this.idxOfRank], workgroups(n, 256));
  }
  /** Sorted values buffer after record (ascending key, ties by index). */
  dispose(): void {
    for (const b of [this.keysA, this.valsA, this.keysB, this.valsB, this.hist, this.histScan, this.rank, this.idxOfRank]) b.destroy();
    this.scanner.dispose();
  }
}

export class GpuPlacement implements Placement {
  private readonly k: Kernels;
  private params: PlacementParams | null = null;
  private u!: Uniforms;
  private scanner!: Scanner;
  private sorter!: RadixSorter;
  private schedule: number[] = [];
  private plan: { nIn: number; remove: number }[] = [];
  private b: Record<string, GPUBuffer> = {};

  constructor(readonly device: GPUDevice) {
    this.k = Object.fromEntries(Object.entries(PLACEMENT_KERNELS).map(([name, spec]) => [name, Kernel.create(device, spec)])) as Kernels;
  }


  prepare(params: PlacementParams): void {
    if (params.knnK !== ADJ_K) throw new Error(`knnK must be ${ADJ_K} in v1`);
    if (params.nMax < params.n || params.n < 1) throw new Error("nMax must be >= n >= 1");
    this.disposeBuffers();
    const { width, height, nMax, n } = params;
    const hw = width * height;
    this.params = params;
    this.u = new Uniforms(this.device, width, height);
    this.schedule = jfaSchedule(width, height);
    this.plan = mergePlan(nMax, n, params.mergeRounds, params.perRoundFraction);
    const d = this.device;
    const B = (name: string, bytes: number) => { this.b[name] = createStorageBuffer(d, bytes, name); return this.b[name]; };
    B("pix", nMax * 8);
    B("ownerA", hw * 4); B("ownerB", hw * 4);
    B("seen", nMax * 4); B("flags", nMax * 4); B("scanOut", nMax * 4); B("lost", nMax * 4); B("counts", 16);
    B("acc", 6 * nMax * 4);
    B("mass", nMax * 4);
    B("slot", nMax * ADJ_K * 4);
    B("tmin", nMax * 4); B("proposed", nMax * 4); B("target", nMax * 4); B("matched", nMax * 4); B("alive", nMax * 4);
    B("ptsB", nMax * 8); B("ptsC", nMax * 8); B("ptsNew", nMax * 8);
    B("partial", Math.ceil(nMax / 1024) * 4); B("total", 16);
    B("weights", nMax * 4);
    B("outPoints", n * 8); B("outWeights", n * 4); B("outMasses", n * 4); B("outOwner", hw * 4);
    this.scanner = new Scanner(d, this.k, this.u, nMax);
    this.sorter = new RadixSorter(d, this.k, this.u, nMax);
    // Pre-create the uniforms the static schedule will use.
    this.u.get(0); this.u.get(hw, 0, NEG_ONE); this.u.get(n); this.u.get(2 * n); this.u.get(6 * n);
    for (const s of this.schedule) this.u.get(0, s);
    for (const r of this.plan) {
      if (r.remove === 0) continue;
      this.u.get(r.nIn); this.u.get(2 * r.nIn); this.u.get(6 * r.nIn); this.u.get(r.nIn * ADJ_K, 0, NEG_ONE); this.u.get(r.nIn, 0, NEG_ONE); this.u.get(r.nIn, 0, 1);
      for (let c = 0; c < ADJ_K; c++) { this.u.get(r.nIn, c); this.u.get(r.nIn, c, r.remove); }
      this.u.get(r.nIn, 0, r.remove);
      const blocks = Math.ceil(r.nIn / SORT_BLOCK);
      for (let s = 0; s < 32; s += 8) this.u.get(r.nIn, s, blocks);
      this.u.get(256 * blocks);
    }
    this.u.get(Math.ceil(n / 1024));
  }

  private need(): PlacementParams {
    if (!this.params) throw new Error("call prepare() first");
    return this.params;
  }

  private fill(pass: GPUComputePassEncoder, buffer: GPUBuffer, count: number, value: number): void {
    if (count <= 0) return;
    this.k.fill_u32.dispatch(pass, [this.u.get(count, 0, value), buffer], workgroups(count, 256));
  }

  private grid2d(): [number, number] {
    const p = this.need();
    return [Math.ceil(p.width / 16), Math.ceil(p.height / 16)];
  }

  /** Assignment of `n` normalized points; returns the internal owner buffer holding the result. */
  private assignment(pass: GPUComputePassEncoder, points: GPUBuffer, n: number, recovery: boolean): GPUBuffer {
    const p = this.need();
    const hw = p.width * p.height;
    const [gx, gy] = this.grid2d();
    const b = this.b;
    this.k.to_pixel.dispatch(pass, [this.u.get(n), points, b.pix], workgroups(n, 256));
    this.fill(pass, b.ownerA, hw, NEG_ONE);
    this.k.owner_init.dispatch(pass, [this.u.get(n), b.pix, b.ownerA], workgroups(n, 256));
    let cur = b.ownerA, nxt = b.ownerB;
    for (const step of this.schedule) {
      this.k.jfa_pass.dispatch(pass, [this.u.get(0, step), b.pix, cur, nxt], gx, gy);
      [cur, nxt] = [nxt, cur];
    }
    if (recovery) {
      this.fill(pass, b.seen, n, 0);
      this.k.mark_seen.dispatch(pass, [this.u.get(0), cur, b.seen], gx, gy);
      this.k.unseen_flags.dispatch(pass, [this.u.get(n), b.seen, b.flags], workgroups(n, 256));
      this.scanner.record(pass, b.flags, b.scanOut, n);
      this.k.compact_lost.dispatch(pass, [this.u.get(n), b.flags, b.scanOut, b.lost, b.counts], workgroups(n, 256));
      this.k.recovery.dispatch(pass, [this.u.get(0), b.pix, cur, nxt, b.lost, b.counts], gx, gy);
      [cur, nxt] = [nxt, cur];
    }
    return cur;
  }

  /** Cell masses of `n` points from an owner map into the internal mass buffer. */
  private masses(pass: GPUComputePassEncoder, owner: GPUBuffer, density: GPUBuffer, n: number): GPUBuffer {
    const [gx, gy] = this.grid2d();
    const b = this.b;
    this.fill(pass, b.acc, 2 * n, 0);
    this.k.mass_accumulate.dispatch(pass, [this.u.get(n), owner, density, b.acc], gx, gy);
    this.k.mass_decode.dispatch(pass, [this.u.get(n), b.acc, b.mass], workgroups(n, 256));
    return b.mass;
  }

  private adjacency(pass: GPUComputePassEncoder, owner: GPUBuffer, rank: GPUBuffer, n: number): GPUBuffer {
    const [gx, gy] = this.grid2d();
    const b = this.b;
    this.fill(pass, b.slot, n * ADJ_K, NEG_ONE);
    for (let r = 0; r < ADJ_K; r++) this.k.adjacency_round.dispatch(pass, [this.u.get(n, r), owner, rank, b.slot], gx, gy);
    return b.slot;
  }

  private matching(pass: GPUComputePassEncoder, rank: GPUBuffer, idxOfRank: GPUBuffer, slot: GPUBuffer, n: number, remove: number): void {
    const b = this.b;
    const wg = workgroups(n, 256);
    this.fill(pass, b.alive, n, 1); this.fill(pass, b.matched, n, 0); this.fill(pass, b.target, n, NEG_ONE);
    for (let col = 0; col < ADJ_K; col++) {
      this.fill(pass, b.tmin, n, NEG_ONE);
      this.k.match_phase1.dispatch(pass, [this.u.get(n, col, remove), rank, slot, idxOfRank, b.alive, b.matched, b.tmin, b.proposed], wg);
      this.k.match_phase2.dispatch(pass, [this.u.get(n), rank, b.proposed, b.tmin, b.target, b.matched, b.alive], wg);
    }
    this.k.match_force.dispatch(pass, [this.u.get(n, 0, remove), rank, idxOfRank, b.target, b.matched, b.alive], wg);
  }

  private mergeRound(pass: GPUComputePassEncoder, ptsIn: GPUBuffer, nIn: number, remove: number, density: GPUBuffer, ptsOut: GPUBuffer): void {
    const b = this.b;
    const wg = workgroups(nIn, 256);
    const owner = this.assignment(pass, ptsIn, nIn, true);
    const mass = this.masses(pass, owner, density, nIn);
    this.sorter.record(pass, mass, nIn);
    const slot = this.adjacency(pass, owner, this.sorter.rank, nIn);
    this.matching(pass, this.sorter.rank, this.sorter.idxOfRank, slot, nIn, remove);
    this.k.merge_acc_init.dispatch(pass, [this.u.get(nIn), ptsIn, mass, b.acc], wg);
    this.k.merge_accumulate.dispatch(pass, [this.u.get(nIn), ptsIn, mass, b.matched, b.target, b.acc], wg);
    this.k.merge_new.dispatch(pass, [this.u.get(nIn), ptsIn, b.acc, b.ptsNew], wg);
    this.scanner.record(pass, b.alive, b.scanOut, nIn);
    this.k.compact_alive.dispatch(pass, [this.u.get(nIn), b.alive, b.scanOut, b.ptsNew, ptsOut], wg);
  }

  private lloyd(pass: GPUComputePassEncoder, points: GPUBuffer, n: number, density: GPUBuffer, iters: number): void {
    const [gx, gy] = this.grid2d();
    const b = this.b;
    for (let it = 0; it < iters; it++) {
      const owner = this.assignment(pass, points, n, false);
      this.fill(pass, b.acc, 6 * n, 0);
      this.k.lloyd_accumulate.dispatch(pass, [this.u.get(n), owner, density, b.acc], gx, gy);
      this.k.lloyd_finalize.dispatch(pass, [this.u.get(n), b.acc, points], workgroups(n, 256));
    }
  }

  private weights(pass: GPUComputePassEncoder, mass: GPUBuffer, n: number): GPUBuffer {
    const b = this.b;
    const blocks = Math.ceil(n / 1024);
    this.k.reduce_partial.dispatch(pass, [this.u.get(n), mass, b.partial], blocks);
    this.k.reduce_final.dispatch(pass, [this.u.get(blocks), b.partial, b.total], 1);
    this.k.weights_finalize.dispatch(pass, [this.u.get(n), mass, b.total, b.weights], workgroups(n, 256));
    return b.weights;
  }

  record(encoder: GPUCommandEncoder, inputs: PlacementInputs): PlacementOutputs {
    const p = this.need();
    const b = this.b;
    const hw = p.width * p.height;
    encoder.copyBufferToBuffer(inputs.points, 0, b.ptsB, 0, p.nMax * 8);
    const pass = encoder.beginComputePass({ label: "placement" });
    let cur = b.ptsB;
    let other = b.ptsC;
    for (const round of this.plan) {
      if (round.remove === 0) continue;
      this.mergeRound(pass, cur, round.nIn, round.remove, inputs.density, other);
      [cur, other] = [other, cur];
    }
    this.lloyd(pass, cur, p.n, inputs.density, p.lloydIters);
    const owner = this.assignment(pass, cur, p.n, true);
    const mass = this.masses(pass, owner, inputs.density, p.n);
    const weights = this.weights(pass, mass, p.n);
    pass.end();
    encoder.copyBufferToBuffer(cur, 0, b.outPoints, 0, p.n * 8);
    encoder.copyBufferToBuffer(mass, 0, b.outMasses, 0, p.n * 4);
    encoder.copyBufferToBuffer(weights, 0, b.outWeights, 0, p.n * 4);
    encoder.copyBufferToBuffer(owner, 0, b.outOwner, 0, hw * 4);
    return { points: b.outPoints, weights: b.outWeights, owner: b.outOwner, masses: b.outMasses };
  }

  recordAssignment(encoder: GPUCommandEncoder, points: GPUBuffer, count: number, owner: GPUBuffer): void {
    const p = this.need();
    if (count > p.nMax) throw new Error("count exceeds prepared nMax");
    const pass = encoder.beginComputePass({ label: "assignment" });
    const result = this.assignment(pass, points, count, true);
    pass.end();
    encoder.copyBufferToBuffer(result, 0, owner, 0, p.width * p.height * 4);
  }

  /** Test hooks. Returned buffers are internal scratch, valid until the next record call. */
  readonly debug = {
    masses: (encoder: GPUCommandEncoder, owner: GPUBuffer, density: GPUBuffer, n: number): GPUBuffer => {
      const pass = encoder.beginComputePass(); const m = this.masses(pass, owner, density, n); pass.end(); return m;
    },
    rank: (encoder: GPUCommandEncoder, mass: GPUBuffer, n: number): { rank: GPUBuffer; idxOfRank: GPUBuffer } => {
      const pass = encoder.beginComputePass(); this.sorter.record(pass, mass, n); pass.end();
      return { rank: this.sorter.rank, idxOfRank: this.sorter.idxOfRank };
    },
    adjacency: (encoder: GPUCommandEncoder, owner: GPUBuffer, rank: GPUBuffer, n: number): GPUBuffer => {
      const pass = encoder.beginComputePass(); const s = this.adjacency(pass, owner, rank, n); pass.end(); return s;
    },
    matching: (encoder: GPUCommandEncoder, rank: GPUBuffer, idxOfRank: GPUBuffer, slot: GPUBuffer, n: number, remove: number): { target: GPUBuffer; matched: GPUBuffer; alive: GPUBuffer } => {
      const pass = encoder.beginComputePass(); this.matching(pass, rank, idxOfRank, slot, n, remove); pass.end();
      return { target: this.b.target, matched: this.b.matched, alive: this.b.alive };
    },
    mergeRound: (encoder: GPUCommandEncoder, ptsIn: GPUBuffer, nIn: number, remove: number, density: GPUBuffer): GPUBuffer => {
      const pass = encoder.beginComputePass(); this.mergeRound(pass, ptsIn, nIn, remove, density, this.b.ptsC); pass.end(); return this.b.ptsC;
    },
    assignmentNoRecovery: (encoder: GPUCommandEncoder, points: GPUBuffer, n: number, owner: GPUBuffer): void => {
      const pass = encoder.beginComputePass(); const r = this.assignment(pass, points, n, false); pass.end();
      encoder.copyBufferToBuffer(r, 0, owner, 0, this.need().width * this.need().height * 4);
    },
    /** Merge rounds only (no Lloyd); returns a buffer holding n merged points. */
    mergeOnly: (encoder: GPUCommandEncoder, inputs: PlacementInputs): GPUBuffer => {
      const p = this.need(); const b = this.b;
      encoder.copyBufferToBuffer(inputs.points, 0, b.ptsB, 0, p.nMax * 8);
      const pass = encoder.beginComputePass();
      let cur = b.ptsB, other = b.ptsC;
      for (const round of this.plan) { if (round.remove === 0) continue; this.mergeRound(pass, cur, round.nIn, round.remove, inputs.density, other); [cur, other] = [other, cur]; }
      pass.end();
      return cur;
    },
    plan: (): { nIn: number; remove: number }[] => this.plan.slice(),
    schedule: (): number[] => this.schedule.slice(),
  };

  private disposeBuffers(): void {
    for (const buf of Object.values(this.b)) buf.destroy();
    this.b = {};
    this.scanner?.dispose();
    this.sorter?.dispose();
    this.u?.dispose();
    for (const kernel of Object.values(this.k)) kernel.clearCache();
  }

  dispose(): void {
    this.disposeBuffers();
    this.params = null;
  }
}

export function createPlacement(device: GPUDevice): GpuPlacement {
  return new GpuPlacement(device);
}
