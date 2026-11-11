/**
 * Exclusive prefix sum over the per-tile counts of the splatter, in a single workgroup: tile
 * counts are at most a few thousand, so one workgroup handles them and writes n+1 entries,
 * the exclusive offsets and the total at index n. The placement stage has its own multi-level
 * scanner (placement.ts `Scanner`) for arrays of up to nMax points, which this does not replace.
 */
import { Kernel, createUniformBuffer, type KernelSpec } from "@gaussifier/wgsl";

export const SCAN_WGSL = `
struct ScanParams { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> sp: ScanParams;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
var<workgroup> partials: array<u32, 256>;
@compute @workgroup_size(256) fn scan(@builtin(local_invocation_id) lid: vec3<u32>) {
  let n = sp.n;
  let per = (n + 255u) / 256u;
  let start = lid.x * per;
  var sum = 0u;
  for (var i = 0u; i < per; i++) { let j = start + i; if (j < n) { sum += input[j]; } }
  partials[lid.x] = sum;
  workgroupBarrier();
  for (var offset = 1u; offset < 256u; offset = offset << 1u) {
    var v = partials[lid.x];
    if (lid.x >= offset) { v += partials[lid.x - offset]; }
    workgroupBarrier();
    partials[lid.x] = v;
    workgroupBarrier();
  }
  var run = 0u;
  if (lid.x > 0u) { run = partials[lid.x - 1u]; }
  for (var i = 0u; i < per; i++) { let j = start + i; if (j < n) { output[j] = run; run += input[j]; } }
  if (lid.x == 255u) { output[n] = partials[255u]; }
}
`;

export const SCAN_SPEC: KernelSpec = { code: SCAN_WGSL, entryPoint: "scan", bindings: ["uniform", "read-only-storage", "storage"], label: "scan" };

export class TileScan {
  private kernel: Kernel;
  private params: GPUBuffer;
  constructor(readonly device: GPUDevice, readonly n: number) {
    this.kernel = Kernel.create(device, SCAN_SPEC);
    this.params = createUniformBuffer(device, new Uint32Array([n, 0, 0, 0]), "scan-params");
  }
  /** input: u32[n]; output: u32[n+1]. */
  record(pass: GPUComputePassEncoder, input: GPUBuffer, output: GPUBuffer): void {
    this.kernel.dispatch(pass, [this.params, input, output], 1);
  }
  dispose(): void { this.params.destroy(); }
}
