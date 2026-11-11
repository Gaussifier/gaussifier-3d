/**
 * The head's hard point count, `rasterize_point_count`: 1 at the integer pixel under each point.
 * ONNX has no portable scatter-add, so the engine counts in WGSL and feeds the head graph.
 */
import { Kernel, createStorageBuffer, createUniformBuffer, workgroups } from "@gaussifier/wgsl";

const PARAMS = /* wgsl */ `
struct Params { n: u32, width: u32, height: u32, stride: u32, offset: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<uniform> p: Params;
`;

const COUNT_WGSL = PARAMS + /* wgsl */ `
@group(0) @binding(1) var<storage, read> state: array<f32>;
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;

// torch: (x * W).clamp(0, W - 1e-3).to(long), the same for y, all in f32.
@compute @workgroup_size(256)
fn count_points(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  let fw = f32(p.width);
  let fh = f32(p.height);
  let x = state[i * p.stride + p.offset];
  let y = state[i * p.stride + p.offset + 1u];
  let px = u32(clamp(x * fw, 0.0, fw - 1e-3));
  let py = u32(clamp(y * fh, 0.0, fh - 1e-3));
  atomicAdd(&counts[py * p.width + px], 1u);
}
`;

const TO_FLOAT_WGSL = PARAMS + /* wgsl */ `
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn to_float(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.width * p.height) { return; }
  out[i] = f32(counts[i]);
}
`;

/** Counts the xy columns of an (N, stride) state; one instance per point count and image size. */
export class PointCounter {
  private readonly count: Kernel;
  private readonly toFloat: Kernel;
  private readonly params: GPUBuffer;
  private readonly scratch: GPUBuffer;
  constructor(readonly device: GPUDevice, readonly n: number, readonly width: number, readonly height: number, stride: number, offset: number) {
    this.count = Kernel.create(device, { code: COUNT_WGSL, entryPoint: "count_points", bindings: ["uniform", "read-only-storage", "storage"], label: "planar-count" });
    this.toFloat = Kernel.create(device, { code: TO_FLOAT_WGSL, entryPoint: "to_float", bindings: ["uniform", "read-only-storage", "storage"], label: "planar-count-f32" });
    this.params = createUniformBuffer(device, new Uint32Array([n, width, height, stride, offset, 0, 0, 0]), "planar-count-params");
    this.scratch = createStorageBuffer(device, width * height * 4, "planar-count-scratch");
  }

  /** Record the count of the points in `state` into `out`, an f32 [H*W] buffer. */
  record(encoder: GPUCommandEncoder, state: GPUBuffer, out: GPUBuffer): void {
    encoder.clearBuffer(this.scratch, 0, this.width * this.height * 4);
    const pass = encoder.beginComputePass({ label: "planar-count" });
    this.count.dispatch(pass, [this.params, state, this.scratch], workgroups(this.n, 256));
    this.toFloat.dispatch(pass, [this.params, this.scratch, out], workgroups(this.width * this.height, 256));
    pass.end();
  }

  dispose(): void {
    this.params.destroy();
    this.scratch.destroy();
    this.count.clearCache();
    this.toFloat.clearCache();
  }
}
