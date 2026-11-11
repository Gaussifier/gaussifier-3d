/** `out = a - b` over f32 buffers: the residual images the quality loop backpropagates. */
import { Kernel, createUniformBuffer, workgroups } from "@gaussifier/wgsl";

const RESIDUAL_WGSL = /* wgsl */ `
struct Params { count: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn residual(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.count) { return; }
  out[i] = a[i] - b[i];
}
`;

export class Residual {
  private readonly kernel: Kernel;
  private readonly params: GPUBuffer;
  constructor(readonly device: GPUDevice, readonly count: number) {
    this.kernel = Kernel.create(device, { code: RESIDUAL_WGSL, entryPoint: "residual", bindings: ["uniform", "read-only-storage", "read-only-storage", "storage"], label: "planar-residual" });
    this.params = createUniformBuffer(device, new Uint32Array([count, 0, 0, 0]), "planar-residual-params");
  }

  record(encoder: GPUCommandEncoder, a: GPUBuffer, b: GPUBuffer, out: GPUBuffer): void {
    const pass = encoder.beginComputePass({ label: "planar-residual" });
    this.kernel.dispatch(pass, [this.params, a, b, out], workgroups(this.count, 256));
    pass.end();
  }

  dispose(): void {
    this.params.destroy();
    this.kernel.clearCache();
  }
}
