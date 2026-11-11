/**
 * The quality model's 30-degree target: the input image painted on the z = 0 plane and seen from
 * the gradient view (`plane_warp(image, camera, black, supersample=4, mode="bicubic")` of the
 * planar loop). Each output pixel casts ss x ss rays, samples the image like `grid_sample`
 * (bicubic, A = -0.75, border padding, align_corners=False), clamps the cubic overshoot, masks rays
 * that miss the image, and box-filters the rays.
 */
import { Kernel, createUniformBuffer, workgroups } from "@gaussifier/wgsl";
import type { PlanarCamera } from "./bundle.js";

const WARP_WGSL = /* wgsl */ `
struct Params {
  r0: vec4<f32>, r1: vec4<f32>, r2: vec4<f32>,   // rows of R (w2c rotation), w unused
  pos: vec4<f32>,                                 // camera position, w unused
  fx: f32, fy: f32, cx: f32, cy: f32,
  out_w: u32, out_h: u32, src_w: u32, src_h: u32,
  ss: u32, _p0: u32, _p1: u32, _p2: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> image: array<f32>;   // (3, src_h, src_w)
@group(0) @binding(2) var<storage, read_write> out: array<f32>; // (3, out_h, out_w)

fn tap(c: u32, x: i32, y: i32) -> f32 {
  let xx = clamp(x, 0, i32(p.src_w) - 1);
  let yy = clamp(y, 0, i32(p.src_h) - 1);
  return image[(c * p.src_h + u32(yy)) * p.src_w + u32(xx)];
}

// PyTorch's cubic convolution coefficients, A = -0.75.
fn cc1(x: f32) -> f32 { let a = -0.75; return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0; }
fn cc2(x: f32) -> f32 { let a = -0.75; return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a; }
fn cubic(v0: f32, v1: f32, v2: f32, v3: f32, t: f32) -> f32 {
  return v0 * cc2(t + 1.0) + v1 * cc1(t) + v2 * cc1(1.0 - t) + v3 * cc2(2.0 - t);
}

@compute @workgroup_size(16, 16)
fn warp(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.out_w || gid.y >= p.out_h) { return; }
  let ssf = f32(p.ss);
  let aspect = f32(p.src_w) / f32(p.src_h);
  var acc = vec3<f32>(0.0);
  for (var j = 0u; j < p.ss; j++) {
    for (var i = 0u; i < p.ss; i++) {
      let u = (f32(gid.x * p.ss + i) + 0.5) / ssf;
      let v = (f32(gid.y * p.ss + j) + 0.5) / ssf;
      let d = vec3<f32>((u - p.cx) / p.fx, (v - p.cy) / p.fy, 1.0);
      // dir_cam @ R: the world direction is R^T d.
      let dw = d.x * p.r0.xyz + d.y * p.r1.xyz + d.z * p.r2.xyz;
      let s = -p.pos.z / dw.z;
      if (!(s > 0.0) || abs(s) > 3.0e38) { continue; }
      let hit = p.pos.xyz + s * dw;
      let gx = 2.0 * hit.x / aspect;
      let gy = 2.0 * hit.y;
      if (abs(gx) > 1.0 || abs(gy) > 1.0) { continue; }
      // grid_sample unnormalize, align_corners=False.
      let ix = ((gx + 1.0) * f32(p.src_w) - 1.0) * 0.5;
      let iy = ((gy + 1.0) * f32(p.src_h) - 1.0) * 0.5;
      let x0 = floor(ix);
      let y0 = floor(iy);
      let tx = ix - x0;
      let ty = iy - y0;
      let bx = i32(x0);
      let by = i32(y0);
      for (var c = 0u; c < 3u; c++) {
        var rows: array<f32, 4>;
        for (var k = 0; k < 4; k++) {
          let y = by - 1 + k;
          rows[k] = cubic(tap(c, bx - 1, y), tap(c, bx, y), tap(c, bx + 1, y), tap(c, bx + 2, y), tx);
        }
        acc[c] += clamp(cubic(rows[0], rows[1], rows[2], rows[3], ty), 0.0, 1.0);
      }
    }
  }
  let inv = 1.0 / (ssf * ssf);
  let o = gid.y * p.out_w + gid.x;
  let plane = p.out_w * p.out_h;
  out[o] = acc.x * inv;
  out[plane + o] = acc.y * inv;
  out[2u * plane + o] = acc.z * inv;
}
`;

export class PlaneWarp {
  private readonly kernel: Kernel;
  constructor(readonly device: GPUDevice) {
    this.kernel = Kernel.create(device, { code: WARP_WGSL, entryPoint: "warp", bindings: ["uniform", "read-only-storage", "storage"], label: "planar-plane-warp" });
  }

  /**
   * Record the warp of `image` (3, srcH, srcW) into `out` (3, camera.height, camera.width) on a
   * black background. Returns the uniform buffer, which the caller destroys after the submit.
   */
  record(encoder: GPUCommandEncoder, image: GPUBuffer, srcWidth: number, srcHeight: number, camera: PlanarCamera, out: GPUBuffer, supersample = 4): GPUBuffer {
    const m = camera.w2c;
    const f = new Float32Array(24);
    f.set([m[0], m[1], m[2], 0, m[4], m[5], m[6], 0, m[8], m[9], m[10], 0], 0);
    f.set([camera.position[0], camera.position[1], camera.position[2], 0], 12);
    f.set([camera.focal_x, camera.focal_y, camera.center_x, camera.center_y], 16);
    const u = new Uint32Array(f.buffer);
    u[20] = camera.width; u[21] = camera.height; u[22] = srcWidth; u[23] = srcHeight;
    const tail = new Uint32Array([supersample, 0, 0, 0]);
    const data = new Uint8Array(28 * 4);
    data.set(new Uint8Array(f.buffer), 0);
    data.set(new Uint8Array(tail.buffer), 24 * 4);
    const params = createUniformBuffer(this.device, data, "planar-plane-warp-params");
    const pass = encoder.beginComputePass({ label: "planar-plane-warp" });
    this.kernel.dispatch(pass, [params, image, out], workgroups(camera.width, 16), workgroups(camera.height, 16));
    pass.end();
    return params;
  }
}
