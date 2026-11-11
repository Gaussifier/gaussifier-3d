/**
 * Shared WGSL sources, each a template string that kernels paste into their code: the splat
 * contribution (used by the engine's splatter and the viewer's WebGPU backend, so the production
 * formula has one implementation), Gaussian projection, bilinear sampling, and the placement
 * kernels in ./placement.ts.
 */

/** Per-Gaussian contribution to a pixel, production simple-sum rules. */
export const SPLAT_CONTRIBUTION_WGSL = `
// sigma = 0.5 (cxx dx^2 + cyy dy^2) + cxy dx dy; alpha = exp(-sigma); cutoff 1/255.
fn splat_alpha(dx: f32, dy: f32, conic: vec3<f32>) -> f32 {
  let sigma = 0.5 * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy;
  if (sigma < 0.0) { return 0.0; }
  let alpha = exp(-sigma);
  if (alpha < 0.00392156862745098) { return 0.0; }
  return alpha;
}
`;

/** Project scale/rotation to conic and radius with the native tile rules. */
export const PROJECT_GAUSSIAN_WGSL = `
struct Projected { center: vec2<f32>, conic: vec3<f32>, radius: f32, valid: u32 };
fn project_gaussian(xy_norm: vec2<f32>, scale: vec2<f32>, rot: f32, width: f32, height: f32) -> Projected {
  var out: Projected;
  out.center = vec2<f32>(xy_norm.x * width, xy_norm.y * height);
  let c = cos(rot); let s = sin(rot);
  let m00 = c * scale.x; let m01 = s * scale.y; let m10 = -s * scale.x; let m11 = c * scale.y;
  let sxx = m00 * m00 + m01 * m01; let sxy = m00 * m10 + m01 * m11; let syy = m10 * m10 + m11 * m11;
  let det = sxx * syy - sxy * sxy;
  if (det == 0.0) { out.valid = 0u; out.radius = 0.0; out.conic = vec3<f32>(0.0); return out; }
  let inv = 1.0 / det;
  out.conic = vec3<f32>(syy * inv, -sxy * inv, sxx * inv);
  let bm = 0.5 * (sxx + syy);
  let root = sqrt(max(0.1, bm * bm - det));
  out.radius = ceil(3.0 * sqrt(max(bm + root, bm - root)));
  out.valid = 1u;
  return out;
}
`;

/** Bilinear sample with align_corners=false and zero padding, one channel plane. */
export const BILINEAR_WGSL = `
fn bilinear_plane(plane_offset: u32, width: u32, height: u32, x_norm: f32, y_norm: f32) -> f32 {
  let xp = x_norm * f32(width) - 0.5; let yp = y_norm * f32(height) - 0.5;
  let x0 = i32(floor(xp)); let y0 = i32(floor(yp)); let fx = xp - f32(x0); let fy = yp - f32(y0);
  var v = 0.0;
  for (var j = 0; j < 2; j++) {
    for (var k = 0; k < 2; k++) {
      let xx = x0 + k; let yy = y0 + j;
      if (xx >= 0 && xx < i32(width) && yy >= 0 && yy < i32(height)) {
        let w = select(1.0 - fx, fx, k == 1) * select(1.0 - fy, fy, j == 1);
        v += w * MAP[plane_offset + u32(yy) * width + u32(xx)];
      }
    }
  }
  return v;
}
`;

export * from "./placement.js";
