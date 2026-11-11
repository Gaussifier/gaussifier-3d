/**
 * WGSL kernels for the K-state loop. Base formulas come from @gaussifier/wgsl so
 * the splat contribution, projection, and bilinear sampling stay shared with the
 * viewer and the placement stage.
 */
import { BILINEAR_WGSL, PROJECT_GAUSSIAN_WGSL, SPLAT_CONTRIBUTION_WGSL } from "@gaussifier/wgsl";

/**
 * Channel layout of the features buffer, the head's [17, H, W] input: first channel of each
 * group. The static part (image through rgb) is packed once per run; the splatter writes
 * rendered and diff every state, and the hard count kernel writes its channel before the head.
 */
export const FEATURES = { image: 0, density: 3, logcov: 4, rgb: 7, rendered: 10, diff: 13, hard: 16, count: 17 } as const;

/** Tile rules of the native renderer: 16-pixel tiles, and a Gaussian covering more than 32 tiles goes to the "large" list. */
export const TILE = 16;
export const LARGE_TILE_THRESHOLD = 32;

export const PARAMS_WGSL = `
struct Params { W: u32, H: u32, N: u32, tilesX: u32, tilesY: u32, capacity: u32, xy_step: f32, cov_shift: f32 };
`;

/** Layout of Params as written by the host (8 x 32-bit). */
export function packParams(p: { W: number; H: number; N: number; tilesX: number; tilesY: number; capacity: number; xyStep: number; covShift?: number }): ArrayBufferView {
  const buf = new ArrayBuffer(32);
  const u = new Uint32Array(buf);
  const f = new Float32Array(buf);
  u[0] = p.W; u[1] = p.H; u[2] = p.N; u[3] = p.tilesX; u[4] = p.tilesY; u[5] = p.capacity;
  f[6] = p.xyStep; f[7] = p.covShift ?? 0;
  return new Uint8Array(buf);
}

/**
 * The static channels of the features buffer: image, density, log_cov, rgb. Written once per run.
 * cov_shift is added to the log_cov diagonal (channels xx and yy), scaling Sigma by exp(cov_shift);
 * the initial attributes sample these channels, so the head and the Gaussians see the same map.
 */
export const PACK_STATIC_WGSL = `
${PARAMS_WGSL}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> image: array<f32>;
@group(0) @binding(2) var<storage, read> density: array<f32>;
@group(0) @binding(3) var<storage, read> logcov: array<f32>;
@group(0) @binding(4) var<storage, read> rgbmap: array<f32>;
@group(0) @binding(5) var<storage, read_write> features: array<f32>;
@compute @workgroup_size(256) fn pack_static(@builtin(global_invocation_id) g: vec3<u32>) {
  let HW = p.W * p.H; let i = g.x; if (i >= HW) { return; }
  features[i] = image[i]; features[HW + i] = image[HW + i]; features[2u * HW + i] = image[2u * HW + i];
  features[${FEATURES.density}u * HW + i] = density[i];
  for (var c = 0u; c < 3u; c++) {
    features[(${FEATURES.logcov}u + c) * HW + i] = logcov[c * HW + i] + select(0.0, p.cov_shift, c != 1u);
    features[(${FEATURES.rgb}u + c) * HW + i] = rgbmap[c * HW + i];
  }
}
`;

/** Initial per-point attributes: bilinear samples of the log_cov and rgb channels (rgb clamped). */
export const INIT_ATTRS_WGSL = `
${PARAMS_WGSL}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> xy: array<f32>;
@group(0) @binding(2) var<storage, read> MAP: array<f32>;
@group(0) @binding(3) var<storage, read_write> cov: array<f32>;
@group(0) @binding(4) var<storage, read_write> rgb: array<f32>;
${BILINEAR_WGSL}
@compute @workgroup_size(256) fn init_attrs(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.N) { return; }
  let HW = p.W * p.H; let x = xy[2u * i]; let y = xy[2u * i + 1u];
  for (var c = 0u; c < 3u; c++) {
    cov[3u * i + c] = bilinear_plane((${FEATURES.logcov}u + c) * HW, p.W, p.H, x, y);
    rgb[3u * i + c] = clamp(bilinear_plane((${FEATURES.rgb}u + c) * HW, p.W, p.H, x, y), 0.0, 1.0);
  }
}
`;

/** Log-covariance to scale and rotation, and weighted color. */
export const PRE_RENDER_WGSL = `
${PARAMS_WGSL}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> cov: array<f32>;
@group(0) @binding(2) var<storage, read> rgb: array<f32>;
@group(0) @binding(3) var<storage, read> weight: array<f32>;
@group(0) @binding(4) var<storage, read_write> scale: array<f32>;
@group(0) @binding(5) var<storage, read_write> rot: array<f32>;
@group(0) @binding(6) var<storage, read_write> color: array<f32>;
@compute @workgroup_size(256) fn pre_render(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.N) { return; }
  let a = cov[3u * i]; let b = cov[3u * i + 1u]; let c = cov[3u * i + 2u];
  let center = 0.5 * (a + c); let half = 0.5 * (a - c);
  let radius = sqrt(half * half + b * b + 1e-12);
  scale[2u * i] = exp(0.5 * (center + radius));
  scale[2u * i + 1u] = exp(0.5 * (center - radius));
  rot[i] = -0.5 * atan2(2.0 * b, a - c);
  let w = weight[i];
  for (var k = 0u; k < 3u; k++) { color[3u * i + k] = clamp(rgb[3u * i + k] * w, 0.0, 1.0); }
}
`;

const TILE_BBOX_WGSL = `
// Native get_tile_bbox: tile-space center and radius, truncation toward zero, clamp to [0, tiles].
fn tile_bbox(center: vec2<f32>, radius: f32) -> vec4<i32> {
  let tcx = center.x / ${TILE}.0; let tcy = center.y / ${TILE}.0; let tr = radius / ${TILE}.0;
  let minx = clamp(i32(tcx - tr), 0, i32(p.tilesX)); let maxx = clamp(i32(tcx + tr + 1.0), 0, i32(p.tilesX));
  let miny = clamp(i32(tcy - tr), 0, i32(p.tilesY)); let maxy = clamp(i32(tcy + tr + 1.0), 0, i32(p.tilesY));
  return vec4<i32>(minx, miny, maxx, maxy);
}
`;

/** Project every Gaussian, count tile hits, and collect large footprints. proj stride is 8 floats. */
export const PROJECT_WGSL = `
${PARAMS_WGSL}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> xy: array<f32>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read> rot: array<f32>;
@group(0) @binding(4) var<storage, read_write> proj: array<f32>;
@group(0) @binding(5) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> large_list: array<u32>;
@group(0) @binding(7) var<storage, read_write> flags: array<atomic<u32>>;
${PROJECT_GAUSSIAN_WGSL}
${TILE_BBOX_WGSL}
@compute @workgroup_size(256) fn project(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.N) { return; }
  let pr = project_gaussian(vec2<f32>(xy[2u * i], xy[2u * i + 1u]), vec2<f32>(scale[2u * i], scale[2u * i + 1u]), rot[i], f32(p.W), f32(p.H));
  let o = 8u * i;
  proj[o] = pr.center.x; proj[o + 1u] = pr.center.y;
  proj[o + 2u] = pr.conic.x; proj[o + 3u] = pr.conic.y; proj[o + 4u] = pr.conic.z;
  proj[o + 5u] = pr.radius; proj[o + 6u] = f32(pr.valid); proj[o + 7u] = 0.0;
  if (pr.valid == 0u || pr.radius <= 0.0) { return; }
  let bb = tile_bbox(pr.center, pr.radius);
  let footprint = (bb.z - bb.x) * (bb.w - bb.y);
  if (footprint > ${LARGE_TILE_THRESHOLD}) { let slot = atomicAdd(&flags[0], 1u); large_list[slot] = i; return; }
  for (var ty = bb.y; ty < bb.w; ty++) {
    for (var tx = bb.x; tx < bb.z; tx++) { atomicAdd(&counts[u32(ty) * p.tilesX + u32(tx)], 1u); }
  }
}
`;

/** Scatter Gaussian ids into per-tile lists using atomic cursors; set the overflow flag past capacity. */
export const SCATTER_WGSL = `
${PARAMS_WGSL}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> proj: array<f32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> cursors: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> list: array<u32>;
@group(0) @binding(5) var<storage, read_write> flags: array<atomic<u32>>;
${TILE_BBOX_WGSL}
@compute @workgroup_size(256) fn scatter(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.N) { return; }
  let o = 8u * i;
  let radius = proj[o + 5u];
  if (proj[o + 6u] < 0.5 || radius <= 0.0) { return; }
  let bb = tile_bbox(vec2<f32>(proj[o], proj[o + 1u]), radius);
  let footprint = (bb.z - bb.x) * (bb.w - bb.y);
  if (footprint > ${LARGE_TILE_THRESHOLD}) { return; }
  for (var ty = bb.y; ty < bb.w; ty++) {
    for (var tx = bb.x; tx < bb.z; tx++) {
      let t = u32(ty) * p.tilesX + u32(tx);
      let w = atomicAdd(&cursors[t], 1u) + offsets[t];
      if (w < p.capacity) { list[w] = i; } else { atomicStore(&flags[1], 1u); }
    }
  }
}
`;

/**
 * Per-tile raster. Writes the clamped render into features channels [10:13]
 * and image - rendered into [13:16]. flags = [large_count, overflow].
 */
export const RASTER_WGSL = `
${PARAMS_WGSL}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> proj: array<f32>;
@group(0) @binding(2) var<storage, read> color: array<f32>;
@group(0) @binding(3) var<storage, read> list: array<u32>;
@group(0) @binding(4) var<storage, read> offsets: array<u32>;
@group(0) @binding(5) var<storage, read> large_list: array<u32>;
@group(0) @binding(6) var<storage, read> flags: array<u32>;
@group(0) @binding(7) var<storage, read_write> features: array<f32>;
${SPLAT_CONTRIBUTION_WGSL}
var<workgroup> s_x: array<f32, 256>;
var<workgroup> s_y: array<f32, 256>;
var<workgroup> s_cx: array<f32, 256>;
var<workgroup> s_cy: array<f32, 256>;
var<workgroup> s_cz: array<f32, 256>;
var<workgroup> s_r: array<f32, 256>;
var<workgroup> s_g: array<f32, 256>;
var<workgroup> s_b: array<f32, 256>;
var<workgroup> s_ok: array<u32, 256>;
fn stage(gid: u32, li: u32) {
  let o = 8u * gid;
  s_x[li] = proj[o]; s_y[li] = proj[o + 1u];
  s_cx[li] = proj[o + 2u]; s_cy[li] = proj[o + 3u]; s_cz[li] = proj[o + 4u];
  s_ok[li] = u32(proj[o + 6u] > 0.5);
  s_r[li] = color[3u * gid]; s_g[li] = color[3u * gid + 1u]; s_b[li] = color[3u * gid + 2u];
}
// Sum the Gaussians with indices [start, end) into acc, staged 256 at a time through workgroup memory.
// source 0: indices are gid directly (overflow fallback over all N); 1: the tile list; 2: the large list.
fn accumulate(start: u32, end: u32, source: u32, li: u32, inside: bool, fpx: f32, fpy: f32, acc: ptr<function, vec3<f32>>) {
  var count = 0u;
  if (end > start) { count = end - start; }
  let blocks = (count + 255u) / 256u;
  for (var b = 0u; b < blocks; b++) {
    let base = start + b * 256u;
    let idx = base + li;
    if (idx < end) {
      var gid = idx;
      if (source == 1u) { gid = list[idx]; } else if (source == 2u) { gid = large_list[idx]; }
      stage(gid, li);
    }
    workgroupBarrier();
    let staged = min(256u, end - base);
    if (inside) {
      for (var t = 0u; t < staged; t++) {
        if (s_ok[t] == 0u) { continue; }
        let a = splat_alpha(s_x[t] - fpx, s_y[t] - fpy, vec3<f32>(s_cx[t], s_cy[t], s_cz[t]));
        *acc += vec3<f32>(s_r[t], s_g[t], s_b[t]) * a;
      }
    }
    workgroupBarrier();
  }
}
@compute @workgroup_size(${TILE}, ${TILE}) fn raster(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let tile = wg.y * p.tilesX + wg.x;
  let px = wg.x * ${TILE}u + lid.x; let py = wg.y * ${TILE}u + lid.y;
  let inside = (px < p.W) && (py < p.H);
  let fpx = f32(px); let fpy = f32(py);
  var acc = vec3<f32>(0.0, 0.0, 0.0);
  if (flags[1] != 0u) {
    // The tile lists overflowed their capacity: every pixel sums every Gaussian.
    accumulate(0u, p.N, 0u, li, inside, fpx, fpy, &acc);
  } else {
    accumulate(offsets[tile], min(offsets[tile + 1u], p.capacity), 1u, li, inside, fpx, fpy, &acc);
    accumulate(0u, flags[0], 2u, li, inside, fpx, fpy, &acc);
  }
  if (inside) {
    let HW = p.W * p.H; let pix = py * p.W + px;
    let out = clamp(acc, vec3<f32>(0.0), vec3<f32>(1.0));
    for (var c = 0u; c < 3u; c++) {
      features[(${FEATURES.rendered}u + c) * HW + pix] = out[c];
      features[(${FEATURES.diff}u + c) * HW + pix] = features[(${FEATURES.image}u + c) * HW + pix] - out[c];
    }
  }
}
`;

/** Hard count: one atomic increment per point at its floor-clamped pixel. */
export const HARD_COUNT_WGSL = `
${PARAMS_WGSL}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> xy: array<f32>;
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
@compute @workgroup_size(256) fn hard_count(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.N) { return; }
  let cx = u32(clamp(floor(xy[2u * i] * f32(p.W)), 0.0, f32(p.W - 1u)));
  let cy = u32(clamp(floor(xy[2u * i + 1u] * f32(p.H)), 0.0, f32(p.H - 1u)));
  atomicAdd(&counts[cy * p.W + cx], 1u);
}
`;

export const HARD_TO_F32_WGSL = `
${PARAMS_WGSL}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> features: array<f32>;
@compute @workgroup_size(256) fn hard_to_f32(@builtin(global_invocation_id) g: vec3<u32>) {
  let HW = p.W * p.H; let i = g.x; if (i >= HW) { return; }
  features[${FEATURES.hard}u * HW + i] = f32(counts[i]);
}
`;

/** Bilinear-sample the 8 delta channels at each point and apply the clamped updates. */
export const POST_HEAD_WGSL = `
${PARAMS_WGSL}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> MAP: array<f32>;
@group(0) @binding(2) var<storage, read_write> xy: array<f32>;
@group(0) @binding(3) var<storage, read_write> cov: array<f32>;
@group(0) @binding(4) var<storage, read_write> rgb: array<f32>;
${BILINEAR_WGSL}
@compute @workgroup_size(256) fn post_head(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.N) { return; }
  let HW = p.W * p.H; let x = xy[2u * i]; let y = xy[2u * i + 1u];
  var d: array<f32, 8>;
  for (var c = 0u; c < 8u; c++) { d[c] = bilinear_plane(c * HW, p.W, p.H, x, y); }
  cov[3u * i] += d[0]; cov[3u * i + 1u] += d[1]; cov[3u * i + 2u] += d[2];
  rgb[3u * i] = clamp(rgb[3u * i] + d[3], 0.0, 1.0);
  rgb[3u * i + 1u] = clamp(rgb[3u * i + 1u] + d[4], 0.0, 1.0);
  rgb[3u * i + 2u] = clamp(rgb[3u * i + 2u] + d[5], 0.0, 1.0);
  xy[2u * i] = clamp(x + p.xy_step * d[6], 0.0, 1.0);
  xy[2u * i + 1u] = clamp(y + p.xy_step * d[7], 0.0, 1.0);
}
`;
