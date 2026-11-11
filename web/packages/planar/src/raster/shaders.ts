/**
 * WGSL port of the forward pass of the Faster-GS rasterizer: preprocess (EWA
 * projection, culling, exact tile counting), a depth sort, instance creation, a stable sort by
 * tile, tile ranges, and front-to-back alpha blending in 16x12 tiles with 8x4 subtiles.
 *
 * The CUDA build uses fast math, so exp and log are written as exp2(x * log2 e) and
 * log2(x) * ln 2 (the instruction sequences of __expf and __logf) and divisions as a reciprocal
 * times the numerator. Every other constant and the order of the floating-point operations
 * follow the CUDA kernels; which multiply-adds get fused is up to the shader compiler, so the
 * 2D covariance can differ from CUDA in the last bits.
 *
 * Sorting differs in mechanism only. The CUDA path sorts depth keys with a stable radix sort
 * after an atomic append, so equal depths keep the order in which threads appended them. Here
 * the append is replaced by the primitive index order, which is the CUDA order inside a warp;
 * across warps the CUDA order is scheduling-dependent. The tile sort is a stable radix sort on
 * the tile bits of a key `tile << rankBits | depthRank`, so each tile keeps depth order.
 */

/** Tile and subtile sizes, blend workgroup size, sort block size. */
export const TILE_W = 16;
export const TILE_H = 12;
export const BLEND_THREADS = TILE_W * TILE_H;
export const SORT_THREADS = 128;
export const SORT_ITEMS = 16;
export const SORT_BLOCK = SORT_THREADS * SORT_ITEMS;
export const SORT_RADIX = 16;
export const SCAN_BLOCK = 256;
/** `SortParams.slot` value meaning "count = Params.n". */
export const SLOT_N = 0xffffffff;
/** Byte size of the Params uniform; see `packRasterParams`. */
export const PARAMS_BYTES = 144;
/** Byte size of one `Proj` record. */
export const PROJ_BYTES = 32;

/** Counter slots in the `counters` buffer (u32). */
export const COUNTER = { instances: 0, sortCount: 1, overflow: 2, bucketOverflow: 3 } as const;

const PRELUDE = /* wgsl */ `
struct Params {
  w2c0: vec4f, w2c1: vec4f, w2c2: vec4f, campos: vec4f, bg: vec4f,
  n: u32, width: u32, height: u32, gridW: u32,
  gridH: u32, nTiles: u32, rankBits: u32, capacity: u32,
  fx: f32, fy: f32, cx: f32, cy: f32,
  near: f32, far: f32, widthF: f32, heightF: f32,
};
// One projected primitive. bounds packs the screen rectangle [x0, x1) x [y0, y1) in pixels as
// (x0 | x1 << 16, y0 | y1 << 16); conicOpacity is (conic.xx, conic.xy, conic.yy, opacity).
struct Proj { mean: vec2f, bounds: vec2u, conicOpacity: vec4f };

const TILE_W: u32 = ${TILE_W}u;
const TILE_H: u32 = ${TILE_H}u;
const MIN_ALPHA: f32 = 0.003921568859368563;  // 1 / 255
const MIN_ALPHA_RCP: f32 = 255.0;
const T_MIN: f32 = 1e-4;
const DILATION: f32 = 0.3;
const MIN_DET: f32 = 1e-6;
const SH_C0: f32 = 0.28209479177387814;
const LOG2E: f32 = 1.4426950408889634;
const LN2: f32 = 0.6931471805599453;
const NO_KEY: u32 = 0xffffffffu;

fn fast_exp(x: f32) -> f32 { return exp2(x * LOG2E); }
fn fast_log(x: f32) -> f32 { return log2(x) * LN2; }
// a / b as compiled by fast math: reciprocal, then multiply.
fn fast_div(a: f32, b: f32) -> f32 { return a * (1.0 / b); }

// Tile rectangle [x0, x1) x [y0, y1) in tiles from the packed pixel bounds.
fn tile_bounds(b: vec2u) -> vec4u {
  let x0 = b.x & 0xffffu; let x1 = b.x >> 16u; let y0 = b.y & 0xffffu; let y1 = b.y >> 16u;
  return vec4u(x0 / TILE_W, (x1 + TILE_W - 1u) / TILE_W, y0 / TILE_H, (y1 + TILE_H - 1u) / TILE_H);
}

// Whether the Gaussian reaches alpha >= 1/255 anywhere in the tile: the maximum of the Gaussian
// over the tile's pixel-centre rectangle against the power threshold. mean is mean2d - 0.5.
fn will_contribute(mean: vec2f, conic: vec3f, tx: u32, ty: u32, thr: f32) -> bool {
  let rect_min = vec2f(f32(tx * TILE_W), f32(ty * TILE_H));
  let rect_max = vec2f(f32((tx + 1u) * TILE_W - 1u), f32((ty + 1u) * TILE_H - 1u));
  let x_min_diff = rect_min.x - mean.x;
  let x_left = select(0.0, 1.0, x_min_diff >= 0.0);
  let not_in_x_range = x_left + select(0.0, 1.0, mean.x > rect_max.x);
  let y_min_diff = rect_min.y - mean.y;
  let y_above = select(0.0, 1.0, y_min_diff >= 0.0);
  let not_in_y_range = y_above + select(0.0, 1.0, mean.y > rect_max.y);
  if (not_in_y_range + not_in_x_range == 0.0) { return true; }
  let closest = vec2f(select(rect_max.x, rect_min.x, x_left == 1.0), select(rect_max.y, rect_min.y, y_above == 1.0));
  let diff = mean - closest;
  let d = vec2f(select(-f32(TILE_W - 1u), f32(TILE_W - 1u), x_min_diff >= 0.0),
                select(-f32(TILE_H - 1u), f32(TILE_H - 1u), y_min_diff >= 0.0));
  let t = vec2f(
    not_in_y_range * saturate(fast_div(d.x * conic.x * diff.x + d.x * conic.y * diff.y, d.x * conic.x * d.x)),
    not_in_x_range * saturate(fast_div(d.y * conic.y * diff.x + d.y * conic.z * diff.y, d.y * conic.z * d.y)));
  let max_point = closest + t * d;
  let delta = mean - max_point;
  let max_power = 0.5 * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
  return max_power <= thr;
}
`;

/**
 * Per primitive: depth culling, opacity, 3D covariance, EWA projection with the 0.3 px dilation,
 * screen bounds, and the exact number of touched tiles. Writes the depth key (NO_KEY when the
 * primitive is culled, so culled primitives sort last) and the touched-tile count.
 */
export const PREPROCESS_WGSL = PRELUDE + /* wgsl */ `
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> means: array<f32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> quats: array<f32>;
@group(0) @binding(4) var<storage, read> opacities: array<f32>;
@group(0) @binding(5) var<storage, read_write> proj: array<Proj>;
@group(0) @binding(6) var<storage, read_write> depthKeys: array<u32>;
@group(0) @binding(7) var<storage, read_write> touched: array<u32>;

@compute @workgroup_size(128) fn preprocess(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.n) { return; }
  touched[i] = 0u;
  depthKeys[i] = NO_KEY;
  let m = vec3f(means[3u * i], means[3u * i + 1u], means[3u * i + 2u]);
  let r1 = P.w2c0; let r2 = P.w2c1; let r3 = P.w2c2;
  let depth = r3.x * m.x + r3.y * m.y + r3.z * m.z + r3.w;
  if (depth < P.near || depth > P.far) { return; }
  let opacity = 1.0 / (1.0 + fast_exp(-opacities[i]));
  if (opacity < MIN_ALPHA) { return; }

  let ls = vec3f(scales[3u * i], scales[3u * i + 1u], scales[3u * i + 2u]);
  let vx = fast_exp(2.0 * ls.x); let vy = fast_exp(2.0 * ls.y); let vz = fast_exp(2.0 * ls.z);
  let qr = quats[4u * i]; let qx = quats[4u * i + 1u]; let qy = quats[4u * i + 2u]; let qz = quats[4u * i + 3u];
  let xx = qx * qx; let yy = qy * qy; let zz = qz * qz;
  let xy = qx * qy; let xz = qx * qz; let yz = qy * qz;
  let rx = qr * qx; let ry = qr * qy; let rz = qr * qz;
  let norm_sq = qr * qr + xx + yy + zz;
  if (norm_sq < 1e-8) { return; }
  let inv = 1.0 / norm_sq;
  let R11 = 1.0 - 2.0 * (yy + zz) * inv; let R12 = 2.0 * (xy - rz) * inv; let R13 = 2.0 * (xz + ry) * inv;
  let R21 = 2.0 * (xy + rz) * inv; let R22 = 1.0 - 2.0 * (xx + zz) * inv; let R23 = 2.0 * (yz - rx) * inv;
  let R31 = 2.0 * (xz - ry) * inv; let R32 = 2.0 * (yz + rx) * inv; let R33 = 1.0 - 2.0 * (xx + yy) * inv;
  let S11 = R11 * vx; let S12 = R12 * vy; let S13 = R13 * vz;
  let S21 = R21 * vx; let S22 = R22 * vy; let S23 = R23 * vz;
  let S31 = R31 * vx; let S32 = R32 * vy; let S33 = R33 * vz;
  let c11 = S11 * R11 + S12 * R12 + S13 * R13;
  let c12 = S11 * R21 + S12 * R22 + S13 * R23;
  let c13 = S11 * R31 + S12 * R32 + S13 * R33;
  let c22 = S21 * R21 + S22 * R22 + S23 * R23;
  let c23 = S21 * R31 + S22 * R32 + S23 * R33;
  let c33 = S31 * R31 + S32 * R32 + S33 * R33;

  let x = fast_div(r1.x * m.x + r1.y * m.y + r1.z * m.z + r1.w, depth);
  let y = fast_div(r2.x * m.x + r2.y * m.y + r2.z * m.z + r2.w, depth);
  let clip_left = fast_div(-0.15 * P.widthF - P.cx, P.fx);
  let clip_right = fast_div(1.15 * P.widthF - P.cx, P.fx);
  let clip_top = fast_div(-0.15 * P.heightF - P.cy, P.fy);
  let clip_bottom = fast_div(1.15 * P.heightF - P.cy, P.fy);
  let x_clipped = max(clip_left, min(x, clip_right));
  let y_clipped = max(clip_top, min(y, clip_bottom));
  let j11 = fast_div(P.fx, depth);
  let j13 = -j11 * x_clipped;
  let j22 = fast_div(P.fy, depth);
  let j23 = -j22 * y_clipped;
  let a1 = j11 * r1.x + j13 * r3.x; let a2 = j11 * r1.y + j13 * r3.y; let a3 = j11 * r1.z + j13 * r3.z;
  let b1 = j22 * r2.x + j23 * r3.x; let b2 = j22 * r2.y + j23 * r3.y; let b3 = j22 * r2.z + j23 * r3.z;
  let ac1 = a1 * c11 + a2 * c12 + a3 * c13;
  let ac2 = a1 * c12 + a2 * c22 + a3 * c23;
  let ac3 = a1 * c13 + a2 * c23 + a3 * c33;
  let bc1 = b1 * c11 + b2 * c12 + b3 * c13;
  let bc2 = b1 * c12 + b2 * c22 + b3 * c23;
  let bc3 = b1 * c13 + b2 * c23 + b3 * c33;
  var cxx = ac1 * a1 + ac2 * a2 + ac3 * a3;
  let cxy = ac1 * b1 + ac2 * b2 + ac3 * b3;
  var cyy = bc1 * b1 + bc2 * b2 + bc3 * b3;
  cxx += DILATION;
  cyy += DILATION;
  let det = cxx * cyy - cxy * cxy;
  if (det < MIN_DET) { return; }
  let rdet = 1.0 / det;
  let conic = vec3f(cyy * rdet, -cxy * rdet, cxx * rdet);
  let mean2d = vec2f(x * P.fx + P.cx, y * P.fy + P.cy);

  let power_threshold = fast_log(opacity * MIN_ALPHA_RCP);
  let cutoff = 2.0 * power_threshold;
  let ext_x = max(sqrt(cxx * cutoff) - 0.5, 0.0);
  let ext_y = max(sqrt(cyy * cutoff) - 0.5, 0.0);
  let pw = f32(P.gridW * TILE_W);
  let ph = f32(P.gridH * TILE_H);
  let x0 = u32(min(pw, max(0.0, floor(mean2d.x - ext_x))));
  let x1 = u32(min(pw, max(0.0, ceil(mean2d.x + ext_x))));
  let y0 = u32(min(ph, max(0.0, floor(mean2d.y - ext_y))));
  let y1 = u32(min(ph, max(0.0, ceil(mean2d.y + ext_y))));
  let bounds = vec2u(x0 | (x1 << 16u), y0 | (y1 << 16u));
  let tb = tile_bounds(bounds);
  if ((tb.y - tb.x) * (tb.w - tb.z) == 0u) { return; }

  let shifted = mean2d - 0.5;
  var n = 0u;
  for (var ty = tb.z; ty < tb.w; ty++) {
    for (var tx = tb.x; tx < tb.y; tx++) {
      if (will_contribute(shifted, conic, tx, ty, power_threshold)) { n++; }
    }
  }
  if (n == 0u) { return; }
  proj[i] = Proj(mean2d, bounds, vec4f(conic, opacity));
  touched[i] = n;
  depthKeys[i] = bitcast<u32>(depth);
}
`;

/**
 * Degree-0 SH to colour, 0.5 + C0 * sh0, stored unclamped (the backward needs the sign); the
 * blend kernels clamp it below at zero when they load it, as the CUDA forward does.
 */
export const COLORS_WGSL = PRELUDE + /* wgsl */ `
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> sh0: array<f32>;
@group(0) @binding(2) var<storage, read_write> colors: array<vec4f>;

@compute @workgroup_size(256) fn colors_main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let c = 0.5 + SH_C0 * vec3f(sh0[3u * i], sh0[3u * i + 1u], sh0[3u * i + 2u]);
  colors[i] = vec4f(c, 0.0);
}
`;

const SORT_PRELUDE = /* wgsl */ `
struct SortParams { pass_index: u32, slot: u32, rank_base: u32, implicit_values: u32 };
const THREADS: u32 = ${SORT_THREADS}u;
const ITEMS: u32 = ${SORT_ITEMS}u;
const BLOCK: u32 = ${SORT_BLOCK}u;
const RADIX: u32 = ${SORT_RADIX}u;
fn sort_shift() -> u32 { return select(0u, P.rankBits, S.rank_base == 1u) + 4u * S.pass_index; }
fn sort_count() -> u32 {
  var count = P.n;
  if (S.slot != 0xffffffffu) { count = counters[S.slot]; }
  return count;
}
`;

/** Radix sort, step 1: per-block digit histogram, written digit-major (digit * numBlocks + block). */
export const RADIX_COUNT_WGSL = PRELUDE + SORT_PRELUDE + /* wgsl */ `
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> S: SortParams;
@group(0) @binding(2) var<storage, read> keys: array<u32>;
@group(0) @binding(3) var<storage, read> counters: array<u32>;
@group(0) @binding(4) var<storage, read_write> blockCounts: array<u32>;
var<workgroup> hist: array<atomic<u32>, RADIX>;

@compute @workgroup_size(THREADS) fn radix_count(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let count = sort_count();
  let numBlocks = (count + BLOCK - 1u) / BLOCK;
  let b = wg.x;
  if (b >= numBlocks) { return; }
  if (li < RADIX) { atomicStore(&hist[li], 0u); }
  workgroupBarrier();
  let shift = sort_shift();
  for (var j = 0u; j < ITEMS; j++) {
    let idx = b * BLOCK + j * THREADS + li;
    if (idx < count) { atomicAdd(&hist[(keys[idx] >> shift) & 15u], 1u); }
  }
  workgroupBarrier();
  if (li < RADIX) { blockCounts[li * numBlocks + b] = atomicLoad(&hist[li]); }
}
`;

const SINGLE_SCAN = /* wgsl */ `
var<workgroup> partials: array<u32, 256>;
// Exclusive scan of data[0..len) in one workgroup of 256 threads; returns the total.
fn scan_in_place(li: u32, len: u32) -> u32 {
  let per = (len + 255u) / 256u;
  let start = li * per;
  var sum = 0u;
  for (var i = 0u; i < per; i++) { let j = start + i; if (j < len) { sum += data[j]; } }
  partials[li] = sum;
  workgroupBarrier();
  for (var offset = 1u; offset < 256u; offset = offset << 1u) {
    var v = partials[li];
    if (li >= offset) { v += partials[li - offset]; }
    workgroupBarrier();
    partials[li] = v;
    workgroupBarrier();
  }
  var run = 0u;
  if (li > 0u) { run = partials[li - 1u]; }
  for (var i = 0u; i < per; i++) { let j = start + i; if (j < len) { let v = data[j]; data[j] = run; run += v; } }
  return partials[255u];
}
`;

/** Radix sort, step 2: exclusive scan of the digit-major block histogram. */
export const RADIX_SCAN_WGSL = PRELUDE + SORT_PRELUDE + /* wgsl */ `
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> S: SortParams;
@group(0) @binding(2) var<storage, read> counters: array<u32>;
@group(0) @binding(3) var<storage, read_write> data: array<u32>;
` + SINGLE_SCAN + /* wgsl */ `
@compute @workgroup_size(256) fn radix_scan(@builtin(local_invocation_index) li: u32) {
  let count = sort_count();
  let numBlocks = (count + BLOCK - 1u) / BLOCK;
  _ = scan_in_place(li, RADIX * numBlocks);
}
`;

/**
 * Radix sort, step 3: stable scatter. Each thread owns ITEMS consecutive keys, so thread order
 * followed by item order is input order; per-thread digit counts are scanned digit-major across
 * the block, which gives every key its stable rank among the block's keys of the same digit.
 */
export function radixScatterWgsl(withValues: boolean): string {
  const bindings = withValues ? /* wgsl */ `
@group(0) @binding(2) var<storage, read> keysIn: array<u32>;
@group(0) @binding(3) var<storage, read> valsIn: array<u32>;
@group(0) @binding(4) var<storage, read> counters: array<u32>;
@group(0) @binding(5) var<storage, read> blockOffsets: array<u32>;
@group(0) @binding(6) var<storage, read_write> keysOut: array<u32>;
@group(0) @binding(7) var<storage, read_write> valsOut: array<u32>;
` : /* wgsl */ `
@group(0) @binding(2) var<storage, read> keysIn: array<u32>;
@group(0) @binding(3) var<storage, read> counters: array<u32>;
@group(0) @binding(4) var<storage, read> blockOffsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> keysOut: array<u32>;
`;
  const write = withValues
    ? `keysOut[dst] = k; var v = idx; if (S.implicit_values == 0u) { v = valsIn[idx]; } valsOut[dst] = v;`
    : `keysOut[dst] = k;`;
  return PRELUDE + SORT_PRELUDE + /* wgsl */ `
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> S: SortParams;
${bindings}
var<workgroup> hist: array<u32, ${SORT_RADIX * SORT_THREADS}>;  // [digit][thread]
var<workgroup> sums: array<u32, THREADS>;
var<workgroup> digitStart: array<u32, RADIX>;

@compute @workgroup_size(THREADS) fn radix_scatter(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let count = sort_count();
  let numBlocks = (count + BLOCK - 1u) / BLOCK;
  let b = wg.x;
  if (b >= numBlocks) { return; }
  let shift = sort_shift();
  let base = b * BLOCK + li * ITEMS;
  for (var d = 0u; d < RADIX; d++) { hist[d * THREADS + li] = 0u; }
  for (var j = 0u; j < ITEMS; j++) {
    let idx = base + j;
    if (idx < count) { let d = (keysIn[idx] >> shift) & 15u; hist[d * THREADS + li] += 1u; }
  }
  workgroupBarrier();
  // exclusive scan of hist flattened digit-major; thread li owns entries [li * RADIX, li * RADIX + RADIX)
  var s = 0u;
  for (var j = 0u; j < RADIX; j++) { s += hist[li * RADIX + j]; }
  sums[li] = s;
  workgroupBarrier();
  for (var offset = 1u; offset < THREADS; offset = offset << 1u) {
    var v = sums[li];
    if (li >= offset) { v += sums[li - offset]; }
    workgroupBarrier();
    sums[li] = v;
    workgroupBarrier();
  }
  var run = 0u;
  if (li > 0u) { run = sums[li - 1u]; }
  for (var j = 0u; j < RADIX; j++) { let v = hist[li * RADIX + j]; hist[li * RADIX + j] = run; run += v; }
  workgroupBarrier();
  if (li < RADIX) { digitStart[li] = hist[li * THREADS]; }
  workgroupBarrier();
  for (var j = 0u; j < ITEMS; j++) {
    let idx = base + j;
    if (idx < count) {
      let k = keysIn[idx];
      let d = (k >> shift) & 15u;
      let pos = hist[d * THREADS + li];
      hist[d * THREADS + li] = pos + 1u;
      let dst = blockOffsets[d * numBlocks + b] + (pos - digitStart[d]);
      ${write}
    }
  }
}
`;
}

/** Instance offsets, step 1: per-block sums of the touched-tile counts in depth order. */
export const OFFSETS_REDUCE_WGSL = PRELUDE + /* wgsl */ `
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> sortedIdx: array<u32>;
@group(0) @binding(2) var<storage, read> touched: array<u32>;
@group(0) @binding(3) var<storage, read_write> blockSums: array<u32>;
var<workgroup> red: array<u32, 256>;

@compute @workgroup_size(256) fn offsets_reduce(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let r = wg.x * 256u + li;
  var v = 0u;
  if (r < P.n) { v = touched[sortedIdx[r]]; }
  red[li] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (li < s) { red[li] += red[li + s]; }
    workgroupBarrier();
  }
  if (li == 0u) { blockSums[wg.x] = red[0]; }
}
`;

/**
 * Instance offsets, step 2: exclusive scan of the block sums; writes the instance total, the
 * count clamped to the capacity (what the tile sort and the range extraction process), and the
 * overflow flag.
 */
export const OFFSETS_SCAN_WGSL = PRELUDE + /* wgsl */ `
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> counters: array<u32>;
` + SINGLE_SCAN + /* wgsl */ `
@compute @workgroup_size(256) fn offsets_scan(@builtin(local_invocation_index) li: u32) {
  let total = scan_in_place(li, (P.n + 255u) / 256u);
  if (li == 0u) {
    let clamped = min(total, P.capacity);
    counters[${COUNTER.instances}] = total;
    counters[${COUNTER.sortCount}] = clamped;
    counters[${COUNTER.overflow}] = select(0u, 1u, total > P.capacity);
    counters[${COUNTER.bucketOverflow}] = 0u;
  }
}
`;

/**
 * Instance creation in depth order: primitive of depth rank r writes `tile << rankBits | r` for
 * every tile it contributes to, from its exclusive offset. Writes are capped at the primitive's
 * touched count and missing ones are filled with a sentinel tile (nTiles), so a counting
 * mismatch can never corrupt a neighbour's slots.
 */
export const CREATE_INSTANCES_WGSL = PRELUDE + /* wgsl */ `
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> sortedIdx: array<u32>;
@group(0) @binding(2) var<storage, read> touched: array<u32>;
@group(0) @binding(3) var<storage, read> blockSums: array<u32>;
@group(0) @binding(4) var<storage, read> proj: array<Proj>;
@group(0) @binding(5) var<storage, read_write> instKeys: array<u32>;
var<workgroup> wsum: array<u32, 256>;

@compute @workgroup_size(256) fn create_instances(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let r = wg.x * 256u + li;
  var cnt = 0u;
  var p = 0u;
  if (r < P.n) { p = sortedIdx[r]; cnt = touched[p]; }
  wsum[li] = cnt;
  workgroupBarrier();
  for (var offset = 1u; offset < 256u; offset = offset << 1u) {
    var v = wsum[li];
    if (li >= offset) { v += wsum[li - offset]; }
    workgroupBarrier();
    wsum[li] = v;
    workgroupBarrier();
  }
  if (cnt == 0u) { return; }
  var off = blockSums[wg.x] + wsum[li] - cnt;
  let end = off + cnt;
  let pr = proj[p];
  let shifted = pr.mean - 0.5;
  let conic = pr.conicOpacity.xyz;
  let thr = fast_log(pr.conicOpacity.w * MIN_ALPHA_RCP);
  let tb = tile_bounds(pr.bounds);
  for (var ty = tb.z; ty < tb.w; ty++) {
    for (var tx = tb.x; tx < tb.y; tx++) {
      if (off < end && will_contribute(shifted, conic, tx, ty, thr)) {
        if (off < P.capacity) { instKeys[off] = ((ty * P.gridW + tx) << P.rankBits) | r; }
        off++;
      }
    }
  }
  for (; off < end; off++) {
    if (off < P.capacity) { instKeys[off] = (P.nTiles << P.rankBits) | r; }
  }
}
`;

/** Per-tile instance ranges [start, end) and the primitive index of every sorted instance. */
export const RANGES_WGSL = PRELUDE + /* wgsl */ `
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> counters: array<u32>;
@group(0) @binding(2) var<storage, read> instKeys: array<u32>;
@group(0) @binding(3) var<storage, read> sortedIdx: array<u32>;
@group(0) @binding(4) var<storage, read_write> ranges: array<u32>;
@group(0) @binding(5) var<storage, read_write> instPrim: array<u32>;

@compute @workgroup_size(256) fn tile_ranges(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let m = counters[${COUNTER.sortCount}];
  if (i >= m) { return; }
  let key = instKeys[i];
  let tile = key >> P.rankBits;
  instPrim[i] = sortedIdx[key & ((1u << P.rankBits) - 1u)];
  if (tile >= P.nTiles) { return; }
  if (i == 0u || (instKeys[i - 1u] >> P.rankBits) != tile) { ranges[2u * tile] = i; }
  if (i + 1u == m || (instKeys[i + 1u] >> P.rankBits) != tile) { ranges[2u * tile + 1u] = i + 1u; }
}
`;

/**
 * Front-to-back blending, one workgroup per 16x12 tile. Each pixel walks the tile's instances in
 * depth order, skips primitives whose screen bounds miss its 8x4 subtile (the CUDA warp test),
 * skips alpha < 1/255, blends, and stops once transmittance falls below 1e-4 (after blending
 * the primitive that crossed it). No alpha cap. The background is added with the final
 * transmittance.
 *
 * `tieMask` builds a diagnostic variant instead of the production kernel: it writes the image
 * and, per pixel, 1 where two contributing primitives with bit-identical depth from different
 * 32-index groups meet (the pixels whose CUDA value depends on thread scheduling), else 0. It
 * keeps checking primitives of the last blended depth after the early stop.
 */
export function blendWgsl(tieMask: boolean): string {
  const bindings = tieMask ? /* wgsl */ `
@group(0) @binding(5) var<storage, read> instKeys: array<u32>;
@group(0) @binding(6) var<storage, read> depthKeysSorted: array<u32>;
@group(0) @binding(7) var<storage, read_write> image: array<f32>;
@group(0) @binding(8) var<storage, read_write> tieOut: array<u32>;
var<workgroup> sKey: array<u32, ${BLEND_THREADS}>;
var<workgroup> sGroup: array<u32, ${BLEND_THREADS}>;
` : /* wgsl */ `
@group(0) @binding(5) var<storage, read_write> image: array<f32>;
@group(0) @binding(6) var<storage, read_write> finalT: array<f32>;
@group(0) @binding(7) var<storage, read_write> nProcessed: array<u32>;
`;
  const load = tieMask
    ? `sKey[li] = depthKeysSorted[instKeys[gi] & ((1u << P.rankBits) - 1u)]; sGroup[li] = p >> 5u;`
    : ``;
  // Early exit of the whole workgroup once every pixel is done (production kernel only).
  const exitCheck = tieMask ? `` : /* wgsl */ `
    if (li == 0u) { sDoneSnap = atomicLoad(&sDone); }
    let nDone = workgroupUniformLoad(&sDoneSnap);
    if (nDone == ${BLEND_THREADS}u) { break; }`;
  const preEval = tieMask ? /* wgsl */ `
      if (done && sKey[j] != lastKey) { finished = true; break; }` : ``;
  const evalTail = tieMask ? /* wgsl */ `
      let key = sKey[j]; let grp = sGroup[j];
      if (key == lastKey && grp != lastGroup) { tie = 1u; }
      if (done) { continue; }
      lastKey = key; lastGroup = grp;
      color += T * alpha * sColor[j].xyz;
      T *= 1.0 - alpha;
      if (T < T_MIN) { done = true; }` : /* wgsl */ `
      color += T * alpha * sColor[j].xyz;
      T *= 1.0 - alpha;
      nProc = start + j + 1u;
      if (T < T_MIN) { done = true; atomicAdd(&sDone, 1u); }`;
  const store = tieMask
    ? `tieOut[pix] = tie;`
    : `finalT[pix] = T; nProcessed[pix] = nProc;`;
  const stopFlag = tieMask ? `finished` : `done`;
  return PRELUDE + /* wgsl */ `
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> ranges: array<u32>;
@group(0) @binding(2) var<storage, read> instPrim: array<u32>;
@group(0) @binding(3) var<storage, read> proj: array<Proj>;
@group(0) @binding(4) var<storage, read> colors: array<vec4f>;
${bindings}
var<workgroup> sMean: array<vec2f, ${BLEND_THREADS}>;
var<workgroup> sBounds: array<vec2u, ${BLEND_THREADS}>;
var<workgroup> sConic: array<vec4f, ${BLEND_THREADS}>;
var<workgroup> sColor: array<vec4f, ${BLEND_THREADS}>;
var<workgroup> sDone: atomic<u32>;
var<workgroup> sDoneSnap: u32;

@compute @workgroup_size(${TILE_W}, ${TILE_H}) fn blend(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(local_invocation_index) li: u32) {
  let tile = wg.y * P.gridW + wg.x;
  let px = wg.x * TILE_W + lid.x;
  let py = wg.y * TILE_H + lid.y;
  let inside = px < P.width && py < P.height;
  let pixel = vec2f(f32(px), f32(py)) + 0.5;
  let sx0 = wg.x * TILE_W + (lid.x / 8u) * 8u;
  let sx1 = sx0 + 8u;
  let sy0 = wg.y * TILE_H + (lid.y / 4u) * 4u;
  let sy1 = sy0 + 4u;
  let rstart = ranges[2u * tile];
  let total = ranges[2u * tile + 1u] - rstart;
  if (li == 0u) { atomicStore(&sDone, 0u); }
  workgroupBarrier();
  var color = vec3f(0.0);
  var T: f32 = 1.0;
  var nProc = 0u;
  var done = !inside;
  var finished = !inside;
  var lastKey = 0xffffffffu;
  var lastGroup = 0xffffffffu;
  var tie = 0u;
  if (done) { atomicAdd(&sDone, 1u); }
  for (var start = 0u; start < total; start += ${BLEND_THREADS}u) {
    workgroupBarrier();${exitCheck}
    let fetch = start + li;
    if (fetch < total) {
      let gi = rstart + fetch;
      let p = instPrim[gi];
      let pr = proj[p];
      sMean[li] = pr.mean; sBounds[li] = pr.bounds; sConic[li] = pr.conicOpacity; sColor[li] = max(colors[p], vec4f(0.0));
      ${load}
    }
    workgroupBarrier();
    let batch = min(${BLEND_THREADS}u, total - start);
    for (var j = 0u; j < batch; j++) {
      if (${stopFlag}) { break; }${preEval}
      let b = sBounds[j];
      if (!((b.x & 0xffffu) < sx1 && sx0 < (b.x >> 16u) && (b.y & 0xffffu) < sy1 && sy0 < (b.y >> 16u))) { continue; }
      let co = sConic[j];
      let d = sMean[j] - pixel;
      let e = -0.5 * (co.x * d.x * d.x + co.z * d.y * d.y) - co.y * d.x * d.y;
      let alpha = co.w * fast_exp(min(e, 0.0));
      if (alpha < MIN_ALPHA) { continue; }${evalTail}
    }
  }
  if (inside) {
    color += T * P.bg.xyz;
    let pix = py * P.width + px;
    let hw = P.width * P.height;
    image[pix] = color.x;
    image[hw + pix] = color.y;
    image[2u * hw + pix] = color.z;
    ${store}
  }
}
`;
}

export interface RasterParamsInput {
  w2c: ArrayLike<number>;
  position: ArrayLike<number>;
  background: ArrayLike<number>;
  n: number;
  width: number;
  height: number;
  gridW: number;
  gridH: number;
  rankBits: number;
  capacity: number;
  focalX: number;
  focalY: number;
  centerX: number;
  centerY: number;
  near: number;
  far: number;
}

/** Pack the Params uniform (PARAMS_BYTES bytes). */
export function packRasterParams(p: RasterParamsInput): ArrayBuffer {
  const buf = new ArrayBuffer(PARAMS_BYTES);
  const f = new Float32Array(buf);
  const u = new Uint32Array(buf);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) f[4 * r + c] = p.w2c[4 * r + c];
  f[12] = p.position[0]; f[13] = p.position[1]; f[14] = p.position[2]; f[15] = 0;
  f[16] = p.background[0]; f[17] = p.background[1]; f[18] = p.background[2]; f[19] = 0;
  u[20] = p.n; u[21] = p.width; u[22] = p.height; u[23] = p.gridW;
  u[24] = p.gridH; u[25] = p.gridW * p.gridH; u[26] = p.rankBits; u[27] = p.capacity;
  f[28] = p.focalX; f[29] = p.focalY; f[30] = p.centerX; f[31] = p.centerY;
  f[32] = p.near; f[33] = p.far; f[34] = p.width; f[35] = p.height;
  return buf;
}

/** Shared WGSL prelude (Params, Proj, constants, fast-math helpers, the tile test) and the one-workgroup scan. */
export { PRELUDE as RASTER_PRELUDE, SINGLE_SCAN };
