/**
 * WGSL port of the backward pass of the Faster-GS rasterizer, for degree-0 SH and
 * proper_antialiasing = false.
 *
 * Faster-GS splits every tile's depth-ordered instance list into buckets of 32. During the forward
 * blend it stores, per bucket and pixel, the accumulated colour and the transmittance at the
 * bucket's start; the backward then runs one warp per bucket, each lane owning one primitive and
 * the pixels passed from lane to lane with shfl_up, so every lane sees the transmittance and the
 * colour still to come exactly in front of its own primitive. Gradients of the 2D quantities go
 * to per-primitive sums with atomics, and a per-primitive pass applies the chain rule to the 3D
 * parameters.
 *
 * Here the snapshots come from a re-blend at the start of the backward (the forward kernel is
 * unchanged), the warp pipeline is a 32-thread workgroup that passes the pixel state through a
 * double-buffered shared ring with one barrier per step, and the float atomics are
 * compare-and-swap loops on u32 bit patterns. Fast-math conventions as in shaders.ts.
 */
import { BLEND_THREADS, COUNTER, RASTER_PRELUDE, SINGLE_SCAN, TILE_H, TILE_W } from "./shaders.js";

/** Workgroups per row of the 2D bucket dispatch. */
export const BUCKET_DISPATCH_X = 32768;
/** Per-primitive accumulator layout: mean2d (2), conic (3), opacity (1, already times the sigmoid derivative), colour (3). */
export const ACC_STRIDE = 9;

const BACKWARD_PARAMS = /* wgsl */ `
// bucketCap: capacity of the bucket buffers; bucketInfo[bucketCap].x holds the bucket count.
struct BackwardParams { bucketCap: u32, accumulate: u32, pad0: u32, pad1: u32 };
const BUCKET: u32 = 32u;
const PIXELS: u32 = ${BLEND_THREADS}u;
const ONE_MINUS_ALPHA_EPS: f32 = 1e-6;
`;

/**
 * Buckets per tile, ceil(instances / 32), scanned into the first bucket of every tile; the total
 * goes to bucketInfo[bucketCap].x and its overflow to the counters.
 */
export const BUCKET_SCAN_WGSL = RASTER_PRELUDE + BACKWARD_PARAMS + /* wgsl */ `
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> BP: BackwardParams;
@group(0) @binding(2) var<storage, read> ranges: array<u32>;
@group(0) @binding(3) var<storage, read_write> data: array<u32>;
@group(0) @binding(4) var<storage, read_write> bucketInfo: array<vec4u>;
@group(0) @binding(5) var<storage, read_write> counters: array<u32>;
` + SINGLE_SCAN + /* wgsl */ `
@compute @workgroup_size(256) fn bucket_scan(@builtin(local_invocation_index) li: u32) {
  let len = P.nTiles;
  let per = (len + 255u) / 256u;
  for (var i = 0u; i < per; i++) {
    let t = li * per + i;
    if (t < len) { data[t] = (ranges[2u * t + 1u] - ranges[2u * t] + BUCKET - 1u) / BUCKET; }
  }
  let total = scan_in_place(li, len);
  if (li == 0u) {
    bucketInfo[BP.bucketCap] = vec4u(total, 0u, 0u, 0u);
    counters[${COUNTER.bucketOverflow}] = select(0u, 1u, total > BP.bucketCap);
  }
}
`;

/** Per bucket: (tile, first bucket of the tile, first instance of the tile, instances in the tile). */
export const BUCKET_INFO_WGSL = RASTER_PRELUDE + BACKWARD_PARAMS + /* wgsl */ `
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> BP: BackwardParams;
@group(0) @binding(2) var<storage, read> ranges: array<u32>;
@group(0) @binding(3) var<storage, read> tileBuckets: array<u32>;
@group(0) @binding(4) var<storage, read_write> bucketInfo: array<vec4u>;

@compute @workgroup_size(256) fn bucket_info(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  if (t >= P.nTiles) { return; }
  let first = tileBuckets[t];
  let r0 = ranges[2u * t];
  let count = ranges[2u * t + 1u] - r0;
  let nb = (count + BUCKET - 1u) / BUCKET;
  for (var q = 0u; q < nb; q++) {
    let b = first + q;
    if (b < BP.bucketCap) { bucketInfo[b] = vec4u(t, first, r0, count); }
  }
}
`;

/**
 * The forward blend again, writing what the backward needs: at the start of every bucket a pixel
 * has not finished, (accumulated colour, transmittance); at the end, each of those snapshots is
 * turned into (image - T_final * bg - accumulated colour, transmittance), the colour still to
 * come as the CUDA backward computes it, and the pixel's (T_final bits, 1 + last blended index).
 */
export const SNAPSHOT_WGSL = RASTER_PRELUDE + BACKWARD_PARAMS + /* wgsl */ `
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> BP: BackwardParams;
@group(0) @binding(2) var<storage, read> ranges: array<u32>;
@group(0) @binding(3) var<storage, read> instPrim: array<u32>;
@group(0) @binding(4) var<storage, read> proj: array<Proj>;
@group(0) @binding(5) var<storage, read> colors: array<vec4f>;
@group(0) @binding(6) var<storage, read> tileBuckets: array<u32>;
@group(0) @binding(7) var<storage, read_write> snapshots: array<vec4f>;
@group(0) @binding(8) var<storage, read_write> pixState: array<vec2u>;
var<workgroup> sMean: array<vec2f, PIXELS>;
var<workgroup> sBounds: array<vec2u, PIXELS>;
var<workgroup> sConic: array<vec4f, PIXELS>;
var<workgroup> sColor: array<vec4f, PIXELS>;
var<workgroup> sDone: atomic<u32>;
var<workgroup> sDoneSnap: u32;

@compute @workgroup_size(${TILE_W}, ${TILE_H}) fn snapshot(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(local_invocation_index) li: u32) {
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
  let firstBucket = tileBuckets[tile];
  if (li == 0u) { atomicStore(&sDone, 0u); }
  workgroupBarrier();
  var color = vec3f(0.0);
  var T: f32 = 1.0;
  var nProc = 0u;
  var nSnap = 0u;
  var done = !inside;
  if (done) { atomicAdd(&sDone, 1u); }
  for (var start = 0u; start < total; start += PIXELS) {
    workgroupBarrier();
    if (li == 0u) { sDoneSnap = atomicLoad(&sDone); }
    let nDone = workgroupUniformLoad(&sDoneSnap);
    if (nDone == PIXELS) { break; }
    let fetch = start + li;
    if (fetch < total) {
      let p = instPrim[rstart + fetch];
      let pr = proj[p];
      sMean[li] = pr.mean; sBounds[li] = pr.bounds; sConic[li] = pr.conicOpacity; sColor[li] = max(colors[p], vec4f(0.0));
    }
    workgroupBarrier();
    let batch = min(PIXELS, total - start);
    for (var j = 0u; j < batch; j++) {
      if (done) { break; }
      if (((start + j) % BUCKET) == 0u) {
        let b = firstBucket + (start + j) / BUCKET;
        if (b < BP.bucketCap) { snapshots[b * PIXELS + li] = vec4f(color, T); }
        nSnap++;
      }
      let bd = sBounds[j];
      if (!((bd.x & 0xffffu) < sx1 && sx0 < (bd.x >> 16u) && (bd.y & 0xffffu) < sy1 && sy0 < (bd.y >> 16u))) { continue; }
      let co = sConic[j];
      let d = sMean[j] - pixel;
      let e = -0.5 * (co.x * d.x * d.x + co.z * d.y * d.y) - co.y * d.x * d.y;
      let alpha = co.w * fast_exp(min(e, 0.0));
      if (alpha < MIN_ALPHA) { continue; }
      color += T * alpha * sColor[j].xyz;
      T *= 1.0 - alpha;
      nProc = start + j + 1u;
      if (T < T_MIN) { done = true; atomicAdd(&sDone, 1u); }
    }
  }
  if (inside) {
    let bg = P.bg.xyz;
    let image = color + T * bg;
    for (var q = 0u; q < nSnap; q++) {
      let b = firstBucket + q;
      if (b < BP.bucketCap) {
        let s = snapshots[b * PIXELS + li];
        snapshots[b * PIXELS + li] = vec4f(image - T * bg - s.xyz, s.w);
      }
    }
    pixState[py * P.width + px] = vec2u(bitcast<u32>(T), nProc);
  }
}
`;

/**
 * One 32-thread workgroup per bucket, one primitive per thread. Step i hands pixel i to thread 0
 * and every thread l works on pixel i - l with the state thread l - 1 left for it at step i - 1;
 * the state is (colour still to come, transmittance, dL/dpixel, -T_final dL/dpixel . bg, last
 * contributor). Per-thread sums go to the per-primitive accumulators with float CAS adds.
 */
export const BLEND_BACKWARD_WGSL = RASTER_PRELUDE + BACKWARD_PARAMS + /* wgsl */ `
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> BP: BackwardParams;
@group(0) @binding(2) var<storage, read> instPrim: array<u32>;
@group(0) @binding(3) var<storage, read> proj: array<Proj>;
@group(0) @binding(4) var<storage, read> colors: array<vec4f>;
@group(0) @binding(5) var<storage, read> bucketInfo: array<vec4u>;
@group(0) @binding(6) var<storage, read> snapshots: array<vec4f>;
@group(0) @binding(7) var<storage, read> pixState: array<vec2u>;
@group(0) @binding(8) var<storage, read> gradImage: array<f32>;
@group(0) @binding(9) var<storage, read_write> acc: array<atomic<u32>>;
var<workgroup> sMax: atomic<u32>;
var<workgroup> sMaxSnap: u32;
var<workgroup> pAfter: array<vec4f, 32>;
var<workgroup> pGrad: array<vec4f, 32>;
var<workgroup> pLast: array<u32, 32>;
var<workgroup> ringA: array<vec4f, 64>;
var<workgroup> ringG: array<vec4f, 64>;
var<workgroup> ringL: array<u32, 64>;

fn add_f32(i: u32, v: f32) {
  if (v == 0.0) { return; }
  var old = atomicLoad(&acc[i]);
  loop {
    let r = atomicCompareExchangeWeak(&acc[i], old, bitcast<u32>(bitcast<f32>(old) + v));
    if (r.exchanged) { break; }
    old = r.old_value;
  }
}

@compute @workgroup_size(32) fn blend_backward(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let b = wg.x + wg.y * ${BUCKET_DISPATCH_X}u;
  let nBuckets = min(bucketInfo[BP.bucketCap].x, BP.bucketCap);
  if (b >= nBuckets) { return; }
  let info = bucketInfo[b];
  let tile = info.x;
  let tileBucket = b - info.y;
  let ox = (tile % P.gridW) * TILE_W;
  let oy = (tile / P.gridW) * TILE_H;
  let hw = P.width * P.height;

  // skip buckets past every pixel's last contributor
  if (li == 0u) { atomicStore(&sMax, 0u); }
  workgroupBarrier();
  for (var q = 0u; q < PIXELS / 32u; q++) {
    let local = q * 32u + li;
    let px = ox + local % TILE_W; let py = oy + local / TILE_W;
    if (px < P.width && py < P.height) { atomicMax(&sMax, pixState[py * P.width + px].y); }
  }
  workgroupBarrier();
  if (li == 0u) { sMaxSnap = atomicLoad(&sMax); }
  let maxProc = workgroupUniformLoad(&sMaxSnap);
  if (tileBucket * BUCKET >= maxProc) { return; }

  let tpi = tileBucket * BUCKET + li;
  let valid = tpi < info.w;
  var p = 0u;
  var mean = vec2f(0.0);
  var conic = vec3f(0.0);
  var opacity = 0.0;
  var color = vec3f(0.0);
  var factor = vec3f(0.0);
  if (valid) {
    p = instPrim[info.z + tpi];
    let pr = proj[p];
    mean = pr.mean;
    conic = pr.conicOpacity.xyz;
    opacity = pr.conicOpacity.w;
    let raw = colors[p].xyz;
    color = max(raw, vec3f(0.0));
    factor = select(vec3f(0.0), vec3f(1.0), raw >= vec3f(0.0));
  }
  var dMean = vec2f(0.0);
  var dConic = vec3f(0.0);
  var dOpacity = 0.0;
  var dColor = vec3f(0.0);

  var after = vec3f(0.0);
  var T = 0.0;
  var grad = vec3f(0.0);
  var gac = 0.0;
  var last = 0u;
  let bg = P.bg.xyz;
  for (var i = 0u; i < PIXELS + 31u; i++) {
    if ((i % 32u) == 0u) {
      let local = i + li;
      if (local < PIXELS) {
        let px = ox + local % TILE_W; let py = oy + local / TILE_W;
        var a = vec4f(0.0); var g = vec4f(0.0); var l = 0u;
        if (px < P.width && py < P.height) {
          let pix = py * P.width + px;
          let st = pixState[pix];
          a = snapshots[b * PIXELS + local];
          let gp = vec3f(gradImage[pix], gradImage[hw + pix], gradImage[2u * hw + pix]);
          g = vec4f(gp, bitcast<f32>(st.x) * -dot(gp, bg));
          l = st.y;
        }
        pAfter[li] = a; pGrad[li] = g; pLast[li] = l;
      }
      workgroupBarrier();
    }
    // take the state of the previous thread (thread 0 keeps its own), then thread 0 loads pixel i
    if (i > 0u && li > 0u) {
      let r = ((i + 1u) % 2u) * 32u + li - 1u;
      let a = ringA[r]; let g = ringG[r];
      after = a.xyz; T = a.w; grad = g.xyz; gac = g.w; last = ringL[r];
    }
    if (li == 0u && i < PIXELS) {
      let c = i % 32u;
      let a = pAfter[c]; let g = pGrad[c];
      after = a.xyz; T = a.w; grad = g.xyz; gac = g.w; last = pLast[c];
    }
    let idx = i32(i) - i32(li);
    if (valid && idx >= 0 && idx < i32(PIXELS) && tpi < last) {
      let u = u32(idx);
      let px = ox + u % TILE_W; let py = oy + u / TILE_W;
      if (px < P.width && py < P.height) {
        let pixel = vec2f(f32(px), f32(py)) + 0.5;
        let delta = mean - pixel;
        let e = -0.5 * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) - conic.y * delta.x * delta.y;
        let gaussian = fast_exp(min(e, 0.0));
        let alpha = opacity * gaussian;
        if (alpha >= MIN_ALPHA) {
          let weight = T * alpha;
          dColor += weight * grad * factor;
          after -= weight * color;
          let oma = 1.0 - alpha;
          let oma_rcp = 1.0 / max(oma, ONE_MINUS_ALPHA_EPS);
          let dAlpha = dot(T * color - after * oma_rcp, grad) + gac * oma_rcp;
          dOpacity += gaussian * dAlpha;
          let helper = -alpha * dAlpha;
          dConic += 0.5 * helper * vec3f(delta.x * delta.x, delta.x * delta.y, delta.y * delta.y);
          dMean += helper * vec2f(conic.x * delta.x + conic.y * delta.y, conic.y * delta.x + conic.z * delta.y);
          T *= oma;
        }
      }
    }
    let w = (i % 2u) * 32u + li;
    ringA[w] = vec4f(after, T); ringG[w] = vec4f(grad, gac); ringL[w] = last;
    workgroupBarrier();
  }
  if (valid) {
    let o = p * ${ACC_STRIDE}u;
    add_f32(o, dMean.x); add_f32(o + 1u, dMean.y);
    add_f32(o + 2u, dConic.x); add_f32(o + 3u, dConic.y); add_f32(o + 4u, dConic.z);
    add_f32(o + 5u, opacity * (1.0 - opacity) * dOpacity);
    add_f32(o + 6u, dColor.x); add_f32(o + 7u, dColor.y); add_f32(o + 8u, dColor.z);
  }
}
`;

/**
 * Chain rule from the accumulated 2D gradients (mean2d, conic) to the means, log scales and raw
 * quaternions; writes (or adds, with BP.accumulate) the three (N, 3|3|4) gradient tensors.
 */
export const PREPROCESS_BACKWARD_WGSL = RASTER_PRELUDE + BACKWARD_PARAMS + /* wgsl */ `
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> BP: BackwardParams;
@group(0) @binding(2) var<storage, read> means: array<f32>;
@group(0) @binding(3) var<storage, read> scales: array<f32>;
@group(0) @binding(4) var<storage, read> quats: array<f32>;
@group(0) @binding(5) var<storage, read> touched: array<u32>;
@group(0) @binding(6) var<storage, read> acc: array<u32>;
@group(0) @binding(7) var<storage, read_write> gMeans: array<f32>;
@group(0) @binding(8) var<storage, read_write> gScales: array<f32>;
@group(0) @binding(9) var<storage, read_write> gQuats: array<f32>;

fn put_means(i: u32, v: vec3f) {
  var s = v;
  if (BP.accumulate == 1u) { s += vec3f(gMeans[3u * i], gMeans[3u * i + 1u], gMeans[3u * i + 2u]); }
  gMeans[3u * i] = s.x; gMeans[3u * i + 1u] = s.y; gMeans[3u * i + 2u] = s.z;
}
fn put_scales(i: u32, v: vec3f) {
  var s = v;
  if (BP.accumulate == 1u) { s += vec3f(gScales[3u * i], gScales[3u * i + 1u], gScales[3u * i + 2u]); }
  gScales[3u * i] = s.x; gScales[3u * i + 1u] = s.y; gScales[3u * i + 2u] = s.z;
}

@compute @workgroup_size(128) fn preprocess_backward(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.n) { return; }
  if (touched[i] == 0u) {
    if (BP.accumulate == 0u) {
      put_means(i, vec3f(0.0)); put_scales(i, vec3f(0.0));
      gQuats[4u * i] = 0.0; gQuats[4u * i + 1u] = 0.0; gQuats[4u * i + 2u] = 0.0; gQuats[4u * i + 3u] = 0.0;
    }
    return;
  }
  let o = i * ${ACC_STRIDE}u;
  let dm2 = vec2f(bitcast<f32>(acc[o]), bitcast<f32>(acc[o + 1u]));
  let dco = vec3f(bitcast<f32>(acc[o + 2u]), bitcast<f32>(acc[o + 3u]), bitcast<f32>(acc[o + 4u]));

  let m = vec3f(means[3u * i], means[3u * i + 1u], means[3u * i + 2u]);
  let r1 = P.w2c0; let r2 = P.w2c1; let r3 = P.w2c2;
  let depth = r3.x * m.x + r3.y * m.y + r3.z * m.z + r3.w;
  let x = fast_div(r1.x * m.x + r1.y * m.y + r1.z * m.z + r1.w, depth);
  let y = fast_div(r2.x * m.x + r2.y * m.y + r2.z * m.z + r2.w, depth);

  let ls = vec3f(scales[3u * i], scales[3u * i + 1u], scales[3u * i + 2u]);
  let variance = vec3f(fast_exp(2.0 * ls.x), fast_exp(2.0 * ls.y), fast_exp(2.0 * ls.z));
  let qr = quats[4u * i]; let qx = quats[4u * i + 1u]; let qy = quats[4u * i + 2u]; let qz = quats[4u * i + 3u];
  let xx = qx * qx; let yy = qy * qy; let zz = qz * qz;
  let xy = qx * qy; let xz = qx * qz; let yz = qy * qz;
  let rx = qr * qx; let ry = qr * qy; let rz = qr * qz;
  let norm_sq = qr * qr + xx + yy + zz;
  let inv = 1.0 / norm_sq;
  let R11 = 1.0 - 2.0 * (yy + zz) * inv; let R12 = 2.0 * (xy - rz) * inv; let R13 = 2.0 * (xz + ry) * inv;
  let R21 = 2.0 * (xy + rz) * inv; let R22 = 1.0 - 2.0 * (xx + zz) * inv; let R23 = 2.0 * (yz - rx) * inv;
  let R31 = 2.0 * (xz - ry) * inv; let R32 = 2.0 * (yz + rx) * inv; let R33 = 1.0 - 2.0 * (xx + yy) * inv;
  let S11 = R11 * variance.x; let S12 = R12 * variance.y; let S13 = R13 * variance.z;
  let S21 = R21 * variance.x; let S22 = R22 * variance.y; let S23 = R23 * variance.z;
  let S31 = R31 * variance.x; let S32 = R32 * variance.y; let S33 = R33 * variance.z;
  let c11 = S11 * R11 + S12 * R12 + S13 * R13;
  let c12 = S11 * R21 + S12 * R22 + S13 * R23;
  let c13 = S11 * R31 + S12 * R32 + S13 * R33;
  let c22 = S21 * R21 + S22 * R22 + S23 * R23;
  let c23 = S21 * R31 + S22 * R32 + S23 * R33;
  let c33 = S31 * R31 + S32 * R32 + S33 * R33;

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

  // 2D covariance gradient
  let a = ac1 * a1 + ac2 * a2 + ac3 * a3 + DILATION;
  let b = ac1 * b1 + ac2 * b2 + ac3 * b3;
  let c = bc1 * b1 + bc2 * b2 + bc3 * b3 + DILATION;
  let aa = a * a; let bb = b * b; let cc = c * c;
  let ac = a * c; let ab = a * b; let bc = b * c;
  let det = ac - bb;
  let det_rcp_sq = 1.0 / (det * det);
  let dc = vec3f(
    2.0 * bc * dco.y - cc * dco.x - bb * dco.z,
    bc * dco.x - (ac + bb) * dco.y + ab * dco.z,
    2.0 * ab * dco.y - bb * dco.x - aa * dco.z) * det_rcp_sq;

  // 3D covariance gradient
  let d11 = a1 * a1 * dc.x + 2.0 * a1 * b1 * dc.y + b1 * b1 * dc.z;
  let d12 = a1 * a2 * dc.x + (a1 * b2 + a2 * b1) * dc.y + b1 * b2 * dc.z;
  let d13 = a1 * a3 * dc.x + (a1 * b3 + a3 * b1) * dc.y + b1 * b3 * dc.z;
  let d22 = a2 * a2 * dc.x + 2.0 * a2 * b2 * dc.y + b2 * b2 * dc.z;
  let d23 = a2 * a3 * dc.x + (a2 * b3 + a3 * b2) * dc.y + b2 * b3 * dc.z;
  let d33 = a3 * a3 * dc.x + 2.0 * a3 * b3 * dc.y + b3 * b3 * dc.z;

  // gradient of J W and of the non-zero entries of J
  let dw1 = 2.0 * vec3f(ac1 * dc.x + bc1 * dc.y, ac2 * dc.x + bc2 * dc.y, ac3 * dc.x + bc3 * dc.y);
  let dw2 = 2.0 * vec3f(ac1 * dc.y + bc1 * dc.z, ac2 * dc.y + bc2 * dc.z, ac3 * dc.y + bc3 * dc.z);
  let dj11 = r1.x * dw1.x + r1.y * dw1.y + r1.z * dw1.z;
  let dj22 = r2.x * dw2.x + r2.y * dw2.y + r2.z * dw2.z;
  let dj13 = r3.x * dw1.x + r3.y * dw1.y + r3.z * dw1.z;
  let dj23 = r3.x * dw2.x + r3.y * dw2.y + r3.z * dw2.z;

  // camera-space mean gradient from mean2d and from J, with the clipping of J's x and y
  var dcam = vec3f(j11 * dm2.x, j22 * dm2.y, -j11 * x * dm2.x - j22 * y * dm2.y);
  let valid_x = x >= clip_left && x <= clip_right;
  let valid_y = y >= clip_top && y <= clip_bottom;
  if (valid_x) { dcam.x -= fast_div(j11 * dj13, depth); }
  if (valid_y) { dcam.y -= fast_div(j22 * dj23, depth); }
  let fxv = 1.0 + select(0.0, 1.0, valid_x);
  let fyv = 1.0 + select(0.0, 1.0, valid_y);
  dcam.z += fast_div(j11 * (fxv * x_clipped * dj13 - dj11) + j22 * (fyv * y_clipped * dj23 - dj22), depth);
  put_means(i, vec3f(
    r1.x * dcam.x + r2.x * dcam.y + r3.x * dcam.z,
    r1.y * dcam.x + r2.y * dcam.y + r3.y * dcam.z,
    r1.z * dcam.x + r2.z * dcam.y + r3.z * dcam.z));

  // log-scale gradient
  let dvar = vec3f(
    R11 * R11 * d11 + R21 * R21 * d22 + R31 * R31 * d33 + 2.0 * (R11 * R21 * d12 + R11 * R31 * d13 + R21 * R31 * d23),
    R12 * R12 * d11 + R22 * R22 * d22 + R32 * R32 * d33 + 2.0 * (R12 * R22 * d12 + R12 * R32 * d13 + R22 * R32 * d23),
    R13 * R13 * d11 + R23 * R23 * d22 + R33 * R33 * d33 + 2.0 * (R13 * R23 * d12 + R13 * R33 * d13 + R23 * R33 * d23));
  put_scales(i, 2.0 * variance * dvar);

  // rotation-matrix gradient, then the normalised-quaternion backward
  let m11 = 2.0 * (S11 * d11 + S21 * d12 + S31 * d13);
  let m12 = 2.0 * (S12 * d11 + S22 * d12 + S32 * d13);
  let m13 = 2.0 * (S13 * d11 + S23 * d12 + S33 * d13);
  let m21 = 2.0 * (S11 * d12 + S21 * d22 + S31 * d23);
  let m22 = 2.0 * (S12 * d12 + S22 * d22 + S32 * d23);
  let m23 = 2.0 * (S13 * d12 + S23 * d22 + S33 * d23);
  let m31 = 2.0 * (S11 * d13 + S21 * d23 + S31 * d33);
  let m32 = 2.0 * (S12 * d13 + S22 * d23 + S32 * d33);
  let m33 = 2.0 * (S13 * d13 + S23 * d23 + S33 * d33);
  let dxx = m22 + m33; let dyy = m11 + m33; let dzz = m11 + m22;
  let drz = m21 - m12; let dxy = m21 + m12;
  let dry = m13 - m31; let dxz = m13 + m31;
  let drx = m32 - m23; let dyz = m32 + m23;
  let two = 2.0 * inv;
  let h = two * (xy * dxy + xz * dxz + yz * dyz + rx * drx + ry * dry + rz * drz - xx * dxx - yy * dyy - zz * dzz);
  var dq = two * vec4f(
    qx * drx + qy * dry + qz * drz - qr * h,
    qr * drx - 2.0 * qx * dxx + qy * dxy + qz * dxz - qx * h,
    qr * dry + qx * dxy - 2.0 * qy * dyy + qz * dyz - qy * h,
    qr * drz + qx * dxz + qy * dyz - 2.0 * qz * dzz - qz * h);
  if (BP.accumulate == 1u) { dq += vec4f(gQuats[4u * i], gQuats[4u * i + 1u], gQuats[4u * i + 2u], gQuats[4u * i + 3u]); }
  gQuats[4u * i] = dq.x; gQuats[4u * i + 1u] = dq.y; gQuats[4u * i + 2u] = dq.z; gQuats[4u * i + 3u] = dq.w;
}
`;

/** Opacity-logit and SH0 gradients from the accumulators: dL/dlogit as summed, dL/dsh0 = C0 dL/dcolour. */
export const FINALIZE_BACKWARD_WGSL = RASTER_PRELUDE + BACKWARD_PARAMS + /* wgsl */ `
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> BP: BackwardParams;
@group(0) @binding(2) var<storage, read> touched: array<u32>;
@group(0) @binding(3) var<storage, read> acc: array<u32>;
@group(0) @binding(4) var<storage, read_write> gOpacity: array<f32>;
@group(0) @binding(5) var<storage, read_write> gSh0: array<f32>;

@compute @workgroup_size(256) fn finalize_backward(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.n) { return; }
  var dOp = 0.0;
  var dSh = vec3f(0.0);
  if (touched[i] != 0u) {
    let o = i * ${ACC_STRIDE}u;
    dOp = bitcast<f32>(acc[o + 5u]);
    dSh = SH_C0 * vec3f(bitcast<f32>(acc[o + 6u]), bitcast<f32>(acc[o + 7u]), bitcast<f32>(acc[o + 8u]));
  } else if (BP.accumulate == 1u) {
    return;
  }
  if (BP.accumulate == 1u) {
    dOp += gOpacity[i];
    dSh += vec3f(gSh0[3u * i], gSh0[3u * i + 1u], gSh0[3u * i + 2u]);
  }
  gOpacity[i] = dOp;
  gSh0[3u * i] = dSh.x; gSh0[3u * i + 1u] = dSh.y; gSh0[3u * i + 2u] = dSh.z;
}
`;
