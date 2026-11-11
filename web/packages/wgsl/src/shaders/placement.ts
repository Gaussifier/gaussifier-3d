/**
 * WGSL kernels for the placement stage (capturable native EMV semantics) plus
 * the generic primitives they need: exclusive prefix sum and a stable LSD radix
 * sort. Every kernel takes the same 32-byte uniform at binding 0:
 *   W, H            image size
 *   N               element count for per-point kernels (or length for scans)
 *   step            JFA step / sort shift / adjacency round / matching column
 *   aux             kernel-specific scalar (fill value, remove count, block count)
 */
import type { KernelSpec } from "../gpu.js";

export const SCAN_BLOCK = 1024;
export const SORT_BLOCK = 1024;
export const ADJ_K = 8;

const HEADER = `
struct P { W: u32, H: u32, N: u32, step: u32, aux: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> p: P;
`;

const FP = `
fn fp_encode_lo(v: f32) -> u32 { let fl = floor(v); return u32(min((v - fl) * 4294967296.0, 4294967295.0)); }
fn fp_encode_hi(v: f32) -> u32 { return u32(floor(v)); }
fn fp_decode(lo: u32, hi: u32) -> f32 { return f32(hi) + f32(lo) * 2.3283064365386963e-10; }
`;

const FP_ADD = `
fn fp_add(base: u32, i: u32, v: f32) {
  let lo = fp_encode_lo(v); let hi = fp_encode_hi(v);
  let old = atomicAdd(&ACC[base + 2u * i], lo);
  var carry = 0u; if (old + lo < old) { carry = 1u; }
  atomicAdd(&ACC[base + 2u * i + 1u], hi + carry);
}
`;

const D2 = `
fn d2(c: i32, x: u32, y: u32) -> f32 {
  let q = pix[u32(c)];
  let dx = f32(x) + 0.5 - q.x; let dy = f32(y) + 0.5 - q.y;
  return dx * dx + dy * dy;
}
`;

function k(label: string, bindings: KernelSpec["bindings"], body: string): KernelSpec {
  return { label, entryPoint: "main", bindings, code: HEADER + body };
}
const U: KernelSpec["bindings"][number] = "uniform";
const R: KernelSpec["bindings"][number] = "read-only-storage";
const W: KernelSpec["bindings"][number] = "storage";

export const PLACEMENT_KERNELS = {
  /** buf[i] = aux for i < N. Used for -1 / 0xFFFFFFFF / 0 fills inside a pass. */
  fill_u32: k("fill_u32", [U, W], `
@group(0) @binding(1) var<storage, read_write> buf: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x < p.N) { buf[g.x] = p.aux; }
}`),

  /** Normalized points -> clamped pixel coordinates. */
  to_pixel: k("to_pixel", [U, R, W], `
@group(0) @binding(1) var<storage, read> pts: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> pix: array<vec2<f32>>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= p.N) { return; }
  let q = pts[g.x];
  pix[g.x] = vec2<f32>(clamp(q.x * f32(p.W), 0.0, f32(p.W) - 1e-3), clamp(q.y * f32(p.H), 0.0, f32(p.H) - 1e-3));
}`),

  /** Pixel-snap seed: owner[cell] = max index among colliding points. */
  owner_init: k("owner_init", [U, R, W], `
@group(0) @binding(1) var<storage, read> pix: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> owner: array<atomic<i32>>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= p.N) { return; }
  let q = pix[g.x];
  let cx = clamp(u32(floor(q.x)), 0u, p.W - 1u); let cy = clamp(u32(floor(q.y)), 0u, p.H - 1u);
  atomicMax(&owner[cy * p.W + cx], i32(g.x));
}`),

  /** One jump-flood pass, ping-pong, distances recomputed from owners. */
  jfa_pass: k("jfa_pass", [U, R, R, W], `
@group(0) @binding(1) var<storage, read> pix: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> owner_in: array<i32>;
@group(0) @binding(3) var<storage, read_write> owner_out: array<i32>;
${D2}
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= p.W || g.y >= p.H) { return; }
  let idx = g.y * p.W + g.x;
  var best = owner_in[idx];
  var bd = 1e30; if (best >= 0) { bd = d2(best, g.x, g.y); }
  let s = i32(p.step);
  for (var dy = -1; dy <= 1; dy++) {
    let ny = i32(g.y) + dy * s; if (ny < 0 || ny >= i32(p.H)) { continue; }
    for (var dx = -1; dx <= 1; dx++) {
      if (dx == 0 && dy == 0) { continue; }
      let nx = i32(g.x) + dx * s; if (nx < 0 || nx >= i32(p.W)) { continue; }
      let c = owner_in[u32(ny) * p.W + u32(nx)]; if (c < 0) { continue; }
      let d = d2(c, g.x, g.y); if (d < bd) { bd = d; best = c; }
    }
  }
  owner_out[idx] = best;
}`),

  /** seen[owner] = 1 for every pixel with an owner. */
  mark_seen: k("mark_seen", [U, R, W], `
@group(0) @binding(1) var<storage, read> owner: array<i32>;
@group(0) @binding(2) var<storage, read_write> seen: array<u32>;
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= p.W || g.y >= p.H) { return; }
  let o = owner[g.y * p.W + g.x]; if (o >= 0) { seen[u32(o)] = 1u; }
}`),

  /** flags[i] = 1 when point i owns no pixel. */
  unseen_flags: k("unseen_flags", [U, R, W], `
@group(0) @binding(1) var<storage, read> seen: array<u32>;
@group(0) @binding(2) var<storage, read_write> flags: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= p.N) { return; }
  flags[g.x] = select(1u, 0u, seen[g.x] != 0u);
}`),

  /** Exclusive scan of one 1024-element block; writes block totals. N = length. */
  scan_local: k("scan_local", [U, R, W, W], `
@group(0) @binding(1) var<storage, read> inp: array<u32>;
@group(0) @binding(2) var<storage, read_write> outp: array<u32>;
@group(0) @binding(3) var<storage, read_write> sums: array<u32>;
var<workgroup> ws: array<u32, 256>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
  let base = wg.x * 1024u + l.x * 4u;
  var v: array<u32, 4>; var s = 0u;
  for (var kk = 0u; kk < 4u; kk++) { let i = base + kk; var x = 0u; if (i < p.N) { x = inp[i]; } v[kk] = x; s += x; }
  ws[l.x] = s; workgroupBarrier();
  for (var off = 1u; off < 256u; off = off << 1u) {
    var t = 0u; if (l.x >= off) { t = ws[l.x - off]; }
    workgroupBarrier(); ws[l.x] = ws[l.x] + t; workgroupBarrier();
  }
  let incl = ws[l.x]; var run = incl - s;
  for (var kk = 0u; kk < 4u; kk++) { let i = base + kk; if (i < p.N) { outp[i] = run; } run += v[kk]; }
  if (l.x == 255u) { sums[wg.x] = incl; }
}`),

  /** outp[i] += scanned block sum of its block. */
  scan_add: k("scan_add", [U, W, R], `
@group(0) @binding(1) var<storage, read_write> outp: array<u32>;
@group(0) @binding(2) var<storage, read> sums: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= p.N) { return; }
  let b = g.x / 1024u; if (b > 0u) { outp[g.x] = outp[g.x] + sums[b]; }
}`),

  /** Stable compaction of flagged indices; counts[0] = total. */
  compact_lost: k("compact_lost", [U, R, R, W, W], `
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read> scan: array<u32>;
@group(0) @binding(3) var<storage, read_write> lost: array<u32>;
@group(0) @binding(4) var<storage, read_write> counts: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= p.N) { return; }
  if (flags[g.x] == 1u) { lost[scan[g.x]] = g.x; }
  if (g.x == p.N - 1u) { counts[0] = scan[g.x] + flags[g.x]; }
}`),

  /** Exact recovery for lost points; copies owner_in to owner_out when none are lost. */
  recovery: k("recovery", [U, R, R, W, R, R], `
@group(0) @binding(1) var<storage, read> pix: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> owner_in: array<i32>;
@group(0) @binding(3) var<storage, read_write> owner_out: array<i32>;
@group(0) @binding(4) var<storage, read> lost: array<u32>;
@group(0) @binding(5) var<storage, read> counts: array<u32>;
var<workgroup> sx: array<f32, 128>; var<workgroup> sy: array<f32, 128>; var<workgroup> si: array<i32, 128>;
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let inside = (g.x < p.W) && (g.y < p.H);
  let idx = select(0u, g.y * p.W + g.x, inside);
  var best = -1; var bd = 1e30;
  if (inside) { best = owner_in[idx]; if (best >= 0) { let q = pix[u32(best)]; let dx = f32(g.x) + 0.5 - q.x; let dy = f32(g.y) + 0.5 - q.y; bd = dx * dx + dy * dy; } }
  let L = counts[0];
  for (var t = 0u; t < L; t += 128u) {
    let n = min(128u, L - t);
    if (li < n) { let c = i32(lost[t + li]); let q = pix[u32(c)]; sx[li] = q.x; sy[li] = q.y; si[li] = c; }
    workgroupBarrier();
    if (inside) {
      for (var j = 0u; j < n; j++) {
        let dx = f32(g.x) + 0.5 - sx[j]; let dy = f32(g.y) + 0.5 - sy[j]; let d = dx * dx + dy * dy;
        if (d < bd) { bd = d; best = si[j]; }
      }
    }
    workgroupBarrier();
  }
  if (inside) { owner_out[idx] = best; }
}`),

  /** Fixed-point cell masses into ACC section 0. */
  mass_accumulate: k("mass_accumulate", [U, R, R, W], `
@group(0) @binding(1) var<storage, read> owner: array<i32>;
@group(0) @binding(2) var<storage, read> density: array<f32>;
@group(0) @binding(3) var<storage, read_write> ACC: array<atomic<u32>>;
${FP}${FP_ADD}
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= p.W || g.y >= p.H) { return; }
  let idx = g.y * p.W + g.x; let o = owner[idx]; let d = density[idx];
  if (o < 0 || d <= 0.0) { return; }
  fp_add(0u, u32(o), d);
}`),

  /** mass[i] = decode(ACC section 0). */
  mass_decode: k("mass_decode", [U, R, W], `
@group(0) @binding(1) var<storage, read> ACC: array<u32>;
@group(0) @binding(2) var<storage, read_write> mass: array<f32>;
${FP}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= p.N) { return; }
  mass[g.x] = fp_decode(ACC[2u * g.x], ACC[2u * g.x + 1u]);
}`),

  /** keys = bitcast(mass), vals = index. */
  sort_init: k("sort_init", [U, R, W, W], `
@group(0) @binding(1) var<storage, read> mass: array<f32>;
@group(0) @binding(2) var<storage, read_write> keys: array<u32>;
@group(0) @binding(3) var<storage, read_write> vals: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= p.N) { return; }
  keys[g.x] = bitcast<u32>(max(mass[g.x], 0.0)); vals[g.x] = g.x;
}`),

  /** Per-block digit histogram, layout hist[digit * numBlocks + block]. step = shift, aux = numBlocks. */
  sort_hist: k("sort_hist", [U, R, W], `
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> hist: array<u32>;
var<workgroup> lh: array<atomic<u32>, 256>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
  atomicStore(&lh[l.x], 0u); workgroupBarrier();
  let base = wg.x * 1024u + l.x * 4u;
  for (var kk = 0u; kk < 4u; kk++) { let i = base + kk; if (i < p.N) { let d = (keys[i] >> p.step) & 255u; atomicAdd(&lh[d], 1u); } }
  workgroupBarrier();
  hist[l.x * p.aux + wg.x] = atomicLoad(&lh[l.x]);
}`),

  /** Stable scatter using scanned histogram offsets. step = shift, aux = numBlocks. */
  sort_scatter: k("sort_scatter", [U, R, R, R, W, W], `
@group(0) @binding(1) var<storage, read> keys_in: array<u32>;
@group(0) @binding(2) var<storage, read> vals_in: array<u32>;
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> vals_out: array<u32>;
var<workgroup> sd: array<u32, 1024>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
  let base = wg.x * 1024u;
  for (var kk = 0u; kk < 4u; kk++) { let j = l.x * 4u + kk; let i = base + j; var d = 0xFFFFFFFFu; if (i < p.N) { d = (keys_in[i] >> p.step) & 255u; } sd[j] = d; }
  workgroupBarrier();
  var d: array<u32, 4>; var cnt: array<u32, 4>;
  for (var kk = 0u; kk < 4u; kk++) { d[kk] = sd[l.x * 4u + kk]; cnt[kk] = 0u; }
  let mine = l.x * 4u;
  for (var m = 0u; m < mine; m++) {
    let s = sd[m];
    for (var kk = 0u; kk < 4u; kk++) { if (s == d[kk]) { cnt[kk] += 1u; } }
  }
  for (var kk = 1u; kk < 4u; kk++) { for (var jj = 0u; jj < kk; jj++) { if (d[jj] == d[kk]) { cnt[kk] += 1u; } } }
  for (var kk = 0u; kk < 4u; kk++) {
    let i = base + l.x * 4u + kk;
    if (i < p.N) { let pos = offsets[d[kk] * p.aux + wg.x] + cnt[kk]; keys_out[pos] = keys_in[i]; vals_out[pos] = vals_in[i]; }
  }
}`),

  /** rank[vals[r]] = r; idx_of_rank[r] = vals[r]. */
  rank_from_sorted: k("rank_from_sorted", [U, R, W, W], `
@group(0) @binding(1) var<storage, read> vals: array<u32>;
@group(0) @binding(2) var<storage, read_write> rank: array<u32>;
@group(0) @binding(3) var<storage, read_write> idx_of_rank: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= p.N) { return; }
  let v = vals[g.x]; rank[v] = g.x; idx_of_rank[g.x] = v;
}`),

  /** One adjacency round r = step: slot[src][r] = min rank(dst) > slot[src][r-1]. */
  adjacency_round: k("adjacency_round", [U, R, R, W], `
@group(0) @binding(1) var<storage, read> owner: array<i32>;
@group(0) @binding(2) var<storage, read> rank: array<u32>;
@group(0) @binding(3) var<storage, read_write> slot: array<atomic<u32>>;
fn ins(src: i32, dst: i32) {
  let kk = rank[u32(dst)]; let r = p.step; var ok = true;
  if (r > 0u) { let l = atomicLoad(&slot[u32(src) * 8u + r - 1u]); ok = (l != 0xFFFFFFFFu) && (kk > l); }
  if (ok) { atomicMin(&slot[u32(src) * 8u + r], kk); }
}
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= p.W || g.y >= p.H) { return; }
  let idx = g.y * p.W + g.x; let a = owner[idx];
  if (a < 0) { return; }
  if (g.x + 1u < p.W) { let b = owner[idx + 1u]; if (b >= 0 && a != b) { ins(a, b); ins(b, a); } }
  if (g.y + 1u < p.H) { let b = owner[idx + p.W]; if (b >= 0 && a != b) { ins(a, b); ins(b, a); } }
}`),

  /** Matching phase 1 for column step; aux = remove. proposed[i] = tgt or -1; atomicMin tmin. */
  match_phase1: k("match_phase1", [U, R, R, R, R, R, W, W], `
@group(0) @binding(1) var<storage, read> rank: array<u32>;
@group(0) @binding(2) var<storage, read> slot: array<u32>;
@group(0) @binding(3) var<storage, read> idx_of_rank: array<u32>;
@group(0) @binding(4) var<storage, read> alive: array<u32>;
@group(0) @binding(5) var<storage, read> matched: array<u32>;
@group(0) @binding(6) var<storage, read_write> tmin: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> proposed: array<i32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.N) { return; }
  let proposer = (rank[i] < p.aux) && (alive[i] == 1u) && (matched[i] == 0u);
  let sv = slot[i * 8u + p.step];
  var t = i32(i);
  if (sv != 0xFFFFFFFFu) { t = i32(idx_of_rank[sv]); }
  let valid = proposer && (t != i32(i)) && (alive[u32(t)] == 1u);
  if (valid) { proposed[i] = t; atomicMin(&tmin[u32(t)], rank[i]); } else { proposed[i] = -1; }
}`),

  /** Matching phase 2: winners take their tgt. */
  match_phase2: k("match_phase2", [U, R, R, R, W, W, W], `
@group(0) @binding(1) var<storage, read> rank: array<u32>;
@group(0) @binding(2) var<storage, read> proposed: array<i32>;
@group(0) @binding(3) var<storage, read> tmin: array<u32>;
@group(0) @binding(4) var<storage, read_write> tgt: array<i32>;
@group(0) @binding(5) var<storage, read_write> matched: array<u32>;
@group(0) @binding(6) var<storage, read_write> alive: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.N) { return; }
  let t = proposed[i]; if (t < 0) { return; }
  if (rank[i] == tmin[u32(t)]) { tgt[i] = t; matched[i] = 1u; alive[i] = 0u; }
}`),

  /** Force-match stalled candidates (rank < aux) to the highest-mass point. */
  match_force: k("match_force", [U, R, R, W, W, W], `
@group(0) @binding(1) var<storage, read> rank: array<u32>;
@group(0) @binding(2) var<storage, read> idx_of_rank: array<u32>;
@group(0) @binding(3) var<storage, read_write> tgt: array<i32>;
@group(0) @binding(4) var<storage, read_write> matched: array<u32>;
@group(0) @binding(5) var<storage, read_write> alive: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.N) { return; }
  let fb = idx_of_rank[p.N - 1u];
  if (rank[i] < p.aux && matched[i] == 0u && alive[i] == 1u && fb != i) { tgt[i] = i32(fb); matched[i] = 1u; alive[i] = 0u; }
}`),

  /** ACC init for the merge: section0 = m, section1 = x*m, section2 = y*m (plain stores). */
  merge_acc_init: k("merge_acc_init", [U, R, R, W], `
@group(0) @binding(1) var<storage, read> pts: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> mass: array<f32>;
@group(0) @binding(3) var<storage, read_write> ACC: array<u32>;
${FP}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.N) { return; }
  let m = mass[i]; let q = pts[i]; let n2 = 2u * p.N;
  ACC[2u * i] = fp_encode_lo(m); ACC[2u * i + 1u] = fp_encode_hi(m);
  ACC[n2 + 2u * i] = fp_encode_lo(q.x * m); ACC[n2 + 2u * i + 1u] = fp_encode_hi(q.x * m);
  ACC[2u * n2 + 2u * i] = fp_encode_lo(q.y * m); ACC[2u * n2 + 2u * i + 1u] = fp_encode_hi(q.y * m);
}`),

  /** Matched sources add their original mass-weighted position and mass into their tgt. */
  merge_accumulate: k("merge_accumulate", [U, R, R, R, R, W], `
@group(0) @binding(1) var<storage, read> pts: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> mass: array<f32>;
@group(0) @binding(3) var<storage, read> matched: array<u32>;
@group(0) @binding(4) var<storage, read> tgt: array<i32>;
@group(0) @binding(5) var<storage, read_write> ACC: array<atomic<u32>>;
${FP}${FP_ADD}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.N) { return; }
  if (matched[i] != 1u) { return; }
  let t = tgt[i]; if (t < 0) { return; }
  let m = mass[i]; let q = pts[i]; let n2 = 2u * p.N;
  fp_add(0u, u32(t), m); fp_add(n2, u32(t), q.x * m); fp_add(2u * n2, u32(t), q.y * m);
}`),

  /** new[t] = mp > 1e-8 ? wp / mp : p[t]. */
  merge_new: k("merge_new", [U, R, R, W], `
@group(0) @binding(1) var<storage, read> pts: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> ACC: array<u32>;
@group(0) @binding(3) var<storage, read_write> pts_new: array<vec2<f32>>;
${FP}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.N) { return; }
  let n2 = 2u * p.N;
  let mp = fp_decode(ACC[2u * i], ACC[2u * i + 1u]);
  if (mp > 1e-8) {
    let wx = fp_decode(ACC[n2 + 2u * i], ACC[n2 + 2u * i + 1u]);
    let wy = fp_decode(ACC[2u * n2 + 2u * i], ACC[2u * n2 + 2u * i + 1u]);
    pts_new[i] = vec2<f32>(wx / mp, wy / mp);
  } else { pts_new[i] = pts[i]; }
}`),

  /** Compact alive rows in index order. */
  compact_alive: k("compact_alive", [U, R, R, R, W], `
@group(0) @binding(1) var<storage, read> alive: array<u32>;
@group(0) @binding(2) var<storage, read> scan: array<u32>;
@group(0) @binding(3) var<storage, read> pts_new: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> pts_out: array<vec2<f32>>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.N) { return; }
  if (alive[i] == 1u) { pts_out[scan[i]] = pts_new[i]; }
}`),

  /** Lloyd accumulators: section0 = wsum, section1 = sum d*cx, section2 = sum d*cy with normalized pixel centers. */
  lloyd_accumulate: k("lloyd_accumulate", [U, R, R, W], `
@group(0) @binding(1) var<storage, read> owner: array<i32>;
@group(0) @binding(2) var<storage, read> density: array<f32>;
@group(0) @binding(3) var<storage, read_write> ACC: array<atomic<u32>>;
${FP}${FP_ADD}
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= p.W || g.y >= p.H) { return; }
  let idx = g.y * p.W + g.x; let o = owner[idx]; let d = density[idx];
  if (o < 0 || d <= 0.0) { return; }
  let cx = (f32(g.x) + 0.5) / f32(p.W); let cy = (f32(g.y) + 0.5) / f32(p.H);
  let n2 = 2u * p.N;
  fp_add(0u, u32(o), d); fp_add(n2, u32(o), d * cx); fp_add(2u * n2, u32(o), d * cy);
}`),

  /** Lloyd finalize in place. */
  lloyd_finalize: k("lloyd_finalize", [U, R, W], `
@group(0) @binding(1) var<storage, read> ACC: array<u32>;
@group(0) @binding(2) var<storage, read_write> pts: array<vec2<f32>>;
${FP}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.N) { return; }
  let n2 = 2u * p.N;
  let w = fp_decode(ACC[2u * i], ACC[2u * i + 1u]);
  if (w > 1e-8) {
    let x = fp_decode(ACC[n2 + 2u * i], ACC[n2 + 2u * i + 1u]) / w;
    let y = fp_decode(ACC[2u * n2 + 2u * i], ACC[2u * n2 + 2u * i + 1u]) / w;
    pts[i] = vec2<f32>(clamp(x, 0.0, 1.0), clamp(y, 0.0, 1.0));
  }
}`),

  /** Block partial sums of a f32 array (1024 elements per block). */
  reduce_partial: k("reduce_partial", [U, R, W], `
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> partial: array<f32>;
var<workgroup> ws: array<f32, 256>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
  let base = wg.x * 1024u + l.x * 4u; var s = 0.0;
  for (var kk = 0u; kk < 4u; kk++) { let i = base + kk; if (i < p.N) { s += inp[i]; } }
  ws[l.x] = s; workgroupBarrier();
  for (var off = 128u; off > 0u; off = off >> 1u) { if (l.x < off) { ws[l.x] += ws[l.x + off]; } workgroupBarrier(); }
  if (l.x == 0u) { partial[wg.x] = ws[0]; }
}`),

  /** Sum of the partials into total[0]; N = number of partials. Single workgroup. */
  reduce_final: k("reduce_final", [U, R, W], `
@group(0) @binding(1) var<storage, read> partial: array<f32>;
@group(0) @binding(2) var<storage, read_write> total: array<f32>;
var<workgroup> ws: array<f32, 256>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_id) l: vec3<u32>) {
  var s = 0.0;
  for (var i = l.x; i < p.N; i += 256u) { s += partial[i]; }
  ws[l.x] = s; workgroupBarrier();
  for (var off = 128u; off > 0u; off = off >> 1u) { if (l.x < off) { ws[l.x] += ws[l.x + off]; } workgroupBarrier(); }
  if (l.x == 0u) { total[0] = ws[0]; }
}`),

  /** weight[i] = mass[i] / (max(total, 1e-8) / N). */
  weights_finalize: k("weights_finalize", [U, R, R, W], `
@group(0) @binding(1) var<storage, read> mass: array<f32>;
@group(0) @binding(2) var<storage, read> total: array<f32>;
@group(0) @binding(3) var<storage, read_write> weights: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.N) { return; }
  weights[i] = mass[i] / (max(total[0], 1e-8) / f32(p.N));
}`),
} as const;

export type PlacementKernelName = keyof typeof PLACEMENT_KERNELS;
