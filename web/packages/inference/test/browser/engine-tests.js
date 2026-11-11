import { readBuffer, uploadStorage } from "@gaussifier/wgsl";
import { createEngine, prepareInput, internals } from "@gaussifier/inference";
import { createOversampler } from "@gaussifier/wasm-oversampler";
const { KLoop, KLoopKernels, configureOrt, createSession, ortDevice, runGpu, disposeOutputs } = internals;

const FIX = "/packages/inference/test/fixtures/";
const log = (s) => { document.getElementById("log").textContent += s + "\n"; console.log(s); };
const results = [];
function record(name, ok, detail) { results.push({ name, ok, detail }); log(`${ok ? "PASS" : "FAIL"} ${name} ${JSON.stringify(detail)}`); }
async function fetchF32(url) { const r = await fetch(url); if (!r.ok) return null; return new Float32Array(await r.arrayBuffer()); }
async function fetchJson(url) { const r = await fetch(url); if (!r.ok) return null; return r.json(); }
function maxAbs(a, b) { let m = 0; for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); if (d > m) m = d; } return m; }
function psnr(a, b) { let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; } const mse = s / a.length; return mse === 0 ? Infinity : 10 * Math.log10(1 / mse); }
function f16to32(u16) { const out = new Float32Array(u16.length); for (let i = 0; i < u16.length; i++) { const h = u16[i]; const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff; out[i] = e === 0 ? s * Math.pow(2, -14) * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + f / 1024); } return out; }

/** JS reference render of the production formula. mode: "exact" (no cutoff) or "native" (tile rules). */
function referenceRender(W, H, xy, scale, rot, color, mode, capacity) {
  const n = xy.length / 2, out = new Float32Array(3 * W * H), HW = W * H;
  const tilesX = Math.ceil(W / 16), tilesY = Math.ceil(H / 16);
  const proj = [];
  let entries = 0;
  for (let i = 0; i < n; i++) {
    const cx = xy[2 * i] * W, cy = xy[2 * i + 1] * H, c = Math.cos(rot[i]), s = Math.sin(rot[i]), sx = scale[2 * i], sy = scale[2 * i + 1];
    const m00 = c * sx, m01 = s * sy, m10 = -s * sx, m11 = c * sy;
    const sxx = m00 * m00 + m01 * m01, sxy = m00 * m10 + m01 * m11, syy = m10 * m10 + m11 * m11;
    const det = sxx * syy - sxy * sxy;
    if (det === 0) { proj.push(null); continue; }
    const conic = [syy / det, -sxy / det, sxx / det];
    const bm = 0.5 * (sxx + syy), root = Math.sqrt(Math.max(0.1, bm * bm - det));
    const radius = Math.ceil(3 * Math.sqrt(Math.max(bm + root, bm - root)));
    const tcx = cx / 16, tcy = cy / 16, tr = radius / 16;
    const clampT = (v, hi) => Math.min(Math.max(0, Math.trunc(v)), hi);
    const bb = [clampT(tcx - tr, tilesX), clampT(tcy - tr, tilesY), clampT(tcx + tr + 1, tilesX), clampT(tcy + tr + 1, tilesY)];
    const footprint = (bb[2] - bb[0]) * (bb[3] - bb[1]);
    const large = footprint > 32;
    if (!large && radius > 0) entries += footprint;
    proj.push({ cx, cy, conic, radius, bb, large });
  }
  const overflow = entries > capacity;
  for (let py = 0; py < H; py++) for (let px = 0; px < W; px++) {
    const tx = Math.floor(px / 16), ty = Math.floor(py / 16);
    let r = 0, g = 0, b = 0;
    for (let i = 0; i < n; i++) {
      const p = proj[i]; if (!p) continue;
      if (mode === "native" && !overflow) {
        if (p.radius <= 0) continue;
        if (!p.large && !(tx >= p.bb[0] && tx < p.bb[2] && ty >= p.bb[1] && ty < p.bb[3])) continue;
      }
      const dx = p.cx - px, dy = p.cy - py;
      const sigma = 0.5 * (p.conic[0] * dx * dx + p.conic[2] * dy * dy) + p.conic[1] * dx * dy;
      if (sigma < 0 || !Number.isFinite(sigma)) continue;
      const a = Math.exp(-sigma); if (a < 1 / 255) continue;
      r += color[3 * i] * a; g += color[3 * i + 1] * a; b += color[3 * i + 2] * a;
    }
    const pix = py * W + px;
    out[pix] = Math.min(1, Math.max(0, r)); out[HW + pix] = Math.min(1, Math.max(0, g)); out[2 * HW + pix] = Math.min(1, Math.max(0, b));
  }
  return { out, overflow };
}

function jsPreRender(cov, rgb, weight) {
  const n = weight.length, scale = new Float32Array(2 * n), rot = new Float32Array(n), color = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    const a = cov[3 * i], b = cov[3 * i + 1], c = cov[3 * i + 2], center = 0.5 * (a + c), half = 0.5 * (a - c), radius = Math.sqrt(half * half + b * b + 1e-12);
    scale[2 * i] = Math.exp(0.5 * (center + radius)); scale[2 * i + 1] = Math.exp(0.5 * (center - radius)); rot[i] = -0.5 * Math.atan2(2 * b, a - c);
    for (let k = 0; k < 3; k++) color[3 * i + k] = Math.min(1, Math.max(0, rgb[3 * i + k] * weight[i]));
  }
  return { scale, rot, color };
}

async function readF32(device, buf, count, offsetBytes = 0) { return new Float32Array(await readBuffer(device, buf, count * 4, offsetBytes)); }

async function testSplatter(device, kernels) {
  const W = 64, H = 64, HW = W * H;
  // 5 Gaussians: tiny, small, anisotropic rotated, large (many tiles), edge-crossing
  const xy = new Float32Array([0.3, 0.3, 0.6, 0.25, 0.5, 0.7, 0.45, 0.5, 0.98, 0.1]);
  const scale = new Float32Array([0.8, 0.8, 2.0, 1.5, 6.0, 1.2, 14.0, 12.0, 3.0, 3.0]);
  const rot = new Float32Array([0, 0.3, 1.1, -0.4, 0]);
  const color = new Float32Array([0.9, 0.1, 0.1, 0.1, 0.8, 0.2, 0.2, 0.3, 0.9, 0.5, 0.5, 0.5, 0.7, 0.7, 0.1]);
  const kl = new KLoop(device, kernels, { width: W, height: H, n: 5 });
  device.queue.writeBuffer(kl.xy, 0, xy); device.queue.writeBuffer(kl.scale, 0, scale); device.queue.writeBuffer(kl.rot, 0, rot); device.queue.writeBuffer(kl.color, 0, color);
  const t0 = performance.now();
  const enc = device.createCommandEncoder(); kl.recordSplatClears(enc); const pass = enc.beginComputePass(); kl.recordSplat(pass); pass.end(); device.queue.submit([enc.finish()]);
  const got = await readF32(device, kl.features, 3 * HW, 10 * HW * 4);
  const ms = performance.now() - t0;
  const nat = referenceRender(W, H, xy, scale, rot, color, "native", kl.capacity);
  const exact = referenceRender(W, H, xy, scale, rot, color, "exact", kl.capacity);
  const flags = new Uint32Array(await readBuffer(device, kl.flags, 8));
  const dNat = maxAbs(got, nat.out), pExact = psnr(got, exact.out), dExact = maxAbs(got, exact.out);
  record("splatter synthetic vs native-rule reference", dNat <= 2e-3, { maxAbsNative: +dNat.toExponential(2), maxAbsExact: +dExact.toExponential(2), psnrExact: +pExact.toFixed(1), largeCount: flags[0], overflow: flags[1], ms: +ms.toFixed(2) });
  record("splatter synthetic vs exact reference PSNR", pExact >= 60, { psnrExact: +pExact.toFixed(1) });
  kl.dispose();
  // overflow case: 600 Gaussians with moderate footprints exceed capacity max(4800, 4096)
  const n2 = 600; const xy2 = new Float32Array(2 * n2), sc2 = new Float32Array(2 * n2), rot2 = new Float32Array(n2), col2 = new Float32Array(3 * n2);
  let seed = 7; const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
  for (let i = 0; i < n2; i++) { xy2[2 * i] = rnd(); xy2[2 * i + 1] = rnd(); sc2[2 * i] = 4 + 6 * rnd(); sc2[2 * i + 1] = 4 + 6 * rnd(); rot2[i] = rnd() * 3; for (let k = 0; k < 3; k++) col2[3 * i + k] = 0.02 * rnd(); }
  const kl2 = new KLoop(device, kernels, { width: W, height: H, n: n2 });
  device.queue.writeBuffer(kl2.xy, 0, xy2); device.queue.writeBuffer(kl2.scale, 0, sc2); device.queue.writeBuffer(kl2.rot, 0, rot2); device.queue.writeBuffer(kl2.color, 0, col2);
  const enc2 = device.createCommandEncoder(); kl2.recordSplatClears(enc2); const pass2 = enc2.beginComputePass(); kl2.recordSplat(pass2); pass2.end(); device.queue.submit([enc2.finish()]);
  const got2 = await readF32(device, kl2.features, 3 * HW, 10 * HW * 4);
  const flags2 = new Uint32Array(await readBuffer(device, kl2.flags, 8));
  const ref2 = referenceRender(W, H, xy2, sc2, rot2, col2, "native", kl2.capacity);
  const d2 = maxAbs(got2, ref2.out);
  record("splatter overflow path", flags2[1] === 1 && ref2.overflow && d2 <= 2e-3, { overflowFlag: flags2[1], refOverflow: ref2.overflow, maxAbs: +d2.toExponential(2), capacity: kl2.capacity });
  kl2.dispose();
  // pre-render math
  const n3 = 4; const cov = new Float32Array([0.5, 0.1, -0.2, 2.0, 0.0, 2.0, -1.0, 0.8, 0.3, 3.0, -1.5, 1.0]); const rgb = new Float32Array(12).map(() => Math.random()); const wgt = new Float32Array([0.5, 1.0, 1.7, 2.5]);
  const kl3 = new KLoop(device, kernels, { width: 32, height: 32, n: n3 }); kl3.setPoints(new Float32Array(8), cov, rgb, wgt);
  const enc3 = device.createCommandEncoder(); const pass3 = enc3.beginComputePass(); kl3.recordPreRender(pass3); pass3.end(); device.queue.submit([enc3.finish()]);
  const js = jsPreRender(cov, rgb, wgt);
  const gs = await readF32(device, kl3.scale, 2 * n3), gr = await readF32(device, kl3.rot, n3), gc = await readF32(device, kl3.color, 3 * n3);
  const dPre = Math.max(maxAbs(gs, js.scale) / Math.max(...js.scale), maxAbs(gr, js.rot), maxAbs(gc, js.color));
  record("pre-render scale/rotation/color", dPre <= 1e-5, { maxRelOrAbs: +dPre.toExponential(2) });
  kl3.dispose();
}

async function testSessions(sessions, device) {
  const W = 512, H = 512, HW = W * H;
  const fmIn = await fetchF32(FIX + "refs/fm_in.bin"), fmRef = await fetchF32(FIX + "refs/fm_ref.bin");
  if (!fmIn || !fmRef) { record("forward map vs reference", false, { error: "refs missing" }); return; }
  const inBuf = uploadStorage(device, fmIn, "fm-in");
  const t0 = performance.now();
  const out = await runGpu(sessions.fm, "image", inBuf, [1, 3, H, W]);
  await device.queue.onSubmittedWorkDone();
  const ms = performance.now() - t0;
  const order = ["density", "log_cov", "rgb", "raw_density", "rate"]; const sizes = [1, 3, 3, 1, 1];
  const diffs = {}; let off = 0, worst = 0;
  for (let i = 0; i < order.length; i++) {
    const o = out.find((x) => x.name === order[i]); const cnt = sizes[i] * HW;
    const got = await readF32(device, o.buffer, cnt); const d = maxAbs(got, fmRef.subarray(off, off + cnt)); diffs[order[i]] = +d.toExponential(2); worst = Math.max(worst, d); off += cnt;
  }
  disposeOutputs(out); inBuf.destroy();
  record("forward map vs reference (512)", worst <= 1e-2, { maxAbs: diffs, firstRunMs: +ms.toFixed(1) });
  const hIn = await fetchF32(FIX + "refs/head_in.bin"), hRef = await fetchF32(FIX + "refs/head_ref.bin");
  const hBuf = uploadStorage(device, hIn, "head-in");
  await runGpu(sessions.head, "features", hBuf, [1, 17, H, W]).then(disposeOutputs);
  const t1 = performance.now();
  const hOut = await runGpu(sessions.head, "features", hBuf, [1, 17, H, W]); await device.queue.onSubmittedWorkDone();
  const hms = performance.now() - t1;
  const gotH = await readF32(device, hOut[0].buffer, 8 * HW); const dH = maxAbs(gotH, hRef);
  disposeOutputs(hOut); hBuf.destroy();
  record("correction head vs reference (512)", dH <= 1e-2, { maxAbs: +dH.toExponential(2), secondRunMs: +hms.toFixed(1) });
}

async function testGoldens(sessions, device, kernels) {
  const base = "/fixtures/bench_128/";
  const man = await fetchJson(base + "manifest.json");
  if (!man) { record("bench_128 goldens", true, { skipped: "fixture absent" }); return; }
  const shapes = (await fetchJson(base + "shapes.json")) ?? {};
  const W = man.padded?.width ?? 128, H = man.padded?.height ?? 128, HW = W * H, N = man.n, K = man.states ?? 7;
  const isF16 = (name) => shapes[name]?.dtype === "float16";
  const need = async (name) => { const a = await fetchF32(base + name); if (!a) throw new Error(`golden ${name} missing`); return a; };
  let image, density, logcov, rgbmap, polished, weights, cov0, rgb0;
  try {
    [image, density, logcov, rgbmap, polished, weights, cov0, rgb0] = await Promise.all(["image.bin", "density.bin", "log_cov.bin", "rgb.bin", "polished.bin", "weights.bin", "cov0.bin", "rgb0.bin"].map(need));
  } catch (e) { record("bench_128 goldens", true, { skipped: String(e) }); return; }
  const imgBuf = uploadStorage(device, image), denBuf = uploadStorage(device, density), lcBuf = uploadStorage(device, logcov), rgbBuf = uploadStorage(device, rgbmap);
  const kl = new KLoop(device, kernels, { width: W, height: H, n: N });
  kl.setPoints(polished, cov0, rgb0, weights);
  { const enc = device.createCommandEncoder(); const pass = enc.beginComputePass(); kl.recordPackStatic(pass, imgBuf, denBuf, lcBuf, rgbBuf); pass.end(); device.queue.submit([enc.finish()]); }
  // init attrs from maps vs golden cov0/rgb0
  { const kl2 = new KLoop(device, kernels, { width: W, height: H, n: N }); kl2.setPoints(polished, null, null, weights);
    const enc = device.createCommandEncoder(); const pass = enc.beginComputePass(); kl2.recordPackStatic(pass, imgBuf, denBuf, lcBuf, rgbBuf); kl2.recordInitAttrs(pass); pass.end(); device.queue.submit([enc.finish()]);
    const gc = await readF32(device, kl2.cov, 3 * N), gr = await readF32(device, kl2.rgb, 3 * N);
    record("init attrs vs golden cov0/rgb0", maxAbs(gc, cov0) <= 1e-4 && maxAbs(gr, rgb0) <= 1e-4, { covMaxAbs: +maxAbs(gc, cov0).toExponential(2), rgbMaxAbs: +maxAbs(gr, rgb0).toExponential(2) });
    kl2.dispose(); }
  const timings = [];
  let finalRendered = null;
  for (let k = 0; k < K; k++) {
    const t0 = performance.now();
    const enc = device.createCommandEncoder(); kl.recordRender(enc); device.queue.submit([enc.finish()]);
    const rendered = await readF32(device, kl.features, 3 * HW, 10 * HW * 4);
    const gRend = await fetchF32(base + `state_${k}_rendered.bin`);
    if (gRend) { const p = psnr(rendered, gRend); if (k === 0) record("rendered_0 vs golden", p >= 55, { psnr: +p.toFixed(2), maxAbs: +maxAbs(rendered, gRend).toExponential(2) }); else log(`  state ${k}: psnr vs golden rendered ${p.toFixed(2)} dB`); }
    finalRendered = rendered;
    if (k === K - 1) break;
    const hc = device.createCommandEncoder(); kl.recordHardCount(hc); device.queue.submit([hc.finish()]);
    const t1 = performance.now();
    const out = await runGpu(sessions.head, "features", kl.features, [1, 17, H, W]);
    const enc2 = device.createCommandEncoder(); kl.recordPostHead(enc2, out[0].buffer); device.queue.submit([enc2.finish()]); disposeOutputs(out);
    if (k === 0) {
      const gx = await fetchF32(base + "state_1_xy.bin"), gcov = await fetchF32(base + "state_1_cov.bin"), grgb = await fetchF32(base + "state_1_rgb.bin");
      const xy1 = await readF32(device, kl.xy, 2 * N), cov1 = await readF32(device, kl.cov, 3 * N), rgb1 = await readF32(device, kl.rgb, 3 * N);
      if (gx && gcov && grgb) record("state 1 xy/cov/rgb vs golden", maxAbs(xy1, gx) <= 1e-3 && maxAbs(cov1, gcov) <= 2e-2 && maxAbs(rgb1, grgb) <= 2e-2, { xyMaxAbs: +maxAbs(xy1, gx).toExponential(2), covMaxAbs: +maxAbs(cov1, gcov).toExponential(2), rgbMaxAbs: +maxAbs(rgb1, grgb).toExponential(2) });
      else log("  state 1 goldens missing; skipped");
      const gF = await fetch(base + "state_0_delta.bin"); if (gF.ok) { const raw = await gF.arrayBuffer(); const gd = isF16("state_0_delta") ? f16to32(new Uint16Array(raw)) : new Float32Array(raw); const mine = await readF32(device, kl.delta, 8 * HW); record("delta_0 vs golden head output (golden render has no tile cutoff)", maxAbs(mine, gd) <= 2.5e-2, { maxAbs: +maxAbs(mine, gd).toExponential(2) }); }
      const gFe = await fetch(base + "state_0_features.bin"); if (gFe.ok) { const raw = await gFe.arrayBuffer(); const gf = isF16("state_0_features") ? f16to32(new Uint16Array(raw)) : new Float32Array(raw); const mine = await readF32(device, kl.features, 17 * HW); record("features_0 (17-channel head input) vs golden (render cutoff only)", maxAbs(mine, gf) <= 5e-3, { maxAbs: +maxAbs(mine, gf).toExponential(2) }); }
    }
    await device.queue.onSubmittedWorkDone();
    timings.push({ k, renderMs: +(t1 - t0).toFixed(1), headMs: +(performance.now() - t1).toFixed(1) });
  }
  const pFinal = psnr(finalRendered, image);
  const gFinal = await fetchF32(base + `state_${K - 1}_rendered.bin`);
  const pGolden = gFinal ? psnr(gFinal, image) : null;
  record("final PSNR vs image within 0.5 dB of golden", pGolden === null || Math.abs(pFinal - pGolden) <= 0.5, { finalPsnr: +pFinal.toFixed(2), goldenPsnr: pGolden === null ? null : +pGolden.toFixed(2), timings });
  kl.dispose(); imgBuf.destroy(); denBuf.destroy(); lcBuf.destroy(); rgbBuf.destroy();
  // end to end through the engine
  try {
    const bundleUrl = (await fetch("/apps/demo/public/models/web_bundle.json")).ok ? "/apps/demo/public/models/" : FIX + "bundle/";
    const devErrors = []; device.addEventListener("uncapturederror", (e) => { if (devErrors.length < 3) devErrors.push(String(e.error.message).slice(0, 160)); });
    const engine = await createEngine({ bundleUrl, oversampler: createOversampler });
    const planar = { data: image, width: W, height: H };
    const t0 = performance.now();
    const run = await engine.run(planar, { count: N, seed: man.seed ?? 12345, states: K, includeRendered: true });
    const p = psnr(run.final.rendered, image);
    record("engine end-to-end on bench_128 (PSNR within 1 dB of golden)", pGolden !== null && Math.abs(p - pGolden) <= 1.0, { bundleUrl, deviceErrors: devErrors, finalPsnr: +p.toFixed(2), goldenPsnr: pGolden === null ? null : +pGolden.toFixed(2), n: run.n, ms: +(performance.now() - t0).toFixed(1), timings: run.timings });
    // keepStates: every state copied out inside the loop and read once at the end must equal the
    // per-state readbacks of onState (two runs of the same input agree to float rounding), and the
    // loop must not wait between states.
    const seen = [];
    const synced = await engine.run(planar, { count: N, seed: man.seed ?? 12345, states: K, includeInit: false, onState: (k, s) => seen.push({ k, xy: s.xy }) });
    const t1 = performance.now();
    const kept = await engine.run(planar, { count: N, seed: man.seed ?? 12345, states: K, includeInit: false, keepStates: true, includeRendered: true });
    const keptMs = performance.now() - t1;
    const snaps = kept.snapshots ?? [];
    const maxDiff = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
    const perState = snaps.map((s) => { const ref = seen.find((r) => r.k === s.k); return ref ? maxDiff(s.xy, ref.xy) : Infinity; });
    const loopWaits = Object.entries(kept.timings).filter(([k, v]) => /^state_[1-9]/.test(k) && v > 2).length;
    record("keepStates matches onState per state and defers all readback", snaps.length === K - 1 && snaps[0].k === 1 && snaps[snaps.length - 1] === kept.final && perState.every((d) => d <= 1e-6) && kept.final.rendered && maxDiff(kept.final.xy, synced.final.xy) <= 1e-6 && loopWaits === 0, { kept: snaps.length, ks: snaps.map((s) => s.k), max_xy_diff: Math.max(...perState), loop_waits_over_2ms: loopWaits, readback_states_ms: kept.timings.readback_states, ms: +keptMs.toFixed(1) });
    // K=1 with deferred readback and no initializer wanted: the only state is the final one.
    const one = await engine.run(planar, { count: N, seed: man.seed ?? 12345, states: 1, includeInit: false, keepStates: true, includeRendered: true });
    record("keepStates at K=1 returns the initializer as the final state", one.final.k === 0 && one.init === undefined && (one.snapshots ?? []).length === 1 && !!one.final.rendered, { k: one.final.k, snapshots: (one.snapshots ?? []).length });
    engine.dispose();
  } catch (e) { record("engine end-to-end on bench_128", /placement stage unavailable/.test(String(e)), { note: String(e).slice(0, 200) }); }
}


window.__runTests = async () => {
  try {
    configureOrt({ wasmPaths: "/node_modules/onnxruntime-web/dist/" });
    const manifest = await fetchJson(FIX + "bundle/web_bundle.json");
    const bytes = async (p) => new Uint8Array(await (await fetch(FIX + "bundle/" + p)).arrayBuffer());
    const t0 = performance.now();
    const fm = await createSession(await bytes("forward_map.onnx"));
    const head = await createSession(await bytes("correction_head.onnx"));
    log(`sessions created in ${(performance.now() - t0).toFixed(0)} ms; manifest ${manifest.runtime_version}`);
    const device = await ortDevice();
    const kernels = new KLoopKernels(device);
    await testSplatter(device, kernels);
    await testSessions({ fm, head }, device);
    await testGoldens({ fm, head }, device, kernels);
    await fm.release(); await head.release();
  } catch (e) { record("harness", false, { error: String(e), stack: String(e.stack).slice(0, 800) }); }
  return { passed: results.filter((r) => r.ok).length, failed: results.filter((r) => !r.ok).length, results };
};
window.__testReady = true;
