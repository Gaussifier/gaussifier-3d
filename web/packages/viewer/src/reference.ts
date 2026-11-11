/** Brute-force CPU render of the production simple-sum formula. Used by tests and as documentation. */
import { ALPHA_CUTOFF, LN255, projectGaussian, type StateLike } from "./math.js";

/**
 * Returns planar [3, H, W] float32. Pixel j samples position j. Each Gaussian
 * contributes color * exp(-sigma) where alpha >= 1/255; the sum is clamped to [0,1].
 * `quantize` applies round(v * 255) / 255, matching an 8-bit display.
 */
export function bruteForceRender(state: StateLike, width: number, height: number, quantize = false): Float32Array {
  const out = new Float32Array(3 * width * height);
  const n = state.xy.length / 2;
  const k = 2 * LN255;
  for (let i = 0; i < n; i++) {
    const p = projectGaussian(state.xy[2 * i], state.xy[2 * i + 1], state.scale[2 * i], state.scale[2 * i + 1], state.rotation[i], width, height);
    if (!p.valid) continue;
    const hx = Math.sqrt(k * p.sxx);
    const hy = Math.sqrt(k * p.syy);
    const x0 = Math.max(0, Math.floor(p.cx - hx - 1));
    const x1 = Math.min(width - 1, Math.ceil(p.cx + hx + 1));
    const y0 = Math.max(0, Math.floor(p.cy - hy - 1));
    const y1 = Math.min(height - 1, Math.ceil(p.cy + hy + 1));
    const r = state.color[3 * i], g = state.color[3 * i + 1], b = state.color[3 * i + 2];
    for (let y = y0; y <= y1; y++) {
      const dy = p.cy - y;
      for (let x = x0; x <= x1; x++) {
        const dx = p.cx - x;
        const sigma = 0.5 * (p.cxx * dx * dx + p.cyy * dy * dy) + p.cxy * dx * dy;
        if (sigma < 0 || !Number.isFinite(sigma)) continue;
        const alpha = Math.exp(-sigma);
        if (alpha < ALPHA_CUTOFF) continue;
        const idx = y * width + x;
        out[idx] += r * alpha;
        out[width * height + idx] += g * alpha;
        out[2 * width * height + idx] += b * alpha;
      }
    }
  }
  for (let i = 0; i < out.length; i++) {
    let v = out[i] < 0 ? 0 : out[i] > 1 ? 1 : out[i];
    if (quantize) v = Math.round(v * 255) / 255;
    out[i] = v;
  }
  return out;
}
