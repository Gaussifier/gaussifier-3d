/** Geometry and color math shared by the loaders, backends, and tests. */

export const LN255 = Math.log(255);
/** Mahalanobis radius at which alpha drops below 1/255. */
export const CUTOFF_RADIUS = Math.sqrt(2 * LN255);
export const ALPHA_CUTOFF = 1 / 255;
/** Floats per instance in the packed instance array. */
export const INSTANCE_FLOATS = 12;

/** Principal scales and rotation from log-covariance channels [xx, xy, yy], exactly as model.py. */
export function covToScaleRotation(
  cov: ArrayLike<number>,
  n: number,
  scale: Float32Array = new Float32Array(n * 2),
  rotation: Float32Array = new Float32Array(n),
): { scale: Float32Array; rotation: Float32Array } {
  for (let i = 0; i < n; i++) {
    const a = cov[3 * i];
    const b = cov[3 * i + 1];
    const c = cov[3 * i + 2];
    const center = 0.5 * (a + c);
    const half = 0.5 * (a - c);
    const radius = Math.sqrt(half * half + b * b + 1e-12);
    scale[2 * i] = Math.exp(0.5 * (center + radius));
    scale[2 * i + 1] = Math.exp(0.5 * (center - radius));
    rotation[i] = -0.5 * Math.atan2(2 * b, a - c);
  }
  return { scale, rotation };
}

export interface GaussianProjection {
  /** Center in source pixels. */
  cx: number;
  cy: number;
  /** Covariance in source pixels squared. */
  sxx: number;
  sxy: number;
  syy: number;
  /** Inverse covariance [xx, xy, yy]. */
  cxx: number;
  cxy: number;
  cyy: number;
  valid: boolean;
}

/** Production projection: R = [[cos, sin], [-sin, cos]], S = diag(sx, sy), Sigma = (R S)(R S)^T. No scale clamp. */
export function projectGaussian(
  x: number, y: number, sx: number, sy: number, rot: number, width: number, height: number,
): GaussianProjection {
  const cx = x * width;
  const cy = y * height;
  const c = Math.cos(rot);
  const s = Math.sin(rot);
  const m00 = c * sx, m01 = s * sy, m10 = -s * sx, m11 = c * sy;
  const sxx = m00 * m00 + m01 * m01;
  const sxy = m00 * m10 + m01 * m11;
  const syy = m10 * m10 + m11 * m11;
  const det = sxx * syy - sxy * sxy;
  if (!(det > 0) || !Number.isFinite(det)) {
    return { cx, cy, sxx, sxy, syy, cxx: 0, cxy: 0, cyy: 0, valid: false };
  }
  const inv = 1 / det;
  return { cx, cy, sxx, sxy, syy, cxx: syy * inv, cxy: -sxy * inv, cyy: sxx * inv, valid: true };
}

export interface StateLike {
  xy: Float32Array;
  scale: Float32Array;
  rotation: Float32Array;
  color: Float32Array;
}

/**
 * Packed per-instance data: [cx, cy, cxx, cxy, cyy, r, g, b, hx, hy, valid, 0].
 * hx, hy are the half extents of the alpha-cutoff ellipse's bounding box in source pixels.
 */
export function buildInstances(state: StateLike, width: number, height: number): Float32Array {
  const n = state.xy.length / 2;
  const out = new Float32Array(n * INSTANCE_FLOATS);
  const k = 2 * LN255;
  for (let i = 0; i < n; i++) {
    const p = projectGaussian(state.xy[2 * i], state.xy[2 * i + 1], state.scale[2 * i], state.scale[2 * i + 1], state.rotation[i], width, height);
    const o = i * INSTANCE_FLOATS;
    out[o] = p.cx; out[o + 1] = p.cy;
    out[o + 2] = p.cxx; out[o + 3] = p.cxy; out[o + 4] = p.cyy;
    out[o + 5] = state.color[3 * i]; out[o + 6] = state.color[3 * i + 1]; out[o + 7] = state.color[3 * i + 2];
    out[o + 8] = p.valid ? Math.sqrt(k * p.sxx) : 0;
    out[o + 9] = p.valid ? Math.sqrt(k * p.syy) : 0;
    out[o + 10] = p.valid ? 1 : 0;
    out[o + 11] = 0;
  }
  return out;
}

/** PSNR in dB between two equally sized arrays of values in [0,1]. Infinity when identical. */
export function psnr(a: ArrayLike<number>, b: ArrayLike<number>, count = a.length): number {
  let mse = 0;
  for (let i = 0; i < count; i++) {
    const d = a[i] - b[i];
    mse += d * d;
  }
  mse /= Math.max(1, count);
  return mse === 0 ? Infinity : 10 * Math.log10(1 / mse);
}

/** PSNR over a crop of planar [3, H, W] images. */
export function psnrPlanar(a: Float32Array, b: Float32Array, width: number, height: number, cropW = width, cropH = height): number {
  let mse = 0;
  let count = 0;
  for (let c = 0; c < 3; c++) {
    for (let y = 0; y < cropH; y++) {
      const row = c * width * height + y * width;
      for (let x = 0; x < cropW; x++) {
        const d = a[row + x] - b[row + x];
        mse += d * d;
        count++;
      }
    }
  }
  mse /= Math.max(1, count);
  return mse === 0 ? Infinity : 10 * Math.log10(1 / mse);
}
