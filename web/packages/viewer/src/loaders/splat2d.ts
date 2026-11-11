import type { Scene } from "../types.js";
import { SPLAT2D_MIN_SIGMA } from "../exporters/splat2d.js";

/** Parse a Splat2D `.splat2d` file into a one-state scene. Effective sigma = 1 / max(stored scale, 0.01). */
export function loadSplat2d(buffer: ArrayBuffer): Scene {
  if (buffer.byteLength < 12) throw new Error("splat2d: file too small");
  const dv = new DataView(buffer);
  const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
  if (magic !== "GS2D") throw new Error("splat2d: bad magic, expected GS2D");
  const n = dv.getUint32(4, true); const height = dv.getUint16(8, true); const width = dv.getUint16(10, true);
  if (buffer.byteLength !== 12 + 32 * n) throw new Error(`splat2d: ${buffer.byteLength} bytes, expected ${12 + 32 * n} for ${n} Gaussians`);
  const f32 = (offset: number, count: number) => new Float32Array(buffer.slice(offset, offset + 4 * count));
  const xy = f32(12, 2 * n);
  const raw = f32(12 + 8 * n, 2 * n);
  const rotation = f32(12 + 16 * n, n);
  const color = f32(12 + 20 * n, 3 * n);
  const scale = new Float32Array(2 * n);
  for (let i = 0; i < 2 * n; i++) scale[i] = 1 / Math.max(raw[i], SPLAT2D_MIN_SIGMA);
  return { width, height, states: [{ xy, scale, rotation, color, label: "final" }], meta: { n, format: "splat2d" } };
}
