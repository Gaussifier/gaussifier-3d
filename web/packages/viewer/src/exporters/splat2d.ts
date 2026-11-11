/**
 * Splat2D `.splat2d` writer: "GS2D", N (u32), H (u16), W (u16), then xy [N,2],
 * scale [N,2], rot [N], color [N,3] as little-endian float32. xy is normalized,
 * rotation is in radians, color is the renderer-ready weighted RGB. The stored
 * scale is the reciprocal of the effective pixel sigma, clamped at 0.01, which
 * is what Splat2D viewers and the training code's reader expect.
 */
import type { Scene } from "../types.js";

export const SPLAT2D_MIN_SIGMA = 0.01;

export interface Splat2dExportOptions {
  /** Which state to export; default the last. */
  stateIndex?: number;
  /**
   * "crop" (default) writes the original image size when the scene was edge padded and rescales
   * xy so every Gaussian keeps its pixel position; centers in the padding land outside [0, 1].
   * "padded" writes the padded frame as is.
   */
  frame?: "crop" | "padded";
}

export function sceneToSplat2d(scene: Scene, opts: Splat2dExportOptions = {}): Blob {
  const st = scene.states[opts.stateIndex ?? scene.states.length - 1];
  if (!st) throw new Error("scene has no states");
  const n = st.xy.length / 2;
  const frame = opts.frame !== "padded" && scene.crop ? scene.crop : { w: scene.width, h: scene.height };
  if (frame.w > 0xffff || frame.h > 0xffff) throw new Error("splat2d stores the frame size as 16-bit integers");
  const bytes = new Uint8Array(12 + 32 * n);
  const dv = new DataView(bytes.buffer);
  bytes.set([0x47, 0x53, 0x32, 0x44], 0);
  dv.setUint32(4, n, true); dv.setUint16(8, frame.h, true); dv.setUint16(10, frame.w, true);
  const xy = new Float32Array(bytes.buffer, 12, 2 * n);
  const fx = scene.width / frame.w, fy = scene.height / frame.h;
  for (let i = 0; i < n; i++) { xy[2 * i] = st.xy[2 * i] * fx; xy[2 * i + 1] = st.xy[2 * i + 1] * fy; }
  const scale = new Float32Array(bytes.buffer, 12 + 8 * n, 2 * n);
  for (let i = 0; i < 2 * n; i++) scale[i] = 1 / Math.max(st.scale[i], SPLAT2D_MIN_SIGMA);
  const rot = new Float32Array(bytes.buffer, 12 + 16 * n, n); rot.set(st.rotation.subarray(0, n));
  const color = new Float32Array(bytes.buffer, 12 + 20 * n, 3 * n); color.set(st.color);
  return new Blob([bytes.buffer], { type: "application/octet-stream" });
}
