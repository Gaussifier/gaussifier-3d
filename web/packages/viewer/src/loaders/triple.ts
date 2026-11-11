import type { Scene } from "../types.js";
import { covToScaleRotation } from "../math.js";
import { npyToFloat32, parseNpy } from "./npy.js";

/**
 * Load the C++ harness output triple: <base>.points.npy [N,2], <base>.cov.npy [N,3], <base>.rgb.npy [N,3].
 * The triple carries no frame size, so width and height must be supplied.
 */
export function loadNpyTriple(
  buffers: { points: ArrayBuffer; cov: ArrayBuffer; rgb: ArrayBuffer },
  frame: { width: number; height: number; crop?: { w: number; h: number } },
): Scene {
  const xy = npyToFloat32(parseNpy(new Uint8Array(buffers.points)));
  const cov = npyToFloat32(parseNpy(new Uint8Array(buffers.cov)));
  const color = npyToFloat32(parseNpy(new Uint8Array(buffers.rgb)));
  const n = xy.length / 2;
  if (cov.length !== n * 3 || color.length !== n * 3) throw new Error("NPY triple: array lengths disagree");
  const { scale, rotation } = covToScaleRotation(cov, n);
  return { width: frame.width, height: frame.height, crop: frame.crop, states: [{ xy, cov, scale, rotation, color, label: "final" }], meta: {} };
}
