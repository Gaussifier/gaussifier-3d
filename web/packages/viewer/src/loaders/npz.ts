import type { Scene, SceneState } from "../types.js";
import { covToScaleRotation } from "../math.js";
import { npyScalar, npyToFloat32, parseNpy, type NpyArray } from "./npy.js";
import { readZip, zipEntryBytes } from "./zip.js";

/** Parse every .npy entry of an NPZ archive into a map keyed by array name. */
export async function readNpz(buffer: ArrayBuffer): Promise<Map<string, NpyArray>> {
  const out = new Map<string, NpyArray>();
  for (const entry of readZip(buffer)) {
    if (!entry.name.endsWith(".npy")) continue;
    out.set(entry.name.slice(0, -4), parseNpy(await zipEntryBytes(entry)));
  }
  return out;
}

function stateFrom(
  xy: Float32Array,
  cov: Float32Array | undefined,
  color: Float32Array | undefined,
  scale?: Float32Array,
  rotation?: Float32Array,
  rendered?: Float32Array,
  label?: string,
): SceneState {
  const n = xy.length / 2;
  if (!scale || !rotation) {
    if (!cov) throw new Error(`state ${label ?? ""}: needs log-covariance channels or scale and rotation`);
    ({ scale, rotation } = covToScaleRotation(cov, n));
  }
  if (!color) {
    color = new Float32Array(n * 3).fill(1);
  }
  if (cov && cov.length !== n * 3) throw new Error("cov length mismatch");
  if (color.length !== n * 3) throw new Error("color length mismatch");
  return { xy, cov, scale, rotation, color, rendered, label };
}

/**
 * Build a Scene from an NPZ written by the CLI, the Python API helper, or `save_scene`.
 * Recognized keys: width, height, density [H, W]; either states_xy [K, N, 2] with states_cov
 * [K, N, 3] and states_color [K, N, 3], or points (+ init_log_covariance_channels or
 * init_scale/init_rotation, init_color, init_rendered) and final_points (+ final_log_covariance_channels,
 * final_color, final_rendered); optional image [3, H, W], point_weights [N], seed, profile, and meta (JSON).
 */
export async function loadNpz(buffer: ArrayBuffer): Promise<Scene> {
  const arrays = await readNpz(buffer);
  const f32 = (key: string): Float32Array | undefined => {
    const a = arrays.get(key);
    return a ? npyToFloat32(a) : undefined;
  };
  const scalar = (key: string): number | undefined => {
    const a = arrays.get(key);
    return a ? npyScalar(a) : undefined;
  };
  let meta: Record<string, unknown> = {};
  const metaArr = arrays.get("meta");
  if (metaArr && metaArr.data instanceof Uint8Array) {
    try { meta = JSON.parse(new TextDecoder().decode(metaArr.data)); } catch { meta = {}; }
  }
  let width = scalar("width");
  let height = scalar("height");
  const density = f32("density");
  const densityShape = arrays.get("density")?.shape;
  const renderedShape = (arrays.get("final_rendered") ?? arrays.get("init_rendered") ?? arrays.get("image"))?.shape;
  if ((!width || !height) && densityShape && densityShape.length === 2) { height = densityShape[0]; width = densityShape[1]; }
  if ((!width || !height) && renderedShape && renderedShape.length >= 2) { height = renderedShape[renderedShape.length - 2]; width = renderedShape[renderedShape.length - 1]; }
  if ((!width || !height) && typeof meta.width === "number" && typeof meta.height === "number") { width = meta.width; height = meta.height; }
  if (!width || !height) throw new Error("NPZ scene: cannot determine width and height (need width/height, density, or a rendered image)");

  const states: SceneState[] = [];
  const statesXy = f32("states_xy");
  if (statesXy) {
    const shape = arrays.get("states_xy")!.shape;
    const K = shape[0], N = shape[1];
    const statesCov = f32("states_cov");
    const statesColor = f32("states_color");
    for (let k = 0; k < K; k++) {
      const xy = statesXy.slice(k * N * 2, (k + 1) * N * 2);
      const cov = statesCov ? statesCov.slice(k * N * 3, (k + 1) * N * 3) : undefined;
      const color = statesColor ? statesColor.slice(k * N * 3, (k + 1) * N * 3) : undefined;
      states.push(stateFrom(xy, cov, color, undefined, undefined, undefined, `state ${k}`));
    }
  } else {
    const points = f32("points");
    const initCov = f32("init_log_covariance_channels");
    const initScale = f32("init_scale");
    const initRot = f32("init_rotation");
    // `points` alone (as written by the browser exporter) carries no initializer attributes; only build the init state when they exist.
    if (points && (initCov || (initScale && initRot))) {
      states.push(stateFrom(points, initCov, f32("init_color"), initScale, initRot, f32("init_rendered"), "initializer"));
    }
    const finalPoints = f32("final_points");
    if (finalPoints) {
      states.push(stateFrom(finalPoints, f32("final_log_covariance_channels"), f32("final_color"), undefined, undefined, f32("final_rendered"), "final"));
    }
  }
  if (states.length === 0) throw new Error("NPZ scene: no point states found (expected points, final_points, or states_xy)");

  const image = f32("image");
  const crop = (meta.crop && typeof meta.crop === "object") ? (meta.crop as { w: number; h: number }) : undefined;
  const seed = scalar("seed");
  if (seed !== undefined) meta.seed = seed;
  const profileArr = arrays.get("profile");
  if (profileArr && profileArr.data instanceof Uint8Array) meta.profile = new TextDecoder().decode(profileArr.data);
  const weights = f32("point_weights");
  if (weights) meta.point_weights = weights;
  return { width, height, crop, states, density, image, meta };
}
