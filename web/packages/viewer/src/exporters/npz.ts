/**
 * Write a scene as NPZ with the key layout of gaussifier_sampler.web.save_scene,
 * so a browser result loads in Python with load_scene and back into this viewer.
 * Entries are stored uncompressed; NPY headers are version 1.0.
 */
import type { Scene } from "../types.js";
import { encodeNpy } from "./npy.js";
import { encodeZipStored } from "./zip.js";

export interface SceneExportOptions {
  /** Which state to export; default the last. */
  stateIndex?: number;
  seed?: number;
  profile?: string;
  meta?: Record<string, unknown>;
  includeImage?: boolean;
  includeRendered?: boolean;
  includeDensity?: boolean;
}

/** Serialize one state of a scene as a save_scene-compatible NPZ. */
export function sceneToNpz(scene: Scene, opts: SceneExportOptions = {}): Blob {
  const idx = opts.stateIndex ?? scene.states.length - 1;
  const st = scene.states[idx];
  if (!st) throw new Error("scene has no states");
  const n = st.xy.length / 2;
  const enc = new TextEncoder();
  const entries: Array<{ name: string; data: Uint8Array }> = [];
  const add = (key: string, data: Float32Array | Int32Array | BigInt64Array | Uint8Array, shape: number[]) => entries.push({ name: `${key}.npy`, data: encodeNpy(data, shape) });
  const seed = opts.seed ?? (typeof scene.meta?.seed === "number" ? (scene.meta.seed as number) : -1);
  add("format_version", Int32Array.of(1), []);
  add("width", Int32Array.of(scene.width), []);
  add("height", Int32Array.of(scene.height), []);
  add("seed", BigInt64Array.of(BigInt(Math.trunc(seed))), []);
  const profileBytes = enc.encode(opts.profile ?? (typeof scene.meta?.profile === "string" ? (scene.meta.profile as string) : "production"));
  add("profile", profileBytes, [profileBytes.length]);
  // `points` is the placement output in save_scene: the first state carries it when the scene keeps
  // the initializer; otherwise the exported state stands in.
  const first = scene.states.length > 1 ? scene.states[0] : st;
  add("points", first.xy.length === st.xy.length ? first.xy : st.xy, [n, 2]);
  if (scene.density && opts.includeDensity !== false) add("density", scene.density, [scene.height, scene.width]);
  if (scene.weights) add("point_weights", scene.weights, [n]);
  add("final_points", st.xy, [n, 2]);
  if (st.cov) add("final_log_covariance_channels", st.cov, [n, 3]);
  add("final_scale", st.scale, [n, 2]);
  add("final_rotation", st.rotation, [n, 1]);
  add("final_color", st.color, [n, 3]);
  if (st.rendered && opts.includeRendered !== false) add("final_rendered", st.rendered, [3, scene.height, scene.width]);
  if (scene.image instanceof Float32Array && opts.includeImage !== false) add("image", scene.image, [3, scene.height, scene.width]);
  const meta = { ...(scene.meta ?? {}), ...(opts.meta ?? {}), n, state_label: st.label ?? String(idx), exporter: "gaussifier-web viewer", crop: scene.crop ?? null };
  const metaBytes = enc.encode(JSON.stringify(meta));
  add("meta", metaBytes, [metaBytes.length]);
  const bytes = encodeZipStored(entries);
  return new Blob([bytes.buffer as ArrayBuffer], { type: "application/octet-stream" });
}
