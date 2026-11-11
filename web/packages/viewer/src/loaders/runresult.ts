import type { Scene, SceneState } from "../types.js";
import { covToScaleRotation } from "../math.js";

/** Structural mirror of the engine's StateSnapshot and RunResult. */
export interface SnapshotLike {
  k: number;
  xy: Float32Array;
  cov: Float32Array;
  color: Float32Array;
  rendered?: Float32Array;
}

/** The fields of the engine's RunResult this loader reads; kept structural so the viewer does not depend on the engine package. */
export interface RunResultLike {
  width: number;
  height: number;
  crop: { w: number; h: number };
  n: number;
  seed: number;
  density: Float32Array;
  weight?: Float32Array;
  init?: SnapshotLike;
  final: SnapshotLike;
  /** Every kept state in order when the run used keepStates; the last one is `final`. */
  snapshots?: SnapshotLike[];
}

/**
 * A SceneState whose scale and rotation are derived from the log-covariance on first access
 * (plain getters, so it reads like any other state), so a result with many kept states only
 * pays for the ones that are shown or exported.
 */
function toState(s: SnapshotLike, label: string): SceneState {
  const n = s.xy.length / 2;
  let derived: { scale: Float32Array; rotation: Float32Array } | null = null;
  const get = () => (derived ??= covToScaleRotation(s.cov, n));
  const state = { xy: s.xy, cov: s.cov, color: s.color, rendered: s.rendered, label } as SceneState;
  Object.defineProperty(state, "scale", { get: () => get().scale, set: (v: Float32Array) => { derived = { scale: v, rotation: get().rotation }; }, enumerable: true, configurable: true });
  Object.defineProperty(state, "rotation", { get: () => get().rotation, set: (v: Float32Array) => { derived = { scale: get().scale, rotation: v }; }, enumerable: true, configurable: true });
  return state;
}

/** Wrap an engine result as a viewer scene without copying the arrays. */
export function fromRunResult(run: RunResultLike): Scene {
  const states: SceneState[] = [];
  // State 0 is the initializer; every later state is one correction step.
  const label = (k: number) => (k === 0 ? "initializer" : `refinement ${k}`);
  if (run.snapshots && run.snapshots.length > 0) {
    for (const s of run.snapshots) states.push(toState(s, label(s.k)));
  } else {
    if (run.init) states.push(toState(run.init, label(run.init.k)));
    if (run.final !== run.init) states.push(toState(run.final, label(run.final.k)));
  }
  return { width: run.width, height: run.height, crop: run.crop, states, density: run.density, weights: run.weight, meta: { n: run.n, seed: run.seed } };
}
