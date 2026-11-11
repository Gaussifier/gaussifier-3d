/**
 * The planar model bundle: `planar_bundle.json` plus one fp32 ONNX file per graph (the release
 * assets `planar_*_web`). The graphs take any
 * image size whose sides divide by `geometry.size_multiple`; the cameras and the slab follow
 * from the size (`camerasFor`).
 */
import { internals } from "@gaussifier/inference";

export const GRAPH_NAMES = ["forward_map", "setup", "pre_render", "head", "update"] as const;
export type GraphName = (typeof GRAPH_NAMES)[number];

/** Faster-GS rasterizer camera (the runtime's `CameraSettings`). */
export interface PlanarCamera {
  /** World-to-camera matrix, 16 values, row major. */
  w2c: number[];
  position: number[];
  width: number;
  height: number;
  focal_x: number;
  focal_y: number;
  center_x: number;
  center_y: number;
  near: number;
  far: number;
}

export interface GraphEntry {
  path: string;
  sha256: string;
  bytes: number;
  inputs: string[];
  outputs: string[];
}

export interface PlanarGeometry {
  /** Image sides must divide by this (the networks halve the image four times). */
  size_multiple: number;
  /** The demo scales pictures to this short side (the scale the model was trained at)... */
  short_side: number;
  /** ...with the long side at most this. */
  max_side: number;
  /** Slab thickness in pixels of the image height. */
  thickness_px: number;
  camera_distance: number;
  near_factor: number;
  far_factor: number;
  gradient_view_tilt_deg: number;
  gradient_view_azimuth_deg: number;
}

export interface PlanarManifest {
  schema_version: number;
  kind: "planar";
  /** True for a model whose loop reads the residual backprojection gradient (quality recipe). */
  gradient_input: boolean;
  meta: { iterations: number; budget_scale: number; points_per_pixel_max: number; render_n_max: number };
  geometry: PlanarGeometry;
  /** The cameras the Python runtime builds at one size, to check `camerasFor` against. */
  reference_cameras: { size: [number, number]; frontal: PlanarCamera; gradient_view: PlanarCamera };
  graphs: Record<GraphName, GraphEntry>;
  placement: { kind: "emv"; seed: number };
  sampler: { oversample: number; merge_rounds: number; per_round_fraction: number; knn_k: number; lloyd_iters: number; tile_cap: number };
  final: { tie_break_rel_step: number };
}

export interface PlanarBundle {
  manifest: PlanarManifest;
  models: Map<GraphName, Uint8Array>;
}

export class PlanarBundleError extends Error {
  override name = "PlanarBundleError";
}

/** Fetch a file, reporting bytes as they arrive (`expected` when the server sends no length). */
export async function fetchBytes(url: string, onBytes?: (received: number) => void, fetchImpl: typeof fetch = fetch): Promise<Uint8Array> {
  const r = await fetchImpl(url);
  if (!r.ok) throw new PlanarBundleError(`cannot fetch ${url}: ${r.status}`);
  if (!r.body || !onBytes) {
    const bytes = new Uint8Array(await r.arrayBuffer());
    onBytes?.(bytes.byteLength);
    return bytes;
  }
  const reader = r.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    onBytes(received);
  }
  const out = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) { out.set(c, offset); offset += c.byteLength; }
  return out;
}

/** Fetch the manifest and every graph, checking size and SHA-256. `onProgress` gets loaded/total bytes. */
export async function loadPlanarBundle(
  bundleUrl: string,
  onProgress?: (loaded: number, total: number) => void,
  fetchImpl: typeof fetch = fetch,
): Promise<PlanarBundle> {
  const base = bundleUrl.endsWith("/") ? bundleUrl : bundleUrl + "/";
  const res = await fetchImpl(base + "planar_bundle.json");
  if (!res.ok) throw new PlanarBundleError(`cannot fetch ${base}planar_bundle.json: ${res.status}`);
  const manifest = (await res.json()) as PlanarManifest;
  if (manifest.schema_version !== 2 || manifest.kind !== "planar") {
    throw new PlanarBundleError(`unsupported bundle (schema ${manifest.schema_version}, kind ${manifest.kind})`);
  }
  const total = GRAPH_NAMES.reduce((s, g) => s + manifest.graphs[g].bytes, 0);
  const received = new Map<GraphName, number>();
  const report = () => onProgress?.([...received.values()].reduce((a, b) => a + b, 0), total);
  const models = new Map<GraphName, Uint8Array>();
  await Promise.all(
    GRAPH_NAMES.map(async (name) => {
      const entry = manifest.graphs[name];
      if (!entry) throw new PlanarBundleError(`bundle has no ${name} graph`);
      const bytes = await fetchBytes(base + entry.path, (n) => { received.set(name, n); report(); }, fetchImpl);
      if (bytes.byteLength !== entry.bytes) throw new PlanarBundleError(`${entry.path}: size ${bytes.byteLength} != ${entry.bytes}`);
      if ((await internals.sha256Hex(bytes)) !== entry.sha256) throw new PlanarBundleError(`${entry.path}: sha256 mismatch`);
      models.set(name, bytes);
    }),
  );
  return { manifest, models };
}

/** `RuntimeMeta.point_count` at height x width: the scaled rate sum, capped per pixel and by render_n_max. */
export function pointCount(manifest: PlanarManifest, rateSum: number, height: number, width: number): number {
  const m = manifest.meta;
  const cap = Math.max(1, Math.trunc(m.points_per_pixel_max * height * width));
  return Math.max(1, Math.min(cap, Math.trunc(m.render_n_max), Math.floor(rateSum)));
}

const f32 = Math.fround;

/** The frontal camera: its render reproduces the image pixel for pixel (focal = distance x height). */
export function frontalCamera(g: PlanarGeometry, height: number, width: number): PlanarCamera {
  const d = g.camera_distance;
  const f = f32(d * height);
  return {
    w2c: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, f32(d), 0, 0, 0, 1],
    position: [-0, -0, f32(-d)],
    width, height,
    focal_x: f, focal_y: f,
    center_x: 0.5 * width, center_y: 0.5 * height,
    near: g.near_factor * d, far: g.far_factor * d,
  };
}

type Vec3 = [number, number, number];
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);
const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];

/** The frontal camera orbited about the image centre: tilt from the plane normal, azimuth in the plane. */
export function orbitCamera(frontal: PlanarCamera, tiltDeg: number, azimuthDeg: number): PlanarCamera {
  const d = Math.abs(frontal.position[2]);
  const th = (tiltDeg * Math.PI) / 180, ph = (azimuthDeg * Math.PI) / 180;
  const position: Vec3 = [d * Math.sin(th) * Math.cos(ph), d * Math.sin(th) * Math.sin(ph), -d * Math.cos(th)];
  // look_at_w2c: camera +y (down) as close to world +y as the view allows.
  let fwd: Vec3 = scale(position, -1);
  fwd = scale(fwd, 1 / Math.max(norm(fwd), 1e-12));
  let right = cross([0, 1, 0], fwd);
  if (norm(right) < 1e-6) right = cross([0, 0, 1], fwd);
  right = scale(right, 1 / Math.max(norm(right), 1e-12));
  const down = cross(fwd, right);
  const rows: Vec3[] = [right, down, fwd];
  const t = rows.map((r) => -(r[0] * position[0] + r[1] * position[1] + r[2] * position[2]));
  const w2c = [...rows[0], t[0], ...rows[1], t[1], ...rows[2], t[2], 0, 0, 0, 1].map(f32);
  return { ...frontal, w2c, position: position.map(f32) };
}

/** Both cameras of the loop for an image of height x width. */
export function camerasFor(manifest: PlanarManifest, height: number, width: number): { frontal: PlanarCamera; gradientView: PlanarCamera } {
  const g = manifest.geometry;
  const frontal = frontalCamera(g, height, width);
  return { frontal, gradientView: orbitCamera(frontal, g.gradient_view_tilt_deg, g.gradient_view_azimuth_deg) };
}

/** Half the slab thickness in world units (image height 1) at this image height. */
export function halfThickness(manifest: PlanarManifest, height: number): number {
  return (0.5 * manifest.geometry.thickness_px) / height;
}
