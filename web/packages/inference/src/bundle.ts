import { BundleMismatchError, type BundleManifest } from "./types.js";

export interface LoadedBundle {
  manifest: BundleManifest;
  /** Model bytes keyed by `${role}:${shape}`. */
  models: Map<string, Uint8Array>;
}

export async function sha256Hex(data: ArrayBuffer | Uint8Array): Promise<string> {
  const view = data instanceof Uint8Array ? data : new Uint8Array(data);
  const copy = new Uint8Array(view.byteLength);
  copy.set(view);
  const digest = await crypto.subtle.digest("SHA-256", copy.buffer as ArrayBuffer);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

function assertModel(m: BundleManifest): void {
  const model = m.model;
  const problems: string[] = [];
  if (model.correction_predict_xy !== true) problems.push("correction_predict_xy must be true");
  if (model.correction_output_channels !== 8) problems.push("correction_output_channels must be 8");
  if (model.correction_input_channels !== 17) problems.push("correction_input_channels must be 17");
  if (Math.abs(model.correction_xy_step_pixels - 2.56) > 1e-9) problems.push("correction_xy_step_pixels must be 2.56");
  if (model.decoder_density_aware !== false) problems.push("decoder_density_aware must be false");
  if (m.input.pad_multiple !== 32) problems.push("input.pad_multiple must be 32");
  if (problems.length) throw new BundleMismatchError(`web_bundle.json: ${problems.join("; ")}`);
}

/** Fetch web_bundle.json and every model file, verifying SHA-256 and the asserted model values. */
export async function loadBundle(bundleUrl: string, fetchImpl: typeof fetch = fetch): Promise<LoadedBundle> {
  const base = bundleUrl.endsWith("/") ? bundleUrl : bundleUrl + "/";
  const res = await fetchImpl(base + "web_bundle.json");
  if (!res.ok) throw new BundleMismatchError(`cannot fetch ${base}web_bundle.json: ${res.status}`);
  const manifest = (await res.json()) as BundleManifest;
  if (manifest.schema_version !== 1) throw new BundleMismatchError(`unsupported schema_version ${manifest.schema_version}`);
  assertModel(manifest);
  const models = new Map<string, Uint8Array>();
  await Promise.all(
    manifest.files.map(async (f) => {
      if (f.precision !== "f32") return;
      const r = await fetchImpl(base + f.path);
      if (!r.ok) throw new BundleMismatchError(`cannot fetch ${f.path}: ${r.status}`);
      const bytes = new Uint8Array(await r.arrayBuffer());
      if (bytes.byteLength !== f.bytes) throw new BundleMismatchError(`${f.path}: size ${bytes.byteLength} != ${f.bytes}`);
      const hash = await sha256Hex(bytes);
      if (hash !== f.sha256) throw new BundleMismatchError(`${f.path}: sha256 mismatch`);
      models.set(`${f.role}:${f.shape}`, bytes);
    }),
  );
  if (!models.has("forward_map:dynamic") || !models.has("correction_head:dynamic")) {
    throw new BundleMismatchError("bundle must contain dynamic forward_map and correction_head models");
  }
  return { manifest, models };
}
