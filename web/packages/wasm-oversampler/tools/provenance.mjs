// Provenance of the prebuilt wasm: the sha256 of every input that shapes it.
//   node tools/provenance.mjs            print the current input hashes
//   node tools/provenance.mjs --write    write prebuilt/BUILD.json (called by build.sh)
// The unit tests compare BUILD.json with inputHashes() so a source change without a rebuild fails.
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const pkg = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const repo = path.resolve(pkg, "../../..");
export const BUILD_JSON = path.join(pkg, "prebuilt/BUILD.json");
/** Repository-relative inputs, in hash order. */
export const INPUTS = [
  "src/gaussifier_sampler/native_sampler_backend/cpu/csrc/density_points.cpp",
  "web/packages/wasm-oversampler/tools/transform.py",
  "web/packages/wasm-oversampler/build.sh",
];

export function sha256(file) {
  return createHash("sha256").update(fs.readFileSync(file)).digest("hex");
}

export function inputHashes() {
  return Object.fromEntries(INPUTS.map((rel) => [rel, sha256(path.join(repo, rel))]));
}

export function readBuild() {
  return JSON.parse(fs.readFileSync(BUILD_JSON, "utf8"));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const inputs = inputHashes();
  if (process.argv.includes("--write")) {
    const record = {
      emscripten: process.env.EMSCRIPTEN_VERSION ?? "unknown",
      flags: (process.env.EM_FLAGS ?? "").split(/\s+/).filter(Boolean),
      built: new Date().toISOString(),
      inputs,
      outputs: Object.fromEntries(["oversampler.mjs", "oversampler.wasm"].map((f) => [f, sha256(path.join(pkg, "prebuilt", f))])),
    };
    fs.writeFileSync(BUILD_JSON, JSON.stringify(record, null, 2) + "\n");
    console.log(`wrote ${BUILD_JSON}`);
  } else {
    console.log(JSON.stringify(inputs, null, 2));
  }
}
