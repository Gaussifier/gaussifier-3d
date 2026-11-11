/** The committed wasm must come from the current sources: build.sh records their hashes in BUILD.json. */
import { existsSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { BUILD_JSON, inputHashes, readBuild } from "../tools/provenance.mjs";

describe("prebuilt oversampler", () => {
  it("ships next to its provenance record", () => {
    expect(existsSync(BUILD_JSON)).toBe(true);
    const build = readBuild();
    expect(build.outputs["oversampler.wasm"]).toMatch(/^[0-9a-f]{64}$/);
    expect(build.emscripten).toMatch(/^\d+\.\d+\.\d+$/);
  });
  it("was built from the current sources (otherwise run `EMSDK=... npm run build:wasm`)", () => {
    expect(readBuild().inputs).toEqual(inputHashes());
  });
});
