/**
 * Parity tests against the CUDA sampler.
 *
 * Fixture provenance (web/fixtures/refs, gitignored .bin files):
 *   density = GaussifierSampler.bundled(device="cuda").model.forward_map(img)["density"][0, 0]
 *   native.diffuse_density_to_points_cuda(density.reshape(-1), N, 512, 512, 12345)
 * for img = tests/fixtures/bench_input_512.png and N in {126194, 22500}.
 */
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { createOversampler } from "../src/index.js";

const here = fileURLToPath(new URL(".", import.meta.url));
const fixtures = new URL("../../../fixtures/", import.meta.url);
const spike = (name: string) => fileURLToPath(new URL(`refs/${name}`, fixtures));
const moduleUrl = new URL("../prebuilt/oversampler.mjs", import.meta.url).href;

function readF32(path: string): Float32Array {
  const buf = readFileSync(path);
  return new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
}

/** Compare two point sets as sorted lists of exact float pairs. Returns the number of points of `a` missing from `b`. */
function missingFrom(a: Float32Array, b: Float32Array): number {
  const key = (arr: Float32Array, i: number) => `${arr[2 * i]},${arr[2 * i + 1]}`;
  const set = new Set<string>();
  for (let i = 0; i < b.length / 2; i++) set.add(key(b, i));
  let missing = 0;
  for (let i = 0; i < a.length / 2; i++) if (!set.has(key(a, i))) missing++;
  return missing;
}

const haveSpike = ["density_512.bin", "cuda_points_126194.bin", "cuda_points_22500.bin"].every((f) => existsSync(spike(f)));

describe("wasm oversampler", () => {
  it.skipIf(!existsSync(fileURLToPath(moduleUrl)))("loads the glue module next to dist", async () => {
    const s = await createOversampler({ moduleUrl });
    const out = s.sample(new Float32Array(16).fill(1), 4, 4, 3, 0);
    expect(out.length).toBe(6);
    s.dispose();
    expect(() => s.sample(new Float32Array(16), 4, 4, 1, 0)).toThrow();
  });

  it("validates inputs", async () => {
    const s = await createOversampler({ moduleUrl });
    expect(() => s.sample(new Float32Array(15), 4, 4, 3, 0)).toThrow(RangeError);
    expect(() => s.sample(new Float32Array(16), 4, 4, 0, 0)).toThrow(RangeError);
    expect(() => s.sample(new Float32Array(16), 4, 4, 3, -1)).toThrow(RangeError);
    expect(() => s.sample(new Float32Array(16), 4, 4, 3, 2 ** 31)).toThrow(RangeError);
    expect(() => s.sample(new Float32Array(16), 4, 4, 3, 1.5)).toThrow(RangeError);
  });

  it.skipIf(!haveSpike)("matches the CUDA sampler bit-exactly at 22500 and within one point at 126194", async () => {
    const s = await createOversampler({ moduleUrl });
    const density = readF32(spike("density_512.bin"));
    expect(density.length).toBe(512 * 512);
    const results: Record<string, unknown> = {};
    for (const [count, maxMissing] of [[22500, 0], [126194, 1]] as const) {
      const ref = readF32(spike(`cuda_points_${count}.bin`));
      let best = Infinity;
      let out = new Float32Array(0);
      for (let rep = 0; rep < 5; rep++) {
        const t0 = performance.now();
        out = s.sample(density, 512, 512, count, 12345);
        best = Math.min(best, performance.now() - t0);
      }
      expect(out.length).toBe(count * 2);
      for (let i = 0; i < out.length; i++) {
        expect(out[i]).toBeGreaterThanOrEqual(0);
        expect(out[i]).toBeLessThan(1);
      }
      const missing = missingFrom(ref, out);
      results[count] = { missing, best_ms: +best.toFixed(2) };
      expect(missing).toBeLessThanOrEqual(maxMissing);
      // Determinism across calls.
      const again = s.sample(density, 512, 512, count, 12345);
      expect(missingFrom(out, again)).toBe(0);
    }
    console.log("oversampler parity vs CUDA:", JSON.stringify(results));
  });

  it.skipIf(!haveSpike)("applies the tile cap only where the uncapped tile exceeds it", async () => {
    const s = await createOversampler({ moduleUrl });
    const density = readF32(spike("density_512.bin"));
    // count 500 at 512x512: mean spacing 22.9 px -> tile 92 uncapped, 32 capped.
    const capped = s.sample(density, 512, 512, 500, 12345, 32);
    const uncapped = s.sample(density, 512, 512, 500, 12345, 0);
    expect(missingFrom(capped, uncapped)).toBeGreaterThan(0);
    // count 22500: tile 14 in both cases -> identical.
    const a = s.sample(density, 512, 512, 22500, 12345, 32);
    const b = s.sample(density, 512, 512, 22500, 12345, 0);
    expect(missingFrom(a, b)).toBe(0);
  });

  it("matches the bench_128 golden oversample when present", async () => {
    const dir = fileURLToPath(new URL("bench_128/", fixtures));
    const shapesPath = `${dir}shapes.json`;
    if (!existsSync(`${dir}oversample.bin`) || !existsSync(`${dir}density.bin`) || !existsSync(shapesPath)) {
      console.log("bench_128 goldens absent; skipping");
      return;
    }
    const shapes = JSON.parse(readFileSync(shapesPath, "utf8")) as Record<string, unknown>;
    const meta = (shapes.meta ?? shapes) as Record<string, unknown>;
    const width = Number(meta.width ?? (shapes.density as number[] | undefined)?.[1] ?? 128);
    const height = Number(meta.height ?? (shapes.density as number[] | undefined)?.[0] ?? 128);
    const seed = Number(meta.seed ?? 12345);
    const density = readF32(`${dir}density.bin`);
    const ref = readF32(`${dir}oversample.bin`);
    const count = ref.length / 2;
    const s = await createOversampler({ moduleUrl });
    const out = s.sample(density, width, height, count, seed);
    const missing = missingFrom(ref, out);
    console.log(`bench_128 oversample parity: ${missing} of ${count} points differ`);
    expect(missing).toBeLessThanOrEqual(1);
  });
});
