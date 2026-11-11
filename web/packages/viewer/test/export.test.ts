import { describe, expect, it } from "vitest";
import { sceneToNpz, loadNpz } from "../src/index.js";
import type { Scene } from "../src/types.js";

function scene(): Scene {
  const n = 3, W = 8, H = 4;
  const xy = Float32Array.from([0.1, 0.2, 0.5, 0.5, 0.9, 0.3]);
  const cov = Float32Array.from([0, 0, 0, 0.5, 0.1, -0.2, 1, 0, 1]);
  return { width: W, height: H, crop: { w: 7, h: 4 }, states: [{ xy, cov, scale: new Float32Array(2 * n).fill(1), rotation: new Float32Array(n), color: Float32Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]), rendered: new Float32Array(3 * W * H).fill(0.25), label: "final" }],
    density: new Float32Array(W * H).fill(1), weights: Float32Array.from([1, 1.5, 0.5]), image: new Float32Array(3 * W * H).fill(0.5), meta: { n, seed: 12345 } };
}

describe("sceneToNpz", () => {
  it("round-trips through loadNpz", async () => {
    const s = scene();
    const blob = sceneToNpz(s, { profile: "production", meta: { note: "test" } });
    const back = await loadNpz(await blob.arrayBuffer());
    expect(back.width).toBe(8); expect(back.height).toBe(4);
    expect(back.states.length).toBe(1);
    expect(Array.from(back.states[0].xy)).toEqual(Array.from(s.states[0].xy));
    expect(Array.from(back.states[0].cov!)).toEqual(Array.from(s.states[0].cov!));
    expect(Array.from(back.states[0].color)).toEqual(Array.from(s.states[0].color));
    expect(back.density && Array.from(back.density)).toEqual(Array.from(s.density!));
    expect(back.states[0].rendered && back.states[0].rendered.length).toBe(3 * 8 * 4);
    expect((back.meta as any).note).toBe("test");
    expect((back.meta as any).n).toBe(3);
  });
});

describe("sceneToNpz points key", () => {
  it("carries the first state's positions when the scene keeps the initializer", async () => {
    const { readNpz } = await import("../src/loaders/npz.js");
    const s = scene();
    const init = { ...s.states[0], xy: Float32Array.from([0.11, 0.22, 0.33, 0.44, 0.55, 0.66]), label: "init" };
    s.states = [init, s.states[0]];
    const arrays = await readNpz(await sceneToNpz(s).arrayBuffer());
    expect(Array.from(arrays.get("points")!.data as Float32Array)).toEqual(Array.from(init.xy));
    expect(Array.from(arrays.get("final_points")!.data as Float32Array)).toEqual(Array.from(s.states[1].xy));
  });
});
