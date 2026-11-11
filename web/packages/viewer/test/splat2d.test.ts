import { describe, expect, it } from "vitest";
import fs from "node:fs";
import { sceneToSplat2d } from "../src/exporters/splat2d.js";
import { loadSplat2d } from "../src/loaders/splat2d.js";
import type { Scene } from "../src/types.js";

// Optional external export (not tracked); the test skips when it is absent.
const SPLAT2D_SAMPLE = new URL("../../../fixtures/refs/kodim01.splat2d", import.meta.url);

describe("splat2d", () => {
  it("round-trips xy, sigma, rotation, and color", async () => {
    const n = 3;
    const scene: Scene = { width: 64, height: 32, states: [{ xy: Float32Array.from([0.1, 0.2, 0.5, 0.5, 0.9, 0.3]), scale: Float32Array.from([1.5, 2.5, 0.5, 0.25, 4, 0.005]), rotation: Float32Array.from([0, 0.3, -1.2]), color: Float32Array.from([1, 0, 0, 0, 1, 0, 0.2, 0.3, 0.4]) }] };
    const blob = sceneToSplat2d(scene);
    expect(blob.size).toBe(12 + 32 * n);
    const back = loadSplat2d(await blob.arrayBuffer());
    expect(back.width).toBe(64); expect(back.height).toBe(32);
    const s = back.states[0];
    expect(Array.from(s.xy)).toEqual(Array.from(scene.states[0].xy));
    expect(Array.from(s.rotation)).toEqual(Array.from(scene.states[0].rotation));
    expect(Array.from(s.color)).toEqual(Array.from(scene.states[0].color));
    // sigma survives the reciprocal storage except below the 0.01 floor
    const want = [1.5, 2.5, 0.5, 0.25, 4, 0.01];
    for (let i = 0; i < 6; i++) expect(s.scale[i]).toBeCloseTo(want[i], 5);
  });
  it("stores the reciprocal sigma like Splat2D", async () => {
    const scene: Scene = { width: 8, height: 8, states: [{ xy: Float32Array.from([0.5, 0.5]), scale: Float32Array.from([4, 0.5]), rotation: Float32Array.from([0]), color: Float32Array.from([1, 1, 1]) }] };
    const raw = new Float32Array((await sceneToSplat2d(scene).arrayBuffer()).slice(12 + 8, 12 + 16));
    expect(raw[0]).toBeCloseTo(0.25, 6); expect(raw[1]).toBeCloseTo(2, 6);
  });
  it("parses a real Splat2D export", () => {
    if (!fs.existsSync(SPLAT2D_SAMPLE)) return;
    const buf = fs.readFileSync(SPLAT2D_SAMPLE);
    const scene = loadSplat2d(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
    expect(scene.width).toBe(768); expect(scene.height).toBe(512);
    expect(scene.states[0].xy.length / 2).toBe(53500);
    const s = scene.states[0];
    let inRange = 0; for (let i = 0; i < s.xy.length; i++) if (s.xy[i] >= 0 && s.xy[i] <= 1) inRange++;
    expect(inRange / s.xy.length).toBeGreaterThan(0.99);
    for (let i = 0; i < s.scale.length; i++) { expect(s.scale[i]).toBeGreaterThan(0); expect(Number.isFinite(s.scale[i])).toBe(true); }
  });
});

describe("splat2d crop", () => {
  const scene = (): Scene => ({ width: 64, height: 32, crop: { w: 50, h: 30 }, states: [{ xy: Float32Array.from([0.5, 0.5, 1, 1]), scale: Float32Array.from([1, 1, 2, 2]), rotation: Float32Array.from([0, 0]), color: Float32Array.from([1, 1, 1, 0, 0, 0]) }] });
  it("writes the original image size and keeps every pixel position", async () => {
    const back = loadSplat2d(await sceneToSplat2d(scene()).arrayBuffer());
    expect(back.width).toBe(50); expect(back.height).toBe(30);
    expect(back.states[0].xy[0]).toBeCloseTo(32 / 50, 6);
    expect(back.states[0].xy[1]).toBeCloseTo(16 / 30, 6);
    // a center in the padding lands outside [0, 1]
    expect(back.states[0].xy[2]).toBeCloseTo(64 / 50, 6);
    expect(back.states[0].xy[3]).toBeCloseTo(32 / 30, 6);
  });
  it("can write the padded frame instead", async () => {
    const back = loadSplat2d(await sceneToSplat2d(scene(), { frame: "padded" }).arrayBuffer());
    expect(back.width).toBe(64); expect(back.height).toBe(32);
    expect(back.states[0].xy[0]).toBeCloseTo(0.5, 6);
  });
});
