import { describe, expect, it } from "vitest";
import { buildOwnerMap } from "../src/voronoi.js";

function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
}

/** O(N * pixels) nearest center, ties to the lower index. */
function bruteOwner(xy: Float32Array, width: number, height: number): Int32Array {
  const n = xy.length / 2;
  const out = new Int32Array(width * height).fill(-1);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    let best = Infinity, idx = -1;
    for (let i = 0; i < n; i++) {
      const dx = xy[2 * i] * width - x, dy = xy[2 * i + 1] * height - y;
      const d = dx * dx + dy * dy;
      if (d < best) { best = d; idx = i; }
    }
    out[y * width + x] = idx;
  }
  return out;
}

describe("buildOwnerMap", () => {
  it("matches the brute-force nearest center, with centers outside the frame too", () => {
    const rnd = lcg(7);
    for (const [w, h, n] of [[40, 30, 25], [64, 64, 300], [17, 5, 3], [8, 8, 1], [9, 200, 40]]) {
      const xy = new Float32Array(2 * n);
      for (let i = 0; i < 2 * n; i++) xy[i] = rnd() * 1.2 - 0.1;
      expect(Array.from(buildOwnerMap(xy, w, h)), `${w}x${h} n=${n}`).toEqual(Array.from(bruteOwner(xy, w, h)));
    }
  });
  it("is -1 everywhere without centers", () => {
    expect(Array.from(buildOwnerMap(new Float32Array(0), 3, 2))).toEqual([-1, -1, -1, -1, -1, -1]);
  });
});

describe("buildOwnerMap bisector", () => {
  it("splits two centers at their bisector", () => {
    const w = 10, h = 4;
    // centers at pixels 2 and 7: pixels 0..4 belong to the first, 5..9 to the second
    const owner = buildOwnerMap(Float32Array.from([0.2, 0.5, 0.7, 0.5]), w, h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) expect(owner[y * w + x], `x=${x}`).toBe(x < 5 ? 0 : 1);
  });
});
