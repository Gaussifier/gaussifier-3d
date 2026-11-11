/** CPU-side pieces of the runtime contract: the final depth-tie break and the frontal PSNR. */

/**
 * `tie_free_means` in f32: points with exactly equal depth get distinct depths, each moved inward
 * by its rank within the tie times `relStep * halfThickness` (the first of a run keeps its depth).
 * Ranks follow a stable sort by depth, as `torch.argsort(stable=True)` does. Modifies `means` and
 * returns how many points moved.
 */
export function tieFreeMeans(means: Float32Array, halfThickness: number, relStep = 1e-6): number {
  const n = means.length / 3;
  const z = new Float32Array(n);
  for (let i = 0; i < n; i++) z[i] = means[3 * i + 2];
  // Array.prototype.sort is stable, so ties keep index order like the stable torch argsort.
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => z[a] - z[b]);
  const step = Math.fround(relStep * halfThickness);
  let moved = 0;
  let runStart = 0;
  for (let j = 0; j < n; j++) {
    const zj = z[order[j]];
    if (j === 0 || zj !== z[order[j - 1]]) runStart = j;
    const rank = j - runStart;
    if (rank === 0) continue;
    const inward = zj >= 0 ? -1 : 1;
    means[3 * order[j] + 2] = Math.fround(zj + Math.fround(inward * rank * step));
    moved++;
  }
  return moved;
}

/** PSNR of a CHW render (clamped to [0, 1]) against a CHW image, as the runtime reports it. */
export function psnr(render: Float32Array, image: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < image.length; i++) {
    const d = Math.min(1, Math.max(0, render[i])) - image[i];
    sum += d * d;
  }
  return -10 * Math.log10(Math.max(sum / image.length, 1e-12));
}
