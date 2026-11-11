/** Nearest-center map for the Voronoi overlay, computed on the CPU once per state. */
/**
 * Index of the nearest center for every source pixel, in the renderer convention
 * (pixel j sits at position j, a center at xy * size). Exact: a uniform grid holds
 * about two centers per cell and the ring search stops once no unvisited cell can
 * hold a closer center. Ties go to the lower index; -1 when there are no centers.
 */
export function buildOwnerMap(xy: Float32Array, width: number, height: number): Int32Array {
  const n = xy.length >> 1;
  const owner = new Int32Array(width * height).fill(-1);
  if (n === 0 || width <= 0 || height <= 0) return owner;
  const cell = Math.max(1, Math.sqrt((2 * width * height) / n));
  const gw = Math.max(1, Math.ceil(width / cell));
  const gh = Math.max(1, Math.ceil(height / cell));
  // Counting sort of the centers into cells. Centers outside the frame land in the edge
  // cells, which only makes them farther away than the ring bound assumes.
  const starts = new Int32Array(gw * gh + 1);
  const cellOf = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const cx = Math.min(gw - 1, Math.max(0, Math.floor((xy[2 * i] * width) / cell)));
    const cy = Math.min(gh - 1, Math.max(0, Math.floor((xy[2 * i + 1] * height) / cell)));
    const c = cy * gw + cx;
    cellOf[i] = c;
    starts[c + 1]++;
  }
  for (let c = 0; c < gw * gh; c++) starts[c + 1] += starts[c];
  const items = new Int32Array(n);
  const fill = starts.slice(0, gw * gh);
  for (let i = 0; i < n; i++) items[fill[cellOf[i]]++] = i;
  const maxRing = Math.max(gw, gh);
  for (let py = 0; py < height; py++) {
    const cy = Math.min(gh - 1, Math.floor(py / cell));
    for (let px = 0; px < width; px++) {
      const cx = Math.min(gw - 1, Math.floor(px / cell));
      let best = Infinity;
      let bestIdx = -1;
      for (let r = 0; r <= maxRing; r++) {
        const y0 = cy - r, y1 = cy + r, x0 = cx - r, x1 = cx + r;
        for (let gy = Math.max(0, y0); gy <= Math.min(gh - 1, y1); gy++) {
          // Whole rows at the top and bottom of the ring, only the two end columns in between.
          const stepX = gy === y0 || gy === y1 ? 1 : x1 - x0;
          for (let gx = x0; gx <= x1; gx += stepX) {
            if (gx < 0 || gx >= gw) continue;
            const c = gy * gw + gx;
            for (let k = starts[c]; k < starts[c + 1]; k++) {
              const i = items[k];
              const dx = xy[2 * i] * width - px;
              const dy = xy[2 * i + 1] * height - py;
              const d = dx * dx + dy * dy;
              if (d < best || (d === best && i < bestIdx)) { best = d; bestIdx = i; }
            }
          }
        }
        // Every center in ring r + 1 or beyond is more than r * cell away.
        const bound = r * cell;
        if (bestIdx >= 0 && best <= bound * bound) break;
      }
      owner[py * width + px] = bestIdx;
    }
  }
  return owner;
}
