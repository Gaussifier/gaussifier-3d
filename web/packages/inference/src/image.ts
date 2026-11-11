import type { PlanarImage } from "./types.js";

export interface PreparedInput {
  /** Planar f32 [3, H, W] in [0,1], edge padded. */
  planar: Float32Array;
  width: number;
  height: number;
  crop: { w: number; h: number };
}

function toPlanar(image: ImageBitmap | ImageData | PlanarImage): PlanarImage {
  if (image instanceof Float32Array || (typeof (image as PlanarImage).data !== "undefined" && (image as PlanarImage).data instanceof Float32Array)) {
    const p = image as PlanarImage;
    if (p.data.length !== 3 * p.width * p.height) throw new Error("PlanarImage data length must be 3*width*height");
    return p;
  }
  let imageData: ImageData;
  if (typeof ImageData !== "undefined" && image instanceof ImageData) {
    imageData = image;
  } else {
    const bmp = image as ImageBitmap;
    const canvas = new OffscreenCanvas(bmp.width, bmp.height);
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2d context unavailable");
    ctx.drawImage(bmp, 0, 0);
    imageData = ctx.getImageData(0, 0, bmp.width, bmp.height);
  }
  const { width, height, data } = imageData;
  const hw = width * height;
  const planar = new Float32Array(3 * hw);
  for (let i = 0; i < hw; i++) {
    planar[i] = data[4 * i] / 255;
    planar[hw + i] = data[4 * i + 1] / 255;
    planar[2 * hw + i] = data[4 * i + 2] / 255;
  }
  return { data: planar, width, height };
}

/** Convert to planar float32 and edge-replicate pad to multiples of `padMultiple`. */
export function prepareInput(image: ImageBitmap | ImageData | PlanarImage, padMultiple = 32): PreparedInput {
  const src = toPlanar(image);
  const w0 = src.width, h0 = src.height;
  const W = Math.ceil(w0 / padMultiple) * padMultiple;
  const H = Math.ceil(h0 / padMultiple) * padMultiple;
  // Output of an earlier prepareInput (padded and clamped): use it as is instead of copying 3*W*H floats.
  if (src.prepared === true && W === w0 && H === h0) {
    return { planar: src.data, width: W, height: H, crop: { w: w0, h: h0 } };
  }
  const out = new Float32Array(3 * W * H);
  for (let c = 0; c < 3; c++) {
    const sBase = c * w0 * h0, dBase = c * W * H;
    for (let y = 0; y < H; y++) {
      const sy = Math.min(y, h0 - 1);
      const sRow = sBase + sy * w0, dRow = dBase + y * W;
      for (let x = 0; x < w0; x++) {
        let v = src.data[sRow + x];
        if (v < 0) v = 0; else if (v > 1) v = 1;
        out[dRow + x] = v;
      }
      const edge = out[dRow + w0 - 1];
      for (let x = w0; x < W; x++) out[dRow + x] = edge;
    }
  }
  return { planar: out, width: W, height: H, crop: { w: w0, h: h0 } };
}
