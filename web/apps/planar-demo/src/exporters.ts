/**
 * Downloads of the model's Gaussians: a standard 3DGS PLY and a SOG bundle (PlayCanvas's
 * compressed format, written by splat-transform in the browser).
 *
 * Both are written in the frame 3DGS viewers expect of these files ("PLY space"): viewers such as
 * PlayCanvas and SuperSplat turn the data 180 degrees about z when they load it. The model works
 * in OpenCV axes (x right, y down, the camera at z = -1 looking at the image plane z = 0), so the
 * files hold the model frame turned 180 degrees about y, (-x, y, -z): after the viewer's turn the
 * image stands upright and faces the viewer's default camera, exactly as in this page.
 */
import type { PlanarSplats } from "@gaussifier/planar";

const PLY_PROPS = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"];

/** The model frame turned 180 degrees about y: positions (-x, y, -z), quaternions (w,x,y,z) -> (-y, z, w, -x). */
export function toFileFrame(s: PlanarSplats): PlanarSplats {
  const means = new Float32Array(s.means.length);
  const quats = new Float32Array(s.quats.length);
  for (let i = 0; i < s.n; i++) {
    means[3 * i] = -s.means[3 * i];
    means[3 * i + 1] = s.means[3 * i + 1];
    means[3 * i + 2] = -s.means[3 * i + 2];
    const w = s.quats[4 * i], x = s.quats[4 * i + 1], y = s.quats[4 * i + 2], z = s.quats[4 * i + 3];
    quats[4 * i] = -y;
    quats[4 * i + 1] = z;
    quats[4 * i + 2] = w;
    quats[4 * i + 3] = -x;
  }
  return { ...s, means, quats };
}

/** Every Gaussian as a small opaque dot at its centre (the Centers view). */
export function centersOf(s: PlanarSplats, radius = 0.8 / 512): PlanarSplats {
  const logScales = new Float32Array(3 * s.n).fill(Math.log(radius));
  const quats = new Float32Array(4 * s.n);
  for (let i = 0; i < s.n; i++) quats[4 * i] = 1;
  return { ...s, logScales, quats, opacityLogits: new Float32Array(s.n).fill(5) };
}

/** Binary little-endian PLY with the 3DGS vertex properties (SH degree 0), in the file frame. */
export function splatsToPly(splats: PlanarSplats): ArrayBuffer {
  const s = toFileFrame(splats);
  const header =
    "ply\nformat binary_little_endian 1.0\n" +
    `element vertex ${s.n}\n` +
    PLY_PROPS.map((p) => `property float ${p}\n`).join("") +
    "end_header\n";
  const head = new TextEncoder().encode(header);
  const stride = PLY_PROPS.length;
  const body = new Float32Array(s.n * stride);
  for (let i = 0; i < s.n; i++) {
    const o = i * stride;
    body[o] = s.means[3 * i]; body[o + 1] = s.means[3 * i + 1]; body[o + 2] = s.means[3 * i + 2];
    body[o + 3] = s.sh0[3 * i]; body[o + 4] = s.sh0[3 * i + 1]; body[o + 5] = s.sh0[3 * i + 2];
    body[o + 6] = s.opacityLogits[i];
    body[o + 7] = s.logScales[3 * i]; body[o + 8] = s.logScales[3 * i + 1]; body[o + 9] = s.logScales[3 * i + 2];
    body[o + 10] = s.quats[4 * i]; body[o + 11] = s.quats[4 * i + 1]; body[o + 12] = s.quats[4 * i + 2]; body[o + 13] = s.quats[4 * i + 3];
  }
  const out = new Uint8Array(head.byteLength + body.byteLength);
  out.set(head, 0);
  out.set(new Uint8Array(body.buffer), head.byteLength);
  return out.buffer;
}

/** A bundled `.sog` (meta.json plus WebP textures in one zip), in the file frame. */
export async function splatsToSog(splats: PlanarSplats): Promise<Uint8Array> {
  const st = await import("@playcanvas/splat-transform");
  const { default: webpUrl } = await import("@playcanvas/splat-transform/lib/webp.wasm?url");
  st.WebPCodec.wasmUrl = new URL(webpUrl, location.href).href;
  const s = toFileFrame(splats);
  const n = s.n;
  const column = (name: string, src: Float32Array, stride: number, offset: number) => {
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) data[i] = src[i * stride + offset];
    return new st.Column(name, data);
  };
  // Labelled as PLY space: the data already is, so the writer stores it unchanged.
  const table = new st.DataTable([
    column("x", s.means, 3, 0), column("y", s.means, 3, 1), column("z", s.means, 3, 2),
    column("scale_0", s.logScales, 3, 0), column("scale_1", s.logScales, 3, 1), column("scale_2", s.logScales, 3, 2),
    column("rot_0", s.quats, 4, 0), column("rot_1", s.quats, 4, 1), column("rot_2", s.quats, 4, 2), column("rot_3", s.quats, 4, 3),
    column("opacity", s.opacityLogits, 1, 0),
    column("f_dc_0", s.sh0, 3, 0), column("f_dc_1", s.sh0, 3, 1), column("f_dc_2", s.sh0, 3, 2),
  ], st.Transform.PLY);
  const fs = new st.MemoryFileSystem();
  // splat-transform 3.10 writes SOG version 2, which every PlayCanvas engine and SuperSplat reads.
  await st.writeSog({ dataTable: table, filename: "scene.sog", bundle: true, iterations: 10, logging: "silent" }, fs);
  const out = fs.results.get("scene.sog");
  if (!out) throw new Error("splat-transform wrote no scene.sog");
  return out;
}

export function download(data: ArrayBuffer | Uint8Array, filename: string): void {
  const url = URL.createObjectURL(new Blob([data as BlobPart], { type: "application/octet-stream" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}
