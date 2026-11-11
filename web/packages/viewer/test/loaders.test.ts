import { describe, expect, it } from "vitest";
import { deflateRawSync } from "node:zlib";
import { covToScaleRotation, projectGaussian } from "../src/math.js";
import { fromRunResult, loadNpz, loadNpyTriple } from "../src/loaders/index.js";
import { parseNpy, npyToFloat32 } from "../src/loaders/npy.js";

function crc32(bytes: Uint8Array): number {
  let c = ~0;
  for (let i = 0; i < bytes.length; i++) {
    c ^= bytes[i];
    for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
  }
  return ~c >>> 0;
}

/** Minimal NPY writer covering versions 1 and 2 (3 shares the v2 layout with utf8 headers). */
function npy(data: Float32Array | Float64Array | Int32Array | BigInt64Array | Uint8Array, shape: number[], version = 1): Uint8Array {
  const descr = data instanceof Float32Array ? "<f4" : data instanceof Float64Array ? "<f8" : data instanceof Int32Array ? "<i4" : data instanceof BigInt64Array ? "<i8" : "|u1";
  const shapeStr = shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(", ")})`;
  let header = `{'descr': '${descr}', 'fortran_order': False, 'shape': ${shapeStr}, }`;
  const pre = version === 1 ? 10 : 12;
  const pad = 64 - ((pre + header.length + 1) % 64);
  header += " ".repeat(pad) + "\n";
  const out = new Uint8Array(pre + header.length + data.byteLength);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, version, 0]);
  const dv = new DataView(out.buffer);
  if (version === 1) dv.setUint16(8, header.length, true); else dv.setUint32(8, header.length, true);
  for (let i = 0; i < header.length; i++) out[pre + i] = header.charCodeAt(i);
  out.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), pre + header.length);
  return out;
}

/** Hand-written zip writer: stored or deflated entries, central directory, EOCD. */
function zip(entries: Record<string, Uint8Array>, deflate: boolean): ArrayBuffer {
  const parts: Uint8Array[] = [];
  const central: Uint8Array[] = [];
  let offset = 0;
  for (const [name, raw] of Object.entries(entries)) {
    const data = deflate ? new Uint8Array(deflateRawSync(raw)) : raw;
    const nameBytes = new TextEncoder().encode(name);
    const local = new Uint8Array(30 + nameBytes.length);
    const ldv = new DataView(local.buffer);
    ldv.setUint32(0, 0x04034b50, true); ldv.setUint16(4, 20, true); ldv.setUint16(8, deflate ? 8 : 0, true);
    ldv.setUint32(14, crc32(raw), true); ldv.setUint32(18, data.length, true); ldv.setUint32(22, raw.length, true);
    ldv.setUint16(26, nameBytes.length, true);
    local.set(nameBytes, 30);
    const cd = new Uint8Array(46 + nameBytes.length);
    const cdv = new DataView(cd.buffer);
    cdv.setUint32(0, 0x02014b50, true); cdv.setUint16(4, 20, true); cdv.setUint16(6, 20, true); cdv.setUint16(10, deflate ? 8 : 0, true);
    cdv.setUint32(16, crc32(raw), true); cdv.setUint32(20, data.length, true); cdv.setUint32(24, raw.length, true);
    cdv.setUint16(28, nameBytes.length, true); cdv.setUint32(42, offset, true);
    cd.set(nameBytes, 46);
    parts.push(local, data);
    central.push(cd);
    offset += local.length + data.length;
  }
  const cdStart = offset;
  let cdSize = 0;
  for (const c of central) { parts.push(c); cdSize += c.length; }
  const eocd = new Uint8Array(22);
  const edv = new DataView(eocd.buffer);
  edv.setUint32(0, 0x06054b50, true); edv.setUint16(8, central.length, true); edv.setUint16(10, central.length, true);
  edv.setUint32(12, cdSize, true); edv.setUint32(16, cdStart, true);
  parts.push(eocd);
  const total = parts.reduce((a, p) => a + p.length, 0);
  const out = new Uint8Array(total);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out.buffer;
}

describe("npy", () => {
  it("parses v1 and v2 headers and dtypes", () => {
    const f = new Float32Array([1.5, -2, 3]);
    for (const v of [1, 2]) {
      const arr = parseNpy(npy(f, [3], v));
      expect(arr.dtype).toBe("<f4");
      expect(arr.shape).toEqual([3]);
      expect(Array.from(arr.data as Float32Array)).toEqual([1.5, -2, 3]);
    }
    const d = parseNpy(npy(new Float64Array([0.25]), [], 1));
    expect(d.shape).toEqual([]);
    expect(npyToFloat32(d)[0]).toBe(0.25);
    const i = parseNpy(npy(new BigInt64Array([12345n]), [1], 2));
    expect(npyToFloat32(i)[0]).toBe(12345);
    const u = parseNpy(npy(new Uint8Array([104, 105]), [2], 1));
    expect(new TextDecoder().decode(u.data as Uint8Array)).toBe("hi");
  });
  it("rejects fortran order", () => {
    const patched = npy(new Float32Array([1]), [1], 1);
    const needle = new TextEncoder().encode("'fortran_order': False");
    const replacement = new TextEncoder().encode("'fortran_order': True ");
    outer: for (let i = 0; i < patched.length - needle.length; i++) {
      for (let j = 0; j < needle.length; j++) if (patched[i + j] !== needle[j]) continue outer;
      patched.set(replacement, i);
      break;
    }
    expect(() => parseNpy(patched)).toThrow(/Fortran/);
  });
});

describe("math", () => {
  it("derives scale and rotation from log-covariance like model.py", () => {
    const cov = new Float32Array([Math.log(4), 0, Math.log(1), 0, 0.3, 0]);
    const { scale, rotation } = covToScaleRotation(cov, 2);
    expect(scale[0]).toBeCloseTo(2, 5);
    expect(scale[1]).toBeCloseTo(1, 5);
    expect(rotation[0]).toBeCloseTo(-0.5 * Math.atan2(0, Math.log(4)), 6);
    const a = 0, b = 0.3, c = 0;
    const center = 0.5 * (a + c), half = 0.5 * (a - c), radius = Math.sqrt(half * half + b * b + 1e-12);
    expect(scale[2]).toBeCloseTo(Math.exp(0.5 * (center + radius)), 6);
    expect(scale[3]).toBeCloseTo(Math.exp(0.5 * (center - radius)), 6);
    expect(rotation[1]).toBeCloseTo(-0.5 * Math.atan2(0.6, 0), 6);
  });
  it("projects with the production rotation convention", () => {
    const p = projectGaussian(0.5, 0.25, 2, 1, Math.PI / 2, 100, 50);
    expect(p.cx).toBeCloseTo(50);
    expect(p.cy).toBeCloseTo(12.5);
    expect(p.sxx).toBeCloseTo(1, 5);
    expect(p.syy).toBeCloseTo(4, 5);
    expect(p.cxx).toBeCloseTo(1, 5);
    expect(p.cyy).toBeCloseTo(0.25, 5);
    expect(p.valid).toBe(true);
  });
});

describe("npz scenes", () => {
  const cliEntries = () => ({
    "points.npy": npy(new Float32Array([0.25, 0.5, 0.75, 0.5]), [2, 2]),
    "density.npy": npy(new Float32Array(16).fill(1), [4, 4]),
    "init_log_covariance_channels.npy": npy(new Float32Array([0, 0, 0, Math.log(2), 0, 0]), [2, 3]),
    "init_color.npy": npy(new Float32Array([1, 0, 0, 0, 1, 0]), [2, 3]),
  });
  it("loads a CLI archive, stored and deflated", async () => {
    for (const deflate of [false, true]) {
      const scene = await loadNpz(zip(cliEntries(), deflate));
      expect(scene.width).toBe(4);
      expect(scene.height).toBe(4);
      expect(scene.states).toHaveLength(1);
      expect(scene.states[0].label).toBe("initializer");
      expect(Array.from(scene.states[0].xy)).toEqual([0.25, 0.5, 0.75, 0.5]);
      expect(scene.states[0].scale[2]).toBeCloseTo(Math.exp(0.5 * Math.log(2)), 5);
      expect(scene.density?.length).toBe(16);
    }
  });
  it("loads web keys with per-state arrays and meta", async () => {
    const meta = new TextEncoder().encode(JSON.stringify({ crop: { w: 3, h: 2 }, note: "x" }));
    const K = 2, N = 2;
    const entries = {
      "width.npy": npy(new BigInt64Array([8n]), []),
      "height.npy": npy(new BigInt64Array([4n]), []),
      "seed.npy": npy(new BigInt64Array([12345n]), []),
      "states_xy.npy": npy(new Float32Array([0.1, 0.1, 0.9, 0.9, 0.2, 0.2, 0.8, 0.8]), [K, N, 2]),
      "states_cov.npy": npy(new Float32Array(K * N * 3), [K, N, 3]),
      "states_color.npy": npy(new Float32Array(K * N * 3).fill(0.5), [K, N, 3]),
      "final_rendered.npy": npy(new Float32Array(3 * 4 * 8), [3, 4, 8]),
      "point_weights.npy": npy(new Float32Array([1, 1]), [2]),
      "meta.npy": npy(meta, [meta.length]),
    };
    const scene = await loadNpz(zip(entries, true));
    expect(scene.width).toBe(8);
    expect(scene.height).toBe(4);
    expect(scene.states).toHaveLength(2);
    expect(Array.from(scene.states[1].xy)).toEqual([0.2, 0.2, 0.8, 0.8].map((v) => Math.fround(v)));
    expect(scene.crop).toEqual({ w: 3, h: 2 });
    expect(scene.meta?.seed).toBe(12345);
    expect(scene.meta?.note).toBe("x");
  });
  it("loads the C++ triple and engine results", () => {
    const t = loadNpyTriple({ points: npy(new Float32Array([0.5, 0.5]), [1, 2]).buffer, cov: npy(new Float32Array([0, 0, 0]), [1, 3]).buffer, rgb: npy(new Float32Array([1, 1, 1]), [1, 3]).buffer }, { width: 16, height: 8 });
    expect(t.states[0].scale[0]).toBeCloseTo(1, 5);
    const snap = { k: 0, xy: new Float32Array([0.5, 0.5]), cov: new Float32Array([0, 0, 0]), color: new Float32Array([1, 1, 1]) };
    const run = { width: 32, height: 32, crop: { w: 30, h: 30 }, n: 1, seed: 1, density: new Float32Array(1024), init: snap, final: { ...snap, k: 6 } };
    const scene = fromRunResult(run);
    expect(scene.states).toHaveLength(2);
    expect(scene.states[1].label).toBe("refinement 6");
    expect(scene.crop).toEqual({ w: 30, h: 30 });
  });
});
