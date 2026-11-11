/** NPY parser: versions 1.0 to 3.0, little-endian, C order. */

export type NpyData = Float32Array | Float64Array | Int32Array | Uint32Array | BigInt64Array | Uint8Array;

export interface NpyArray {
  dtype: string;
  shape: number[];
  data: NpyData;
}

const MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];

export function parseNpy(bytes: Uint8Array): NpyArray {
  for (let i = 0; i < MAGIC.length; i++) {
    if (bytes[i] !== MAGIC[i]) throw new Error("not an NPY file");
  }
  const major = bytes[6];
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let headerLen: number;
  let offset: number;
  if (major === 1) {
    headerLen = dv.getUint16(8, true);
    offset = 10;
  } else if (major === 2 || major === 3) {
    headerLen = dv.getUint32(8, true);
    offset = 12;
  } else {
    throw new Error(`unsupported NPY version ${major}`);
  }
  let header = "";
  for (let i = offset; i < offset + headerLen; i++) header += String.fromCharCode(bytes[i]);
  const descr = /'descr':\s*'([^']+)'/.exec(header);
  const fortran = /'fortran_order':\s*(True|False)/.exec(header);
  const shapeMatch = /'shape':\s*\(([^)]*)\)/.exec(header);
  if (!descr || !fortran || !shapeMatch) throw new Error(`malformed NPY header: ${header}`);
  if (fortran[1] === "True") throw new Error("Fortran-order NPY arrays are not supported");
  const shape = shapeMatch[1].split(",").map((s) => s.trim()).filter((s) => s.length > 0).map((s) => Number(s));
  const count = shape.reduce((a, b) => a * b, 1);
  const dataOffset = offset + headerLen;
  const dtype = descr[1];
  const make = (Ctor: { new (buf: ArrayBuffer): NpyData; BYTES_PER_ELEMENT: number }) => {
    const byteLength = count * Ctor.BYTES_PER_ELEMENT;
    if (dataOffset + byteLength > bytes.byteLength) throw new Error("NPY data truncated");
    const copy = bytes.slice(dataOffset, dataOffset + byteLength);
    return new Ctor(copy.buffer);
  };
  let data: NpyData;
  switch (dtype) {
    case "<f4": data = make(Float32Array); break;
    case "<f8": data = make(Float64Array); break;
    case "<i4": data = make(Int32Array); break;
    case "<u4": data = make(Uint32Array); break;
    case "<i8": data = make(BigInt64Array); break;
    case "|u1": case "|b1": case "|i1": data = make(Uint8Array); break;
    default: throw new Error(`unsupported NPY dtype ${dtype}`);
  }
  return { dtype, shape, data };
}

/** Convert any supported NPY payload to float32. */
export function npyToFloat32(arr: NpyArray): Float32Array {
  if (arr.data instanceof Float32Array) return arr.data;
  const out = new Float32Array(arr.data.length);
  if (arr.data instanceof BigInt64Array) {
    for (let i = 0; i < out.length; i++) out[i] = Number(arr.data[i]);
  } else {
    for (let i = 0; i < out.length; i++) out[i] = Number(arr.data[i]);
  }
  return out;
}

/** First element as a number; for 0-d scalar arrays. */
export function npyScalar(arr: NpyArray): number {
  const v = arr.data[0];
  return typeof v === "bigint" ? Number(v) : Number(v);
}
