/** NPY writer (version 1.0 headers, little-endian, C order). */

type Dtype = "<f4" | "<i4" | "<i8" | "|u1";

/** Encode one array as NPY (little endian, C order). */
export function encodeNpy(data: Float32Array | Int32Array | BigInt64Array | Uint8Array, shape: number[]): Uint8Array {
  const dtype: Dtype = data instanceof Float32Array ? "<f4" : data instanceof Int32Array ? "<i4" : data instanceof BigInt64Array ? "<i8" : "|u1";
  const shapeText = shape.length === 0 ? "()" : shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(", ")})`;
  let header = `{'descr': '${dtype}', 'fortran_order': False, 'shape': ${shapeText}, }`;
  const prefix = 10; // magic (6) + version (2) + header length (2)
  const pad = 64 - ((prefix + header.length + 1) % 64);
  header = header + " ".repeat(pad % 64) + "\n";
  const bytes = new Uint8Array(prefix + header.length + data.byteLength);
  bytes.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
  new DataView(bytes.buffer).setUint16(8, header.length, true);
  for (let i = 0; i < header.length; i++) bytes[prefix + i] = header.charCodeAt(i);
  bytes.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), prefix + header.length);
  return bytes;
}
