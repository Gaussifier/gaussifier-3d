/** Minimal zip reader for NPZ archives: central directory walk plus raw-deflate via DecompressionStream. */

export interface ZipEntry {
  name: string;
  method: number;
  compressedSize: number;
  size: number;
  /** Compressed payload view into the archive buffer. */
  data: Uint8Array;
}

function latin1(bytes: Uint8Array): string {
  let s = "";
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  return s;
}

export function readZip(buffer: ArrayBuffer): ZipEntry[] {
  const dv = new DataView(buffer);
  const u8 = new Uint8Array(buffer);
  let eocd = -1;
  for (let i = buffer.byteLength - 22; i >= Math.max(0, buffer.byteLength - 22 - 65535); i--) {
    if (dv.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  }
  if (eocd < 0) throw new Error("not a zip archive: no end-of-central-directory record");
  const count = dv.getUint16(eocd + 10, true);
  let p = dv.getUint32(eocd + 16, true);
  const entries: ZipEntry[] = [];
  for (let e = 0; e < count; e++) {
    if (dv.getUint32(p, true) !== 0x02014b50) throw new Error("corrupt zip central directory");
    const method = dv.getUint16(p + 10, true);
    const compressedSize = dv.getUint32(p + 20, true);
    const size = dv.getUint32(p + 24, true);
    const nameLen = dv.getUint16(p + 28, true);
    const extraLen = dv.getUint16(p + 30, true);
    const commentLen = dv.getUint16(p + 32, true);
    const localOffset = dv.getUint32(p + 42, true);
    const name = latin1(u8.subarray(p + 46, p + 46 + nameLen));
    if (dv.getUint32(localOffset, true) !== 0x04034b50) throw new Error(`corrupt zip local header for ${name}`);
    const localNameLen = dv.getUint16(localOffset + 26, true);
    const localExtraLen = dv.getUint16(localOffset + 28, true);
    const dataStart = localOffset + 30 + localNameLen + localExtraLen;
    entries.push({ name, method, compressedSize, size, data: u8.subarray(dataStart, dataStart + compressedSize) });
    p += 46 + nameLen + extraLen + commentLen;
  }
  return entries;
}

export async function inflateRaw(data: Uint8Array): Promise<Uint8Array> {
  const copy = data.slice();
  const stream = new Blob([copy]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export async function zipEntryBytes(entry: ZipEntry): Promise<Uint8Array> {
  if (entry.method === 0) return entry.data.slice();
  if (entry.method === 8) {
    const out = await inflateRaw(entry.data);
    if (out.byteLength !== entry.size) throw new Error(`zip entry ${entry.name}: inflated ${out.byteLength} bytes, expected ${entry.size}`);
    return out;
  }
  throw new Error(`zip entry ${entry.name}: unsupported compression method ${entry.method}`);
}
