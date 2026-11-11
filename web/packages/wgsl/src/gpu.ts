/**
 * Small GPU helper shared by the engine and the viewer. Kernels use explicit bind group layouts
 * (auto layouts drop unused bindings and then reject full bind groups), bind at most eight
 * storage buffers per stage (the default limit), never alias one buffer across two bindings of
 * a dispatch, and callers record whole stages into one encoder.
 */

export type BindingKind = "uniform" | "read-only-storage" | "storage";

export interface KernelSpec {
  code: string;
  entryPoint: string;
  /** Binding kinds in binding-index order. */
  bindings: BindingKind[];
  label?: string;
}

const bufferIds = new WeakMap<GPUBuffer, number>();
let nextBufferId = 1;
function bufferId(b: GPUBuffer): number {
  let id = bufferIds.get(b);
  if (id === undefined) { id = nextBufferId++; bufferIds.set(b, id); }
  return id;
}

export class Kernel {
  readonly pipeline: GPUComputePipeline;
  readonly layout: GPUBindGroupLayout;
  private readonly cache = new Map<string, GPUBindGroup>();
  private constructor(readonly device: GPUDevice, readonly spec: KernelSpec, pipeline: GPUComputePipeline, layout: GPUBindGroupLayout) {
    this.pipeline = pipeline;
    this.layout = layout;
  }

  static create(device: GPUDevice, spec: KernelSpec): Kernel {
    const storageCount = spec.bindings.filter((b) => b !== "uniform").length;
    if (storageCount > device.limits.maxStorageBuffersPerShaderStage) {
      throw new Error(`${spec.label ?? spec.entryPoint}: ${storageCount} storage buffers exceed the device limit ${device.limits.maxStorageBuffersPerShaderStage}`);
    }
    const layout = device.createBindGroupLayout({
      label: spec.label,
      entries: spec.bindings.map((kind, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: kind } })),
    });
    const module = device.createShaderModule({ code: spec.code, label: spec.label });
    const pipeline = device.createComputePipeline({
      label: spec.label,
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: spec.entryPoint },
    });
    return new Kernel(device, spec, pipeline, layout);
  }

  bindGroup(buffers: GPUBuffer[]): GPUBindGroup {
    if (buffers.length !== this.spec.bindings.length) {
      throw new Error(`${this.spec.label ?? this.spec.entryPoint}: expected ${this.spec.bindings.length} buffers, got ${buffers.length}`);
    }
    const seen = new Set<GPUBuffer>();
    buffers.forEach((b, i) => {
      if (this.spec.bindings[i] === "storage" && seen.has(b)) {
        throw new Error(`${this.spec.label ?? this.spec.entryPoint}: buffer aliased across bindings with a writable use`);
      }
      seen.add(b);
    });
    const key = buffers.map(bufferId).join(",");
    let group = this.cache.get(key);
    if (!group) {
      group = this.device.createBindGroup({
        layout: this.layout,
        entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      this.cache.set(key, group);
    }
    return group;
  }

  /** Drop cached bind groups, for example after buffers were destroyed. */
  clearCache(): void { this.cache.clear(); }

  dispatch(pass: GPUComputePassEncoder, buffers: GPUBuffer[], x: number, y = 1, z = 1): void {
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup(buffers));
    pass.dispatchWorkgroups(x, y, z);
  }
}

/** GPUBufferUsage.STORAGE | COPY_SRC | COPY_DST as the spec's fixed flag values, so importing this module needs no WebGPU globals (Node tests). */
export const STORAGE_USAGE = 0x80 | 0x4 | 0x8;

export function createStorageBuffer(device: GPUDevice, byteLength: number, label?: string, usage = STORAGE_USAGE): GPUBuffer {
  return device.createBuffer({ size: Math.max(16, Math.ceil(byteLength / 4) * 4), usage, label });
}

export function createUniformBuffer(device: GPUDevice, data: ArrayBufferView, label?: string): GPUBuffer {
  const size = Math.max(16, Math.ceil(data.byteLength / 16) * 16);
  const buffer = device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label });
  device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
  return buffer;
}

export function uploadStorage(device: GPUDevice, data: ArrayBufferView, label?: string, usage = STORAGE_USAGE): GPUBuffer {
  const buffer = createStorageBuffer(device, data.byteLength, label, usage);
  device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
  return buffer;
}

/** Copy `byteLength` bytes of `source` into a staging buffer and map it. Submits its own command buffer. */
export async function readBuffer(device: GPUDevice, source: GPUBuffer, byteLength: number, offset = 0): Promise<ArrayBuffer> {
  const size = Math.ceil(byteLength / 4) * 4;
  const staging = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, offset, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const copy = staging.getMappedRange().slice(0, byteLength);
  staging.unmap();
  staging.destroy();
  return copy;
}

export function workgroups(count: number, size: number): number {
  return Math.max(1, Math.ceil(count / size));
}
