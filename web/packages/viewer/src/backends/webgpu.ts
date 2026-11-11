import { SPLAT_CONTRIBUTION_WGSL } from "@gaussifier/wgsl";
import type { RenderBackend, SceneState, View } from "../types.js";
import { CUTOFF_RADIUS, INSTANCE_FLOATS, buildInstances } from "../math.js";
import type { OverlayState } from "../overlays.js";
import { BackendBase, CELL_TINT_WGSL, COLORMAP_WGSL, DOT_CORE, DOT_CORE_ALPHA, DOT_HALO_ALPHA, INSTANCE_ATTRIBUTES, RING_CORE, RING_CORE_ALPHA, RING_FADE_WGSL, RING_HALO_ALPHA, VORONOI_TINT_MIX, centerMarker, gpuBytes, maxOf, planarToRgba, type BackendExtras } from "./common.js";

const INSTANCE_VERTEX_ATTRIBUTES: GPUVertexAttribute[] = INSTANCE_ATTRIBUTES.map((a) => ({ shaderLocation: a.location, offset: a.offset, format: a.size === 1 ? "float32" : (`float32x${a.size}` as GPUVertexFormat) }));

const SPLAT_WGSL = `
struct View { zoom: f32, tx: f32, ty: f32, pad0: f32, canvas: vec2<f32>, pad1: vec2<f32> };
@group(0) @binding(0) var<uniform> view: View;
struct Inst {
  @location(0) center: vec2<f32>, @location(1) conic: vec3<f32>, @location(2) color: vec3<f32>,
  @location(3) ext: vec2<f32>, @location(4) valid: f32,
};
struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) @interpolate(flat) center: vec2<f32>,
  @location(1) @interpolate(flat) conic: vec3<f32>,
  @location(2) @interpolate(flat) color: vec3<f32>,
};
${SPLAT_CONTRIBUTION_WGSL}
@vertex fn vs_main(@builtin(vertex_index) vi: u32, inst: Inst) -> VSOut {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
  let c = corners[vi];
  let dc = vec2<f32>(view.zoom * (inst.center.x + 0.5) + view.tx, view.zoom * (inst.center.y + 0.5) + view.ty);
  let half = inst.ext * view.zoom + vec2<f32>(1.0, 1.0);
  var p = dc + c * half;
  if (inst.valid < 0.5) { p = vec2<f32>(-10.0, -10.0); }
  var out: VSOut;
  out.pos = vec4<f32>(p.x / view.canvas.x * 2.0 - 1.0, 1.0 - p.y / view.canvas.y * 2.0, 0.0, 1.0);
  out.center = inst.center; out.conic = inst.conic; out.color = inst.color;
  return out;
}
@fragment fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let sx = (in.pos.x - view.tx) / view.zoom - 0.5;
  let sy = (in.pos.y - view.ty) / view.zoom - 0.5;
  let a = splat_alpha(in.center.x - sx, in.center.y - sy, in.conic);
  if (a <= 0.0) { discard; }
  return vec4<f32>(in.color * a, 1.0);
}
`;

const RESOLVE_WGSL = `
struct RP { zoom: f32, tx: f32, ty: f32, gain: f32, canvas: vec2<f32>, src: vec2<f32>, densityMax: f32, flags: f32, hasImage: f32, hasDensity: f32, hasVoronoi: f32, borderMix: f32, pad1: f32, pad2: f32 };
@group(0) @binding(0) var<uniform> rp: RP;
@group(0) @binding(1) var accum: texture_2d<f32>;
@group(0) @binding(2) var img: texture_2d<f32>;
@group(0) @binding(3) var dens: texture_2d<f32>;
@group(0) @binding(4) var own: texture_2d<u32>;
${COLORMAP_WGSL}
${CELL_TINT_WGSL}
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var pts = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(pts[vi], 0.0, 1.0);
}
@fragment fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let px = vec2<i32>(pos.xy);
  var r = clamp(textureLoad(accum, px, 0).rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  r = round(r * 255.0) / 255.0;
  let s = (pos.xy - vec2<f32>(rp.tx, rp.ty)) / rp.zoom - 0.5;
  let si = vec2<i32>(floor(s + vec2<f32>(0.5, 0.5)));
  let inside = si.x >= 0 && si.y >= 0 && si.x < i32(rp.src.x) && si.y < i32(rp.src.y);
  let flags = u32(rp.flags);
  var out = r;
  if (!inside) {
    out = vec3<f32>(0.12, 0.12, 0.13);
  } else {
    var im = vec3<f32>(0.0);
    if (rp.hasImage > 0.5) { im = textureLoad(img, si, 0).rgb; }
    if ((flags & 1u) != 0u && rp.hasImage > 0.5) { out = im; }
    if ((flags & 2u) != 0u && rp.hasImage > 0.5) { out = clamp(abs(r - im) * rp.gain, vec3<f32>(0.0), vec3<f32>(1.0)); }
    if ((flags & 4u) != 0u && rp.hasDensity > 0.5) {
      let d = textureLoad(dens, si, 0).r / max(rp.densityMax, 1e-6);
      out = colormap(clamp(d, 0.0, 1.0));
    }
    if ((flags & 8u) != 0u && rp.hasVoronoi > 0.5) {
      // Every cell is tinted by its owner; a border is one display pixel wide, found by comparing
      // with the source pixels under the next display pixel right and down.
      let o = textureLoad(own, si, 0).r;
      out = mix(out, cell_tint(o), ${VORONOI_TINT_MIX});
      let sr = vec2<i32>(floor(s + vec2<f32>(1.0 / rp.zoom, 0.0) + vec2<f32>(0.5, 0.5)));
      let sd = vec2<i32>(floor(s + vec2<f32>(0.0, 1.0 / rp.zoom) + vec2<f32>(0.5, 0.5)));
      var border = sr.x < i32(rp.src.x) && textureLoad(own, sr, 0).r != o;
      if (sd.y < i32(rp.src.y) && textureLoad(own, sd, 0).r != o) { border = true; }
      if (border) { out = mix(out, vec3<f32>(0.04, 0.04, 0.05), rp.borderMix); }
    }
  }
  return vec4<f32>(out, 1.0);
}
`;

const BLIT_WGSL = `
@group(0) @binding(0) var src: texture_2d<f32>;
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var pts = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(pts[vi], 0.0, 1.0);
}
@fragment fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  return textureLoad(src, vec2<i32>(pos.xy), 0);
}
`;

/** Center discs per instance: a screen-space quad around the center, shaded by pixel distance. Premultiplied. */
const DOT_WGSL = `
struct DP { zoom: f32, tx: f32, ty: f32, radius: f32, canvas: vec2<f32>, alpha: f32, pad: f32 };
@group(0) @binding(0) var<uniform> dp: DP;
struct Inst {
  @location(0) center: vec2<f32>, @location(1) conic: vec3<f32>, @location(2) color: vec3<f32>,
  @location(3) ext: vec2<f32>, @location(4) valid: f32,
};
struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) @interpolate(flat) dc: vec2<f32> };
@vertex fn vs_main(@builtin(vertex_index) vi: u32, inst: Inst) -> VSOut {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
  let dc = vec2<f32>(dp.zoom * (inst.center.x + 0.5) + dp.tx, dp.zoom * (inst.center.y + 0.5) + dp.ty);
  var p = dc + corners[vi] * (dp.radius + 2.0);
  if (inst.valid < 0.5) { p = vec2<f32>(-10.0, -10.0); }
  var out: VSOut;
  out.pos = vec4<f32>(p.x / dp.canvas.x * 2.0 - 1.0, 1.0 - p.y / dp.canvas.y * 2.0, 0.0, 1.0);
  out.dc = dc;
  return out;
}
@fragment fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let d = length(in.pos.xy - in.dc);
  let core = 1.0 - smoothstep(dp.radius - 0.5, dp.radius + 0.5, d);
  let halo = 1.0 - smoothstep(dp.radius + 0.5, dp.radius + 1.5, d);
  if (halo <= 0.002) { discard; }
  let alpha = max(core * ${DOT_CORE_ALPHA}, halo * ${DOT_HALO_ALPHA}) * dp.alpha;
  let rgb = vec3<f32>(${DOT_CORE[0]}, ${DOT_CORE[1]}, ${DOT_CORE[2]}) * core * ${DOT_CORE_ALPHA} * dp.alpha;
  return vec4<f32>(rgb, alpha);
}
`;

/**
 * Ellipse rings drawn per instance from the same packed instance buffer as the splats: the
 * fragment computes the Mahalanobis distance d and shades an antialiased ring of constant
 * display width where d = sigma, using fwidth(d) as the pixel scale. Output is premultiplied.
 */
const RING_WGSL = `
struct RG { zoom: f32, tx: f32, ty: f32, sigma: f32, canvas: vec2<f32>, width: f32, pad: f32 };
@group(0) @binding(0) var<uniform> rg: RG;
struct Inst {
  @location(0) center: vec2<f32>, @location(1) conic: vec3<f32>, @location(2) color: vec3<f32>,
  @location(3) ext: vec2<f32>, @location(4) valid: f32,
};
struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) @interpolate(flat) center: vec2<f32>,
  @location(1) @interpolate(flat) conic: vec3<f32>,
  @location(2) @interpolate(flat) fade: f32,
};
@vertex fn vs_main(@builtin(vertex_index) vi: u32, inst: Inst) -> VSOut {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
  let c = corners[vi];
  let dc = vec2<f32>(rg.zoom * (inst.center.x + 0.5) + rg.tx, rg.zoom * (inst.center.y + 0.5) + rg.ty);
  // ext is the cutoff ellipse's half extent; scale it to the ring radius and pad for the halo.
  let radius = inst.ext * (rg.sigma / ${CUTOFF_RADIUS}) * rg.zoom;
  let half = radius + vec2<f32>(rg.width + 2.5);
  var p = dc + c * half;
  if (inst.valid < 0.5) { p = vec2<f32>(-10.0, -10.0); }
  var out: VSOut;
  out.pos = vec4<f32>(p.x / rg.canvas.x * 2.0 - 1.0, 1.0 - p.y / rg.canvas.y * 2.0, 0.0, 1.0);
  out.center = inst.center; out.conic = inst.conic;
  let rpx = max(radius.x, radius.y);
  out.fade = ${RING_FADE_WGSL};
  return out;
}
@fragment fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let sx = (in.pos.x - rg.tx) / rg.zoom - 0.5;
  let sy = (in.pos.y - rg.ty) / rg.zoom - 0.5;
  let dx = in.center.x - sx; let dy = in.center.y - sy;
  let d = sqrt(max(in.conic.x * dx * dx + 2.0 * in.conic.y * dx * dy + in.conic.z * dy * dy, 0.0));
  let px = abs(d - rg.sigma) / max(fwidth(d), 1e-6);
  let core = 1.0 - smoothstep(rg.width * 0.5 - 0.5, rg.width * 0.5 + 0.5, px);
  let halo = 1.0 - smoothstep(rg.width * 0.5 + 0.5, rg.width * 0.5 + 1.5, px);
  if (halo <= 0.002) { discard; }
  let alpha = max(core * ${RING_CORE_ALPHA}, halo * ${RING_HALO_ALPHA}) * in.fade;
  let rgb = vec3<f32>(${RING_CORE[0]}, ${RING_CORE[1]}, ${RING_CORE[2]}) * core * ${RING_CORE_ALPHA} * in.fade;
  return vec4<f32>(rgb, alpha);
}
`;


export class WebGpuBackend extends BackendBase implements RenderBackend, BackendExtras {
  readonly kind = "webgpu" as const;
  readonly accumFormat: GPUTextureFormat;

  private instanceBuf: GPUBuffer | null = null;
  private accumTex: GPUTexture | null = null;
  private displayTex: GPUTexture | null = null;
  private imageTex: GPUTexture;
  private densityTex: GPUTexture;
  private ownerTex: GPUTexture;
  private readonly viewBuf: GPUBuffer;
  private readonly resolveBuf: GPUBuffer;
  private readonly ringBuf: GPUBuffer;
  private readonly dotBuf: GPUBuffer;
  private readonly splatPipeline: GPURenderPipeline;
  private readonly resolvePipeline: GPURenderPipeline;
  private readonly blitPipeline: GPURenderPipeline;
  private readonly ringPipeline: GPURenderPipeline;
  private readonly dotPipeline: GPURenderPipeline;
  private readonly splatLayout: GPUBindGroupLayout;
  private readonly resolveLayout: GPUBindGroupLayout;
  private readonly blitLayout: GPUBindGroupLayout;
  private readonly lineLayout: GPUBindGroupLayout;

  private constructor(readonly device: GPUDevice, readonly canvas: HTMLCanvasElement, private readonly context: GPUCanvasContext, readonly canvasFormat: GPUTextureFormat) {
    super();
    this.accumFormat = device.features.has("float32-blendable") ? "rgba32float" : "rgba16float";
    const uniform = (size: number, label: string) => device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label });
    this.viewBuf = uniform(32, "viewer view");
    this.resolveBuf = uniform(64, "viewer resolve");
    this.ringBuf = uniform(32, "viewer rings");
    this.dotBuf = uniform(32, "viewer dots");
    this.splatLayout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }] });
    const tex = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } });
    this.resolveLayout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }, tex(1), tex(2), tex(3), { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "uint" } }] });
    this.blitLayout = device.createBindGroupLayout({ entries: [tex(0)] });
    this.lineLayout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }] });

    const splatModule = device.createShaderModule({ code: SPLAT_WGSL, label: "viewer splat" });
    this.splatPipeline = device.createRenderPipeline({
      label: "viewer splat",
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.splatLayout] }),
      vertex: {
        module: splatModule, entryPoint: "vs_main",
        buffers: [{
          arrayStride: INSTANCE_FLOATS * 4, stepMode: "instance",
          attributes: INSTANCE_VERTEX_ATTRIBUTES,
        }],
      },
      fragment: {
        module: splatModule, entryPoint: "fs_main",
        targets: [{ format: this.accumFormat, blend: { color: { srcFactor: "one", dstFactor: "one", operation: "add" }, alpha: { srcFactor: "one", dstFactor: "one", operation: "add" } } }],
      },
      primitive: { topology: "triangle-list" },
    });
    const resolveModule = device.createShaderModule({ code: RESOLVE_WGSL, label: "viewer resolve" });
    this.resolvePipeline = device.createRenderPipeline({
      label: "viewer resolve",
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.resolveLayout] }),
      vertex: { module: resolveModule, entryPoint: "vs_main" },
      fragment: { module: resolveModule, entryPoint: "fs_main", targets: [{ format: "rgba8unorm" }] },
      primitive: { topology: "triangle-list" },
    });
    const blitModule = device.createShaderModule({ code: BLIT_WGSL, label: "viewer blit" });
    this.blitPipeline = device.createRenderPipeline({
      label: "viewer blit",
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.blitLayout] }),
      vertex: { module: blitModule, entryPoint: "vs_main" },
      fragment: { module: blitModule, entryPoint: "fs_main", targets: [{ format: canvasFormat }] },
      primitive: { topology: "triangle-list" },
    });
    // Rings and dots are instanced over the splat instance buffer and write premultiplied color.
    const markerPipeline = (code: string, label: string) => device.createRenderPipeline({
      label,
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.lineLayout] }),
      vertex: {
        module: device.createShaderModule({ code, label }), entryPoint: "vs_main",
        buffers: [{
          arrayStride: INSTANCE_FLOATS * 4, stepMode: "instance",
          attributes: INSTANCE_VERTEX_ATTRIBUTES,
        }],
      },
      fragment: { module: device.createShaderModule({ code, label }), entryPoint: "fs_main", targets: [{ format: "rgba8unorm", blend: { color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" }, alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" } } }] },
      primitive: { topology: "triangle-list" },
    });
    this.ringPipeline = markerPipeline(RING_WGSL, "viewer rings");
    this.dotPipeline = markerPipeline(DOT_WGSL, "viewer dots");
    this.imageTex = this.makeDummy("rgba32float");
    this.densityTex = this.makeDummy("r32float");
    this.ownerTex = this.makeDummy("r32uint");
    this.resize(canvas.width || 1, canvas.height || 1);
  }

  static async create(canvas: HTMLCanvasElement, device?: GPUDevice): Promise<WebGpuBackend> {
    if (typeof navigator === "undefined" || !navigator.gpu) throw new Error("WebGPU is not available in this browser");
    if (!device) {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) throw new Error("WebGPU: no adapter");
      const wanted: GPUFeatureName[] = ["float32-blendable"];
      device = await adapter.requestDevice({ requiredFeatures: wanted.filter((f) => adapter.features.has(f)) });
    }
    const context = canvas.getContext("webgpu");
    if (!context) throw new Error("WebGPU: canvas context unavailable");
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: "opaque" });
    return new WebGpuBackend(device, canvas, context, format);
  }

  private makeDummy(format: GPUTextureFormat): GPUTexture {
    return this.device.createTexture({ size: [1, 1], format, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  }

  resize(width: number, height: number): void {
    width = Math.max(1, Math.floor(width));
    height = Math.max(1, Math.floor(height));
    if (this.canvas.width !== width) this.canvas.width = width;
    if (this.canvas.height !== height) this.canvas.height = height;
    if (width === this.width && height === this.height && this.accumTex) return;
    this.width = width;
    this.height = height;
    this.accumTex?.destroy();
    this.displayTex?.destroy();
    this.accumTex = this.device.createTexture({ size: [width, height], format: this.accumFormat, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING, label: "viewer accum" });
    this.displayTex = this.device.createTexture({ size: [width, height], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC, label: "viewer display" });
  }

  setState(state: SceneState, sourceWidth: number, sourceHeight: number): void {
    this.srcW = sourceWidth;
    this.srcH = sourceHeight;
    const inst = buildInstances(state, sourceWidth, sourceHeight);
    this.instanceCount = state.xy.length / 2;
    this.instanceBuf?.destroy();
    this.instanceBuf = this.device.createBuffer({ size: Math.max(16, inst.byteLength), usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, label: "viewer instances" });
    this.device.queue.writeBuffer(this.instanceBuf, 0, gpuBytes(inst));
  }

  setImage(image: Float32Array | null, width: number, height: number): void {
    this.imageTex.destroy();
    if (!image) { this.imageTex = this.makeDummy("rgba32float"); this.hasImage = false; return; }
    this.imageTex = this.device.createTexture({ size: [width, height], format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST, label: "viewer image" });
    const rgba = planarToRgba(image, width, height);
    this.device.queue.writeTexture({ texture: this.imageTex }, gpuBytes(rgba), { bytesPerRow: width * 16, rowsPerImage: height }, [width, height]);
    this.hasImage = true;
  }

  setDensity(density: Float32Array | null, width: number, height: number): void {
    this.densityTex.destroy();
    if (!density) { this.densityTex = this.makeDummy("r32float"); this.hasDensity = false; return; }
    this.densityTex = this.device.createTexture({ size: [width, height], format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST, label: "viewer density" });
    this.device.queue.writeTexture({ texture: this.densityTex }, gpuBytes(density), { bytesPerRow: width * 4, rowsPerImage: height }, [width, height]);
    this.densityMax = maxOf(density);
    this.hasDensity = true;
  }

  setOwners(owners: Uint32Array | null, width: number, height: number): void {
    this.ownerTex.destroy();
    if (!owners) { this.ownerTex = this.makeDummy("r32uint"); this.hasVoronoi = false; return; }
    this.ownerTex = this.device.createTexture({ size: [width, height], format: "r32uint", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST, label: "viewer owners" });
    this.device.queue.writeTexture({ texture: this.ownerTex }, gpuBytes(owners), { bytesPerRow: width * 4, rowsPerImage: height }, [width, height]);
    this.hasVoronoi = true;
  }

  render(view: View): void {
    if (this.disposed || !this.accumTex || !this.displayTex) return;
    const { device } = this;
    const w = this.width, h = this.height;
    device.queue.writeBuffer(this.viewBuf, 0, gpuBytes(new Float32Array([view.zoom, view.tx, view.ty, 0, w, h, 0, 0])));
    const r = this.resolveParams(view);
    device.queue.writeBuffer(this.resolveBuf, 0, gpuBytes(new Float32Array([r.zoom, r.tx, r.ty, r.gain, r.canvasW, r.canvasH, r.srcW, r.srcH, r.densityMax, r.flags, r.hasImage, r.hasDensity, r.hasVoronoi, r.borderMix, 0, 0])));
    const encoder = device.createCommandEncoder({ label: "viewer frame" });
    const accumPass = encoder.beginRenderPass({ colorAttachments: [{ view: this.accumTex.createView(), loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: "store" }] });
    if (this.instanceBuf && this.instanceCount > 0) {
      accumPass.setPipeline(this.splatPipeline);
      accumPass.setBindGroup(0, device.createBindGroup({ layout: this.splatLayout, entries: [{ binding: 0, resource: { buffer: this.viewBuf } }] }));
      accumPass.setVertexBuffer(0, this.instanceBuf);
      accumPass.draw(6, this.instanceCount);
    }
    accumPass.end();
    const displayView = this.displayTex.createView();
    const resolvePass = encoder.beginRenderPass({ colorAttachments: [{ view: displayView, loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: "store" }] });
    resolvePass.setPipeline(this.resolvePipeline);
    resolvePass.setBindGroup(0, device.createBindGroup({ layout: this.resolveLayout, entries: [
      { binding: 0, resource: { buffer: this.resolveBuf } },
      { binding: 1, resource: this.accumTex.createView() },
      { binding: 2, resource: this.imageTex.createView() },
      { binding: 3, resource: this.densityTex.createView() },
      { binding: 4, resource: this.ownerTex.createView() },
    ] }));
    resolvePass.draw(3);
    resolvePass.end();
    const haveInstances = this.instanceBuf !== null && this.instanceCount > 0;
    const drawRings = this.overlays.ellipses && haveInstances;
    const drawDots = this.overlays.centers && haveInstances;
    if (drawRings || drawDots) {
      const overlayPass = encoder.beginRenderPass({ colorAttachments: [{ view: displayView, loadOp: "load", storeOp: "store" }] });
      if (drawRings) {
        device.queue.writeBuffer(this.ringBuf, 0, gpuBytes(new Float32Array([view.zoom, view.tx, view.ty, this.overlays.ellipseSigma, w, h, this.overlays.ellipseWidth, 0])));
        overlayPass.setPipeline(this.ringPipeline);
        overlayPass.setBindGroup(0, device.createBindGroup({ layout: this.lineLayout, entries: [{ binding: 0, resource: { buffer: this.ringBuf } }] }));
        overlayPass.setVertexBuffer(0, this.instanceBuf!);
        overlayPass.draw(6, this.instanceCount);
      }
      if (drawDots) {
        const m = centerMarker(view.zoom, this.srcW, this.srcH, this.instanceCount);
        device.queue.writeBuffer(this.dotBuf, 0, gpuBytes(new Float32Array([view.zoom, view.tx, view.ty, m.radius, w, h, m.alpha, 0])));
        overlayPass.setPipeline(this.dotPipeline);
        overlayPass.setBindGroup(0, device.createBindGroup({ layout: this.lineLayout, entries: [{ binding: 0, resource: { buffer: this.dotBuf } }] }));
        overlayPass.setVertexBuffer(0, this.instanceBuf!);
        overlayPass.draw(6, this.instanceCount);
      }
      overlayPass.end();
    }
    const blitPass = encoder.beginRenderPass({ colorAttachments: [{ view: this.context.getCurrentTexture().createView(), loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: "store" }] });
    blitPass.setPipeline(this.blitPipeline);
    blitPass.setBindGroup(0, device.createBindGroup({ layout: this.blitLayout, entries: [{ binding: 0, resource: displayView }] }));
    blitPass.draw(3);
    blitPass.end();
    device.queue.submit([encoder.finish()]);
  }

  async readPixels(): Promise<Uint8ClampedArray> {
    if (!this.displayTex) return new Uint8ClampedArray(0);
    const w = this.width, h = this.height;
    const bytesPerRow = Math.ceil((w * 4) / 256) * 256;
    const staging = this.device.createBuffer({ size: bytesPerRow * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture: this.displayTex }, { buffer: staging, bytesPerRow, rowsPerImage: h }, [w, h]);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const padded = new Uint8Array(staging.getMappedRange());
    const out = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) out.set(padded.subarray(y * bytesPerRow, y * bytesPerRow + w * 4), y * w * 4);
    staging.unmap();
    staging.destroy();
    return out;
  }

  dispose(): void {
    this.disposed = true;
    this.accumTex?.destroy();
    this.displayTex?.destroy();
    this.instanceBuf?.destroy();
    this.imageTex.destroy();
    this.densityTex.destroy();
    this.ownerTex.destroy();
    this.viewBuf.destroy();
    this.resolveBuf.destroy();
    this.ringBuf.destroy();
    this.dotBuf.destroy();
    try { this.context.unconfigure(); } catch { /* ignore */ }
  }
}
