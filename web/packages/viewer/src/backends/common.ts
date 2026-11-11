import type { RenderBackend, View } from "../types.js";
import { DEFAULT_OVERLAYS, type OverlayState } from "../overlays.js";

/** Extra methods both backends implement beyond the public RenderBackend contract. */
export interface BackendExtras {
  setImage(image: Float32Array | null, width: number, height: number): void;
  setDensity(density: Float32Array | null, width: number, height: number): void;
  setOverlays(overlays: Partial<OverlayState>): void;
  getOverlays(): OverlayState;
  /** Nearest-center index per source pixel for the Voronoi overlay, or null to drop it. */
  setOwners(owners: Uint32Array | null, width: number, height: number): void;
}

export type FullBackend = RenderBackend & BackendExtras;

/** The values the resolve shader needs, in the order both backends hand them over. */
export interface ResolveParams {
  zoom: number; tx: number; ty: number; gain: number;
  canvasW: number; canvasH: number; srcW: number; srcH: number;
  densityMax: number; flags: number; hasImage: number; hasDensity: number; hasVoronoi: number; borderMix: number;
}

/**
 * State both backends keep: canvas and source sizes, what has been uploaded, and the overlay
 * settings. Each backend adds its API's resources and implements the uploads and the frame.
 */
export abstract class BackendBase {
  protected width = 0;
  protected height = 0;
  protected srcW = 1;
  protected srcH = 1;
  protected instanceCount = 0;
  protected hasImage = false;
  protected hasDensity = false;
  protected hasVoronoi = false;
  protected densityMax = 1;
  protected overlays: OverlayState = { ...DEFAULT_OVERLAYS };
  protected disposed = false;

  setOverlays(overlays: Partial<OverlayState>): void {
    this.overlays = { ...this.overlays, ...overlays };
  }

  getOverlays(): OverlayState {
    return { ...this.overlays };
  }

  protected resolveParams(view: View): ResolveParams {
    return {
      zoom: view.zoom, tx: view.tx, ty: view.ty, gain: this.overlays.diffGain,
      canvasW: this.width, canvasH: this.height, srcW: this.srcW, srcH: this.srcH,
      densityMax: this.densityMax, flags: flagsOf(this.overlays), hasImage: +this.hasImage, hasDensity: +this.hasDensity, hasVoronoi: +this.hasVoronoi,
      borderMix: voronoiBorderMix(view.zoom, this.srcW, this.srcH, this.instanceCount),
    };
  }
}

/** The packed instance attributes as a vertex layout: location, float offset, and component count. */
export const INSTANCE_ATTRIBUTES: Array<{ location: number; offset: number; size: 1 | 2 | 3 }> = [
  { location: 0, offset: 0, size: 2 },   // center
  { location: 1, offset: 8, size: 3 },   // conic
  { location: 2, offset: 20, size: 3 },  // color
  { location: 3, offset: 32, size: 2 },  // cutoff half extents
  { location: 4, offset: 40, size: 1 },  // valid
];

/** GLSL twin of SPLAT_CONTRIBUTION_WGSL from @gaussifier/wgsl: the production simple-sum alpha. */
export const SPLAT_CONTRIBUTION_GLSL = `
float splat_alpha(float dx, float dy, vec3 conic) {
  float sigma = 0.5 * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy;
  if (sigma < 0.0) return 0.0;
  float alpha = exp(-sigma);
  if (alpha < 0.00392156862745098) return 0.0;
  return alpha;
}
`;

/** Overlay flag bits shared by the resolve shaders. */
export const FLAG_IMAGE = 1;
export const FLAG_DIFF = 2;
export const FLAG_DENSITY = 4;
export const FLAG_VORONOI = 8;

/**
 * Voronoi overlay look, identical in WGSL and GLSL: every cell takes a pastel tint from a
 * hash of its owner index (golden-ratio hue steps, so neighbours differ), and cell borders
 * are drawn dark. The tint is what makes cells readable when they are only a few pixels wide.
 */
export const VORONOI_TINT_MIX = 0.42;
export const VORONOI_BORDER_MIX = 0.78;

/**
 * Overlay detail scales with its size on screen so a zoomed-out view is not buried under
 * one-pixel borders and rings. Cell borders fade out below ~8 display pixels per cell;
 * ellipse rings fade to a quarter strength below ~7 display pixels of radius.
 */
export function voronoiBorderMix(zoom: number, srcW: number, srcH: number, count: number): number {
  const cellPx = zoom * Math.sqrt((srcW * srcH) / Math.max(1, count));
  const t = Math.min(1, Math.max(0, (cellPx - 2) / 6));
  return VORONOI_BORDER_MIX * t * t * (3 - 2 * t);
}
export const RING_FADE_WGSL = `mix(0.25, 1.0, smoothstep(2.0, 7.0, rpx))`;
export const RING_FADE_GLSL = `mix(0.25, 1.0, smoothstep(2.0, 7.0, rpx))`;

/**
 * Center markers: antialiased discs of constant screen size with a dark halo, like the rings.
 * The radius shrinks from 2.5 to 1 display pixel and the alpha drops as centers pack closer
 * than a few pixels, so a zoomed-out view shows a fine stipple instead of a solid orange wash.
 */
export const DOT_CORE = [1.0, 0.5, 0.15];
export const DOT_CORE_ALPHA = 0.95;
export const DOT_HALO_ALPHA = 0.7;
export function centerMarker(zoom: number, srcW: number, srcH: number, count: number): { radius: number; alpha: number } {
  const spacing = zoom * Math.sqrt((srcW * srcH) / Math.max(1, count));
  const t = Math.min(1, Math.max(0, (spacing - 3) / 9));
  return { radius: 1 + 1.5 * t, alpha: 0.45 + 0.55 * t };
}
export const CELL_TINT_WGSL = `
fn cell_tint(id: u32) -> vec3<f32> {
  let h = fract(f32(id % 4096u) * 0.618033988749895 + f32(id / 4096u) * 0.271828);
  let rgb = clamp(abs(fract(h + vec3<f32>(0.0, 0.6666667, 0.3333333)) * 6.0 - 3.0) - 1.0, vec3<f32>(0.0), vec3<f32>(1.0));
  return mix(vec3<f32>(1.0), rgb, 0.55);
}
`;
export const CELL_TINT_GLSL = `
vec3 cell_tint(uint id) {
  float h = fract(float(id % 4096u) * 0.618033988749895 + float(id / 4096u) * 0.271828);
  vec3 rgb = clamp(abs(fract(h + vec3(0.0, 0.6666667, 0.3333333)) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
  return mix(vec3(1.0), rgb, 0.55);
}
`;

/** Ellipse ring: light core over a dark halo so it reads on light and dark image regions. */
export const RING_CORE = [0.62, 0.96, 1.0];
export const RING_CORE_ALPHA = 0.95;
export const RING_HALO_ALPHA = 0.6;

/** Three-stop color ramp, identical in WGSL and GLSL. */
export const COLORMAP_WGSL = `
fn colormap(t: f32) -> vec3<f32> {
  let a = vec3<f32>(0.05, 0.03, 0.2); let b = vec3<f32>(0.2, 0.5, 0.9); let c = vec3<f32>(1.0, 0.9, 0.2);
  if (t < 0.5) { return mix(a, b, t * 2.0); }
  return mix(b, c, (t - 0.5) * 2.0);
}
`;
export const COLORMAP_GLSL = `
vec3 colormap(float t) {
  vec3 a = vec3(0.05, 0.03, 0.2); vec3 b = vec3(0.2, 0.5, 0.9); vec3 c = vec3(1.0, 0.9, 0.2);
  if (t < 0.5) return mix(a, b, t * 2.0);
  return mix(b, c, (t - 0.5) * 2.0);
}
`;

/** Convert planar [3, H, W] to interleaved RGBA float32 for texture upload. */
export function planarToRgba(planar: Float32Array, width: number, height: number): Float32Array {
  const hw = width * height;
  const out = new Float32Array(hw * 4);
  for (let i = 0; i < hw; i++) {
    out[4 * i] = planar[i];
    out[4 * i + 1] = planar[hw + i];
    out[4 * i + 2] = planar[2 * hw + i];
    out[4 * i + 3] = 1;
  }
  return out;
}

export function maxOf(values: Float32Array): number {
  let m = 0;
  for (let i = 0; i < values.length; i++) if (values[i] > m) m = values[i];
  return m;
}

export function flagsOf(o: OverlayState): number {
  return (o.image ? FLAG_IMAGE : 0) | (o.diff ? FLAG_DIFF : 0) | (o.density ? FLAG_DENSITY : 0) | (o.voronoi ? FLAG_VORONOI : 0);
}

/** @webgpu/types requires ArrayBufferView<ArrayBuffer>; typed arrays we create are never shared. */
export type GpuBytes = ArrayBufferView<ArrayBuffer>;
export function gpuBytes(view: ArrayBufferView): GpuBytes {
  return view as GpuBytes;
}
