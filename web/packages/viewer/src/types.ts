/** Viewer contract: scenes, views, overlays, events, and the backend interface. */

export interface SceneState {
  /** Normalized xy, length 2N. */
  xy: Float32Array;
  /** Log-covariance channels [xx, xy, yy], length 3N. Optional when scale/rotation are given. */
  cov?: Float32Array;
  /** Pixel-unit principal scales, length 2N. Derived from cov when absent. */
  scale: Float32Array;
  /** Rotation in radians, length N. Derived from cov when absent. */
  rotation: Float32Array;
  /** Renderer-ready color, length 3N (already weighted and clamped). */
  color: Float32Array;
  /** Planar [3, H, W] rendered image for this state, optional. */
  rendered?: Float32Array;
  label?: string;
}

export interface Scene {
  /** Padded frame that xy refers to. */
  width: number;
  height: number;
  /** Original image size before padding, optional. */
  crop?: { w: number; h: number };
  states: SceneState[];
  density?: Float32Array;
  /** Per-point mass weights, length N, when known. */
  weights?: Float32Array;
  /** Source image, planar [3, H, W] float32 in [0,1], or a bitmap. */
  image?: Float32Array | ImageBitmap;
  meta?: Record<string, unknown>;
}

export interface View {
  /** Display pixels per source pixel. */
  zoom: number;
  /** Translation of the source origin in display pixels. */
  tx: number;
  ty: number;
}

export interface RenderBackend {
  readonly kind: "webgpu" | "webgl2";
  resize(width: number, height: number): void;
  setState(state: SceneState, sourceWidth: number, sourceHeight: number): void;
  render(view: View): void;
  readPixels(): Promise<Uint8ClampedArray>;
  dispose(): void;
}

export type OverlayName = "centers" | "ellipses" | "density" | "voronoi" | "diff" | "image";

export interface ViewerOptions {
  backend?: "auto" | "webgpu" | "webgl2";
  device?: GPUDevice;
  pixelRatio?: number;
}

/** Event payloads: `statechange` after the shown state changes, `viewchange` after pan or zoom, `hover` on pointer moves. */
export interface ViewerEvents {
  statechange: { k: number };
  viewchange: View;
  hover: { display: { x: number; y: number }; source: { x: number; y: number } };
}
export type ViewerEvent = keyof ViewerEvents;

export interface Viewer {
  /** Resolves once the backend is initialized; rejects when neither backend works. */
  readonly ready: Promise<void>;
  /** Kind of backend actually in use, after `ready`. */
  readonly backendKind: "webgpu" | "webgl2" | null;
  setScene(scene: Scene): void;
  setState(k: number): void;
  getState(): number;
  setView(view: View): void;
  getView(): View;
  fit(): void;
  oneToOne(): void;
  /** Options: `gain` for diff, `sigma` (Mahalanobis radius) and `width` (display pixels) for ellipses. */
  setOverlay(name: OverlayName, on: boolean, opts?: Record<string, unknown>): void;
  psnr(k?: number): number | null;
  exportPng(): Promise<Blob>;
  /** Render synchronously now instead of waiting for the next animation frame. */
  renderNow(): Promise<void>;
  /** RGBA8 pixels of the last rendered frame at the canvas size. */
  readPixels(): Promise<Uint8ClampedArray>;
  /** Source-pixel coordinates under a display point, for hit testing. */
  toSource(displayX: number, displayY: number): { x: number; y: number };
  on<E extends ViewerEvent>(event: E, cb: (e: ViewerEvents[E]) => void): () => void;
  dispose(): void;
}
