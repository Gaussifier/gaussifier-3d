import type { OverlayName, Scene, SceneState, View, Viewer, ViewerEvent, ViewerEvents, ViewerOptions } from "./types.js";
import { psnrPlanar } from "./math.js";
import { buildOwnerMap } from "./voronoi.js";
import { bruteForceRender } from "./reference.js";
import { WebGpuBackend } from "./backends/webgpu.js";
import { WebGl2Backend } from "./backends/webgl2.js";
import type { FullBackend } from "./backends/common.js";

type Listener = (e: unknown) => void;

class ViewerImpl implements Viewer {
  readonly ready: Promise<void>;
  backendKind: "webgpu" | "webgl2" | null = null;
  private backend: FullBackend | null = null;
  private scene: Scene | null = null;
  private k = 0;
  private view: View = { zoom: 1, tx: 0, ty: 0 };
  private viewSet = false;
  private readonly overlays: Record<OverlayName, boolean> = { centers: false, ellipses: false, density: false, voronoi: false, diff: false, image: false };
  private diffGain = 4;
  private ellipseSigma = 1;
  private ellipseWidth = 1.25;
  /** Nearest-center maps per state for the Voronoi overlay, computed on first use. */
  private readonly ownerMaps = new WeakMap<SceneState, Int32Array>();
  private readonly listeners = new Map<ViewerEvent, Set<Listener>>();
  /** What the backend currently holds, so a scene that reuses the same arrays skips the upload. */
  private uploaded: { image: Float32Array | null; density: Float32Array | null; width: number; height: number } | null = null;
  private readonly pixelRatio: number;
  private frame: number | null = null;
  private disposed = false;
  private dragging: { x: number; y: number; tx: number; ty: number } | null = null;
  /** Active pointers by id, for one-finger pan and two-finger pinch on touch screens. */
  private pointers = new Map<number, { x: number; y: number }>();
  private pinch: { distance: number; zoom: number } | null = null;
  private readonly cleanup: Array<() => void> = [];
  private resizeObserver: ResizeObserver | null = null;

  constructor(readonly canvas: HTMLCanvasElement, opts: ViewerOptions) {
    this.pixelRatio = opts.pixelRatio ?? (typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1);
    this.ready = this.init(opts);
    this.attachControls();
  }

  private async init(opts: ViewerOptions): Promise<void> {
    const want = opts.backend ?? "auto";
    let backend: FullBackend | null = null;
    let firstError: unknown = null;
    if (want === "auto" || want === "webgpu") {
      try { backend = await WebGpuBackend.create(this.canvas, opts.device); } catch (e) { firstError = e; if (want === "webgpu") throw e; }
    }
    if (!backend && (want === "auto" || want === "webgl2")) {
      try { backend = new WebGl2Backend(this.canvas); } catch (e) { throw new Error(`no viewer backend: ${String(firstError ?? "")} / ${String(e)}`); }
    }
    if (!backend || this.disposed) { backend?.dispose(); return; }
    this.backend = backend;
    this.backendKind = backend.kind;
    this.syncSize();
    if (typeof ResizeObserver !== "undefined" && this.canvas.isConnected) {
      this.resizeObserver = new ResizeObserver(() => { this.syncSize(); this.requestRender(); });
      this.resizeObserver.observe(this.canvas);
    }
    if (this.scene) this.applyScene();
    this.requestRender();
  }

  /** Match the backing store to the CSS size when the canvas is laid out; otherwise keep the attribute size. */
  private syncSize(): void {
    if (!this.backend) return;
    const cssW = this.canvas.clientWidth, cssH = this.canvas.clientHeight;
    const w = cssW > 0 ? Math.round(cssW * this.pixelRatio) : this.canvas.width;
    const h = cssH > 0 ? Math.round(cssH * this.pixelRatio) : this.canvas.height;
    this.backend.resize(Math.max(1, w), Math.max(1, h));
  }

  private applyScene(): void {
    if (!this.backend || !this.scene) return;
    const s = this.scene;
    this.k = Math.min(this.k, s.states.length - 1);
    this.backend.setState(s.states[this.k], s.width, s.height);
    const image = s.image instanceof Float32Array ? s.image : null;
    const density = s.density ?? null;
    const u = this.uploaded;
    const sameFrame = u !== null && u.width === s.width && u.height === s.height;
    if (!sameFrame || u.image !== image) this.backend.setImage(image, s.width, s.height);
    if (!sameFrame || u.density !== density) this.backend.setDensity(density, s.width, s.height);
    this.uploaded = { image, density, width: s.width, height: s.height };
    this.pushOverlays();
    // Keep the user's pan/zoom across states and re-runs; refit when the frame size changes.
    if (!this.viewSet || !sameFrame) this.fit();
  }

  private pushOverlays(): void {
    if (!this.backend) return;
    this.backend.setOverlays({ ...this.overlays, diffGain: this.diffGain, ellipseSigma: this.ellipseSigma, ellipseWidth: this.ellipseWidth });
    this.syncVoronoi();
  }

  /** Upload the owner map of the current state while the Voronoi overlay is on; drop it otherwise. */
  private syncVoronoi(): void {
    if (!this.backend) return;
    if (!this.overlays.voronoi || !this.scene) { this.backend.setOwners(null, 0, 0); return; }
    const s = this.scene;
    const state = s.states[this.k];
    let owners = this.ownerMaps.get(state);
    if (!owners) {
      owners = buildOwnerMap(state.xy, s.width, s.height);
      this.ownerMaps.set(state, owners);
    }
    this.backend.setOwners(new Uint32Array(owners.buffer, owners.byteOffset, owners.length), s.width, s.height);
  }

  setScene(scene: Scene): void {
    if (scene.states.length === 0) throw new Error("scene has no states");
    this.scene = scene;
    this.k = 0;
    this.applyScene();
    this.requestRender();
    this.emit("statechange", { k: this.k });
  }

  setState(k: number): void {
    if (!this.scene) return;
    const clamped = Math.max(0, Math.min(this.scene.states.length - 1, Math.floor(k)));
    if (clamped === this.k && this.backend) return;
    this.k = clamped;
    this.backend?.setState(this.scene.states[this.k], this.scene.width, this.scene.height);
    if (this.overlays.voronoi) this.syncVoronoi();
    this.requestRender();
    this.emit("statechange", { k: this.k });
  }

  getState(): number {
    return this.k;
  }

  setView(view: View): void {
    this.view = { zoom: Math.max(1e-3, view.zoom), tx: view.tx, ty: view.ty };
    this.viewSet = true;
    this.requestRender();
    this.emit("viewchange", { ...this.view });
  }

  getView(): View {
    return { ...this.view };
  }

  fit(): void {
    if (!this.scene) return;
    const cw = this.canvas.width, ch = this.canvas.height;
    const zoom = Math.min(cw / this.scene.width, ch / this.scene.height);
    this.setView({ zoom, tx: (cw - this.scene.width * zoom) / 2, ty: (ch - this.scene.height * zoom) / 2 });
  }

  oneToOne(): void {
    if (!this.scene) return;
    const cw = this.canvas.width, ch = this.canvas.height;
    this.setView({ zoom: 1, tx: Math.floor((cw - this.scene.width) / 2), ty: Math.floor((ch - this.scene.height) / 2) });
  }

  /** Options: `gain` for diff, `sigma` (Mahalanobis radius) and `width` (display pixels) for ellipses. */
  setOverlay(name: OverlayName, on: boolean, opts?: Record<string, unknown>): void {
    this.overlays[name] = on;
    if (opts && typeof opts.gain === "number") this.diffGain = opts.gain;
    if (opts && typeof opts.sigma === "number") this.ellipseSigma = opts.sigma;
    if (opts && typeof opts.width === "number") this.ellipseWidth = opts.width;
    this.pushOverlays();
    this.requestRender();
  }

  /**
   * PSNR of a state's render against the scene image over the crop, or null without an image.
   * A state without `rendered` planes gets them from the CPU reference renderer on first use
   * (about 100 ms for 84K Gaussians at 512², cached on the state), so a loaded asset can be
   * scored against a source image too. Do not call it per frame on states you replace often.
   */
  psnr(k?: number): number | null {
    if (!this.scene || !(this.scene.image instanceof Float32Array)) return null;
    const state = this.scene.states[k ?? this.k];
    if (!state) return null;
    state.rendered ??= bruteForceRender(state, this.scene.width, this.scene.height);
    const crop = this.scene.crop ?? { w: this.scene.width, h: this.scene.height };
    return psnrPlanar(state.rendered, this.scene.image, this.scene.width, this.scene.height, crop.w, crop.h);
  }

  async exportPng(): Promise<Blob> {
    await this.renderNow();
    const pixels = await this.readPixels();
    const w = this.canvas.width, h = this.canvas.height;
    const off = document.createElement("canvas");
    off.width = w; off.height = h;
    const ctx = off.getContext("2d");
    if (!ctx) throw new Error("2D context unavailable for PNG export");
    ctx.putImageData(new ImageData(pixels as Uint8ClampedArray<ArrayBuffer>, w, h), 0, 0);
    return new Promise((resolve, reject) => off.toBlob((b) => (b ? resolve(b) : reject(new Error("toBlob failed"))), "image/png"));
  }

  on<E extends ViewerEvent>(event: E, cb: (e: ViewerEvents[E]) => void): () => void {
    let set = this.listeners.get(event);
    if (!set) { set = new Set(); this.listeners.set(event, set); }
    set.add(cb as Listener);
    return () => { set!.delete(cb as Listener); };
  }

  private emit<E extends ViewerEvent>(event: E, payload: ViewerEvents[E]): void {
    this.listeners.get(event)?.forEach((cb) => { try { cb(payload); } catch { /* listener errors do not break rendering */ } });
  }

  toSource(displayX: number, displayY: number): { x: number; y: number } {
    return { x: (displayX - this.view.tx) / this.view.zoom - 0.5, y: (displayY - this.view.ty) / this.view.zoom - 0.5 };
  }

  private requestRender(): void {
    if (this.frame !== null || this.disposed) return;
    const raf = typeof requestAnimationFrame === "function" ? requestAnimationFrame : (cb: FrameRequestCallback) => setTimeout(() => cb(performance.now()), 16) as unknown as number;
    this.frame = raf(() => { this.frame = null; this.renderFrame(); });
  }

  private renderFrame(): void {
    if (!this.backend || !this.scene) return;
    this.backend.render(this.view);
  }

  async renderNow(): Promise<void> {
    await this.ready;
    if (this.frame !== null && typeof cancelAnimationFrame === "function") { cancelAnimationFrame(this.frame); this.frame = null; }
    this.renderFrame();
  }

  async readPixels(): Promise<Uint8ClampedArray> {
    await this.ready;
    if (!this.backend) return new Uint8ClampedArray(0);
    return this.backend.readPixels();
  }

  private displayPoint(e: MouseEvent): { x: number; y: number } {
    const rect = this.canvas.getBoundingClientRect();
    const sx = rect.width > 0 ? this.canvas.width / rect.width : 1;
    const sy = rect.height > 0 ? this.canvas.height / rect.height : 1;
    return { x: (e.clientX - rect.left) * sx, y: (e.clientY - rect.top) * sy };
  }

  private attachControls(): void {
    const c = this.canvas;
    if (typeof c.addEventListener !== "function") return;
    if (!c.hasAttribute("tabindex")) c.tabIndex = 0;
    // Zoom about a display point, keeping that point fixed on screen.
    const zoomAt = (p: { x: number; y: number }, zoom: number) => {
      zoom = Math.max(0.05, Math.min(64, zoom));
      const ratio = zoom / this.view.zoom;
      this.setView({ zoom, tx: p.x - (p.x - this.view.tx) * ratio, ty: p.y - (p.y - this.view.ty) * ratio });
    };
    const pinchState = () => {
      const [a, b] = [...this.pointers.values()];
      return { centre: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }, distance: Math.hypot(a.x - b.x, a.y - b.y) };
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      zoomAt(this.displayPoint(e), this.view.zoom * Math.pow(1.1, -e.deltaY / 100));
    };
    const onDown = (e: PointerEvent) => {
      const p = this.displayPoint(e);
      this.pointers.set(e.pointerId, p);
      c.setPointerCapture?.(e.pointerId);
      if (this.pointers.size === 2) {
        this.dragging = null;
        this.pinch = { distance: pinchState().distance, zoom: this.view.zoom };
      } else {
        this.dragging = { x: p.x, y: p.y, tx: this.view.tx, ty: this.view.ty };
      }
    };
    const onMove = (e: PointerEvent) => {
      const p = this.displayPoint(e);
      if (this.pointers.has(e.pointerId)) this.pointers.set(e.pointerId, p);
      if (this.pinch && this.pointers.size >= 2) {
        const { centre, distance } = pinchState();
        zoomAt(centre, this.pinch.zoom * (distance / Math.max(1, this.pinch.distance)));
      } else if (this.dragging) {
        this.setView({ zoom: this.view.zoom, tx: this.dragging.tx + (p.x - this.dragging.x), ty: this.dragging.ty + (p.y - this.dragging.y) });
      } else {
        this.emit("hover", { display: p, source: this.toSource(p.x, p.y) });
      }
    };
    const onUp = (e: PointerEvent) => {
      this.pointers.delete(e.pointerId);
      c.releasePointerCapture?.(e.pointerId);
      this.dragging = null;
      this.pinch = null;
      // Lifting one finger of a pinch continues as a pan from the remaining one.
      const rest = [...this.pointers.values()];
      if (rest.length === 1) this.dragging = { x: rest[0].x, y: rest[0].y, tx: this.view.tx, ty: this.view.ty };
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "[") { this.setState(this.k - 1); e.preventDefault(); }
      else if (e.key === "]") { this.setState(this.k + 1); e.preventDefault(); }
      else if (e.key === "1") { this.oneToOne(); e.preventDefault(); }
      else if (e.key === "f" || e.key === "F") { this.fit(); e.preventDefault(); }
    };
    c.addEventListener("wheel", onWheel, { passive: false });
    c.addEventListener("pointerdown", onDown);
    c.addEventListener("pointermove", onMove);
    c.addEventListener("pointerup", onUp);
    c.addEventListener("pointercancel", onUp);
    c.addEventListener("keydown", onKey);
    this.cleanup.push(() => {
      c.removeEventListener("wheel", onWheel);
      c.removeEventListener("pointerdown", onDown);
      c.removeEventListener("pointermove", onMove);
      c.removeEventListener("pointerup", onUp);
      c.removeEventListener("pointercancel", onUp);
      c.removeEventListener("keydown", onKey);
    });
  }

  dispose(): void {
    this.disposed = true;
    if (this.frame !== null && typeof cancelAnimationFrame === "function") cancelAnimationFrame(this.frame);
    this.frame = null;
    this.resizeObserver?.disconnect();
    for (const fn of this.cleanup) fn();
    this.backend?.dispose();
    this.backend = null;
  }
}

/** Create a viewer on a canvas. Backend initialization is asynchronous; await `viewer.ready` before reading pixels. */
export function createViewer(canvas: HTMLCanvasElement, opts: ViewerOptions = {}): Viewer {
  return new ViewerImpl(canvas, opts);
}
