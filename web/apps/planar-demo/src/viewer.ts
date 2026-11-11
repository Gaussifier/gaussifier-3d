/**
 * PlayCanvas view of the model's Gaussians with an orbit camera.
 *
 * The view shows splat files (.ply, .sog) the way 3DGS viewers such as SuperSplat do: the splat
 * entity is turned 180 degrees about z, the load-time convention for these formats. The page
 * writes its files in that frame (see exporters.ts), so the image stands upright and faces the
 * camera, which starts at +distance on z looking at the origin: the model's frontal view.
 */
import * as pc from "playcanvas";

export interface ViewSettings {
  /** Vertical field of view of the model's frontal camera, in degrees. */
  fovDeg: number;
  /** Distance of the frontal camera from the image plane. */
  distance: number;
}

const DEG = Math.PI / 180;

export class SplatView {
  autoRotate = false;
  private yaw = 0;
  private pitch = 0;
  private distance: number;
  private readonly target = new pc.Vec3();
  /** Loaded splats by key (one per shown state and mode), so scrubbing back and forth is instant. */
  private readonly cache = new Map<string, { entity: pc.Entity; asset: pc.Asset; url: string }>();
  private shown: string | null = null;

  private constructor(
    private readonly app: pc.AppBase,
    private readonly camera: pc.Entity,
    private readonly root: pc.Entity,
    private readonly settings: ViewSettings,
    private readonly canvas: HTMLCanvasElement,
  ) {
    this.distance = settings.distance;
    this.bindInput();
    app.on("update", (dt: number) => {
      if (this.autoRotate && !this.dragging) this.yaw += dt * 20;
      this.applyCamera();
    });
  }

  static async create(canvas: HTMLCanvasElement, settings: ViewSettings): Promise<SplatView> {
    const device = await pc.createGraphicsDevice(canvas, { deviceTypes: ["webgl2"], antialias: false, alpha: false });
    device.maxPixelRatio = Math.min(window.devicePixelRatio, 2);
    const options = new pc.AppOptions();
    options.graphicsDevice = device;
    options.componentSystems = [pc.CameraComponentSystem, pc.GSplatComponentSystem];
    // SOG files carry WebP textures, which load as texture assets.
    options.resourceHandlers = [pc.GSplatHandler, pc.TextureHandler];
    const app = new pc.AppBase(canvas);
    app.init(options);
    app.setCanvasFillMode(pc.FILLMODE_NONE);
    app.setCanvasResolution(pc.RESOLUTION_AUTO);
    const resize = () => app.resizeCanvas(canvas.clientWidth, canvas.clientHeight);
    new ResizeObserver(resize).observe(canvas);
    resize();
    const camera = new pc.Entity("camera");
    camera.addComponent("camera", {
      clearColor: new pc.Color(0.055, 0.063, 0.075),
      fov: settings.fovDeg,
      nearClip: 0.005,
      farClip: 50,
    });
    app.root.addChild(camera);
    const root = new pc.Entity("splats");
    root.setLocalEulerAngles(0, 0, 180);
    app.root.addChild(root);
    app.start();
    const view = new SplatView(app, camera, root, settings, canvas);
    view.applyCamera();
    return view;
  }

  /**
   * Show the splats under `key`, loading them from `make()` (a .ply or .sog file's bytes, named by
   * `filename`) the first time. Other cached splats are hidden, not unloaded.
   */
  async show(key: string, make: () => ArrayBuffer | Uint8Array | Promise<ArrayBuffer | Uint8Array>, filename = "scene.ply"): Promise<void> {
    let item = this.cache.get(key);
    if (!item) {
      const bytes = await make();
      const url = URL.createObjectURL(new Blob([bytes as BlobPart], { type: "application/octet-stream" }));
      // PlayCanvas picks the parser by the URL's extension; a blob URL has none, so the filename
      // rides along as a fragment (fetch ignores it).
      const asset = new pc.Asset(key, "gsplat", { url: `${url}#${filename}`, filename });
      this.app.assets.add(asset);
      await new Promise<void>((resolve, reject) => {
        asset.once("load", () => resolve());
        asset.once("error", (err: unknown) => reject(new Error(String(err))));
        this.app.assets.load(asset);
      });
      const entity = new pc.Entity(key);
      entity.addComponent("gsplat", { asset });
      entity.enabled = false;
      this.root.addChild(entity);
      item = { entity, asset, url };
      this.cache.set(key, item);
    }
    for (const [k, v] of this.cache) v.entity.enabled = k === key;
    this.shown = key;
    this.app.renderNextFrame = true;
  }

  /** Drop every cached splat set (a new run or file replaces them all). */
  clear(): void {
    for (const { entity, asset, url } of this.cache.values()) {
      entity.destroy();
      asset.unload();
      this.app.assets.remove(asset);
      URL.revokeObjectURL(url);
    }
    this.cache.clear();
    this.shown = null;
  }

  /**
   * Stop or resume drawing. Inference waits on GPU readbacks, and a page that also redraws its
   * canvas every frame stretches each of those waits to the next frame, so runs pause the view.
   */
  setRendering(on: boolean): void {
    this.app.autoRender = on;
    if (on) this.app.renderNextFrame = true;
  }

  /** Width over height of the shown image (the slab spans height 1 and this width). */
  private imageAspect = 1;
  setImageAspect(aspect: number): void { this.imageAspect = aspect > 0 ? aspect : 1; }

  /** How much farther than the model's frontal camera the view stands so the whole image fits. */
  fitScale(): number {
    const stage = this.canvas.clientWidth / Math.max(1, this.canvas.clientHeight);
    return Math.max(1, this.imageAspect / Math.max(stage, 1e-3));
  }

  /** Back to the model's frontal camera, standing back as far as the image's width needs. */
  frontal(): void {
    this.yaw = 0;
    this.pitch = 0;
    this.distance = this.settings.distance * this.fitScale();
    this.target.set(0, 0, 0);
  }

  private applyCamera(): void {
    const cp = Math.cos(this.pitch * DEG), sp = Math.sin(this.pitch * DEG);
    const cy = Math.cos(this.yaw * DEG), sy = Math.sin(this.yaw * DEG);
    const offset = new pc.Vec3(this.distance * cp * sy, this.distance * sp, this.distance * cp * cy);
    this.camera.setPosition(this.target.x + offset.x, this.target.y + offset.y, this.target.z + offset.z);
    this.camera.lookAt(this.target, pc.Vec3.UP);
  }

  private dragging = false;
  private bindInput(): void {
    const c = this.canvas;
    let mode: "orbit" | "pan" | null = null;
    let lastX = 0, lastY = 0;
    const pointers = new Map<number, { x: number; y: number }>();
    let pinch = 0;
    c.addEventListener("contextmenu", (e) => e.preventDefault());
    c.addEventListener("pointerdown", (e) => {
      c.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      mode = e.button === 2 || e.shiftKey || pointers.size === 2 ? "pan" : "orbit";
      lastX = e.clientX; lastY = e.clientY;
      this.dragging = true;
      if (pointers.size === 2) { const [a, b] = [...pointers.values()]; pinch = Math.hypot(a.x - b.x, a.y - b.y); }
    });
    c.addEventListener("pointermove", (e) => {
      if (!mode) return;
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (pinch > 0) this.zoom(pinch / d);
        pinch = d;
      }
      const dx = e.clientX - lastX, dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      if (mode === "orbit") {
        this.yaw -= dx * 0.3;
        this.pitch = Math.max(-89, Math.min(89, this.pitch + dy * 0.3));
      } else {
        const scale = (2 * this.distance * Math.tan((this.settings.fovDeg * DEG) / 2)) / Math.max(1, c.clientHeight);
        const right = this.camera.right.clone().mulScalar(-dx * scale);
        const up = this.camera.up.clone().mulScalar(dy * scale);
        this.target.add(right).add(up);
      }
    });
    const end = (e: PointerEvent) => {
      pointers.delete(e.pointerId);
      if (pointers.size === 0) { mode = null; this.dragging = false; pinch = 0; }
    };
    c.addEventListener("pointerup", end);
    c.addEventListener("pointercancel", end);
    c.addEventListener("wheel", (e) => { e.preventDefault(); this.zoom(Math.exp(e.deltaY * 0.001)); }, { passive: false });
    c.addEventListener("dblclick", () => this.frontal());
  }

  private zoom(factor: number): void {
    this.distance = Math.max(0.05, Math.min(10, this.distance * factor));
  }
}
