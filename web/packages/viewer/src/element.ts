/**
 * <gaussifier-viewer src="scene.splat2d"> custom element.
 *
 * Attributes: src (URL of a .splat2d or .npz asset), backend ("auto" | "webgpu" | "webgl2"),
 * state (index into multi-state scenes), controls ("off" disables pan and zoom).
 * Events: "load" with detail { scene }, "error" with detail { error }.
 * Properties: viewer (the Viewer once ready), scene.
 */
import { createViewer } from "./viewer.js";
import { loadNpz } from "./loaders/npz.js";
import { loadSplat2d } from "./loaders/splat2d.js";
import type { Scene, Viewer } from "./types.js";

export async function loadSceneFromUrl(url: string, init?: RequestInit): Promise<Scene> {
  const res = await fetch(url, init);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
  const buf = await res.arrayBuffer();
  return decodeScene(buf, url);
}

/** Detect the format from the bytes first, then the extension. */
export async function decodeScene(buf: ArrayBuffer, hint = ""): Promise<Scene> {
  const head = new Uint8Array(buf, 0, Math.min(4, buf.byteLength));
  const magic = String.fromCharCode(...head);
  if (magic === "GS2D") return loadSplat2d(buf);
  if (magic.startsWith("PK")) return loadNpz(buf);
  if (/\.splat2d(\?|$)/i.test(hint)) return loadSplat2d(buf);
  if (/\.npz(\?|$)/i.test(hint)) return loadNpz(buf);
  throw new Error("unrecognized scene format; expected a .splat2d (GS2D) or .npz file");
}

export class GaussifierViewerElement extends HTMLElement {
  static get observedAttributes(): string[] { return ["src", "state"]; }
  private canvas: HTMLCanvasElement;
  private message: HTMLDivElement;
  private handle: Viewer | null = null;
  private current: Scene | null = null;
  private loadToken = 0;

  constructor() {
    super();
    const root = this.attachShadow({ mode: "open" });
    root.innerHTML = `<style>
      :host { display: block; position: relative; background: #0e1013; min-height: 120px; overflow: hidden; }
      canvas { display: block; width: 100%; height: 100%; outline: none; }
      .msg { position: absolute; inset: 0; display: grid; place-items: center; color: #9aa3b2; font: 13px system-ui, sans-serif; pointer-events: none; text-align: center; padding: 12px; }
      .msg[hidden] { display: none; }
    </style><canvas tabindex="0"></canvas><div class="msg" hidden></div>`;
    this.canvas = root.querySelector("canvas")!;
    this.message = root.querySelector(".msg")!;
  }

  get viewer(): Viewer | null { return this.handle; }
  get scene(): Scene | null { return this.current; }

  connectedCallback(): void {
    if (!this.handle) {
      const backend = (this.getAttribute("backend") ?? "auto") as "auto" | "webgpu" | "webgl2";
      this.handle = createViewer(this.canvas, { backend });
      this.handle.ready.catch((e) => this.fail(e));
    }
    const src = this.getAttribute("src");
    if (src) void this.load(src);
  }

  disconnectedCallback(): void {
    this.handle?.dispose(); this.handle = null; this.current = null;
  }

  attributeChangedCallback(name: string, oldValue: string | null, value: string | null): void {
    if (!this.isConnected || oldValue === value) return;
    if (name === "src" && value) void this.load(value);
    if (name === "state" && value && this.handle && this.current) this.handle.setState(Number(value));
  }

  /** Fetch and display an asset by URL. Resolves with the scene. */
  async load(url: string): Promise<Scene> {
    const token = ++this.loadToken;
    this.note("Loading…");
    try {
      const scene = await loadSceneFromUrl(url);
      if (token !== this.loadToken) return scene;
      return this.show(scene);
    } catch (e) {
      if (token === this.loadToken) this.fail(e);
      throw e;
    }
  }

  /** Display an already decoded scene. */
  async show(scene: Scene): Promise<Scene> {
    if (!this.handle) throw new Error("viewer not connected");
    await this.handle.ready;
    this.current = scene;
    this.handle.setScene(scene);
    const state = this.getAttribute("state");
    this.handle.setState(state ? Number(state) : scene.states.length - 1);
    this.handle.fit();
    this.note(null);
    this.dispatchEvent(new CustomEvent("load", { detail: { scene } }));
    return scene;
  }

  private note(text: string | null): void {
    this.message.hidden = text === null;
    this.message.textContent = text ?? "";
  }

  private fail(e: unknown): void {
    this.note(`Could not display this asset: ${String((e as Error)?.message ?? e)}`);
    this.dispatchEvent(new CustomEvent("error", { detail: { error: e } }));
  }
}

export function defineGaussifierViewer(tag = "gaussifier-viewer"): void {
  if (typeof customElements !== "undefined" && !customElements.get(tag)) customElements.define(tag, GaussifierViewerElement);
}
