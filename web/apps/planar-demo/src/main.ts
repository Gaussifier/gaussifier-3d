import "./style.css";
import type { PlanarEngine, PlanarRunResult, PlanarSplats } from "@gaussifier/planar";
import { SplatView } from "./viewer.js";
import { centersOf, download, splatsToPly, splatsToSog } from "./exporters.js";
import { getEngine, hasModel, modelSize, prepareImage, type ModelKind, type PreparedImage } from "./run.js";

// ---- elements ----
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const ui = {
  canvas: $<HTMLCanvasElement>("view"),
  overlay: $<HTMLCanvasElement>("overlay"),
  stage: $<HTMLElement>("stage"),
  empty: $<HTMLDivElement>("empty"),
  live: $<HTMLDivElement>("live"),
  gpuPill: $<HTMLSpanElement>("gpuPill"),
  runStatus: $<HTMLDivElement>("runStatus"),
  status: $<HTMLDivElement>("status"),
  imageDrop: $<HTMLDivElement>("imageDrop"),
  imageInput: $<HTMLInputElement>("image"),
  thumb: $<HTMLImageElement>("thumb"),
  imageName: $<HTMLDivElement>("imageName"),
  imageDims: $<HTMLDivElement>("imageDims"),
  progress: $<HTMLDivElement>("progress"),
  samples: $<HTMLDivElement>("samples"),
  modelFast: $<HTMLButtonElement>("modelFast"),
  modelQuality: $<HTMLButtonElement>("modelQuality"),
  refine: $<HTMLInputElement>("refine"),
  refineValue: $<HTMLSpanElement>("refineValue"),
  count: $<HTMLInputElement>("count"),
  countValue: $<HTMLSpanElement>("countValue"),
  statN: $<HTMLDivElement>("statN"),
  psnr: $<HTMLSpanElement>("psnr"),
  shownRow: $<HTMLDivElement>("shownRow"),
  shown: $<HTMLInputElement>("shown"),
  shownValue: $<HTMLSpanElement>("shownValue"),
  autoRotate: $<HTMLInputElement>("autoRotate"),
  centers: $<HTMLInputElement>("ov-centers"),
  exportSog: $<HTMLButtonElement>("exportSog"),
  exportPly: $<HTMLButtonElement>("exportPly"),
  drop: $<HTMLDivElement>("drop"),
  pick: $<HTMLButtonElement>("pick"),
  file: $<HTMLInputElement>("file"),
};

// ---- state ----
/** The frontal camera of the planar model: image height 1 at distance 1, focal 512 px for 512 px. */
const FRONTAL = { fovDeg: (2 * Math.atan(0.5) * 180) / Math.PI, distance: 1 };
interface ShownState { label: string; splats?: PlanarSplats; render?: Float32Array; psnr?: number }
interface Scene {
  id: number;
  name: string;
  /** The model behind a run, or null for a saved file. */
  kind: ModelKind | null;
  states: ShownState[];
  count: number | null;
  image: Float32Array | null;
  density: Float32Array | null;
  width: number;
  height: number;
  /** A saved file: its bytes, shown as they are. */
  file?: { bytes: ArrayBuffer; filename: string };
}
let model: ModelKind = new URLSearchParams(location.search).get("model") === "quality" ? "quality" : "fast";
let scene: Scene | null = null;
let sceneIds = 0;
let imageFile: File | null = null;
let prepared: { file: File; input: PreparedImage } | null = null;
let autoCount: number | null = null;
let running = false;
let queued = false;
let lastRun: { kind: ModelKind; run: PlanarRunResult; name: string } | null = null;
const sogCache = new Map<string, Uint8Array>();
const fmtInt = (n: number) => n.toLocaleString("en-US");
const NEEDS_WEBGPU = "This page needs WebGPU: a current Chrome or Edge, Safari 26, or Firefox on Windows, with a GPU.";

// ---- view ----
const viewReady = SplatView.create(ui.canvas, FRONTAL);
let view: SplatView | null = null;
viewReady.then((v) => { view = v; v.autoRotate = ui.autoRotate.checked; }).catch((e) => showStatus(`The 3D view failed to start: ${String(e)}`));

// ---- engine and GPU status ----
// The engine of the selected model loads as soon as the page opens (14 MB of model files plus the
// ONNX Runtime, then shader compilation), so the first run does not pay for it.
const ready = new Set<ModelKind>();
let engineError: string | null = null;
let adapterName = "WebGPU";
type GpuSpeed = "fast" | "slow" | "software" | "unknown";
function gpuSpeed(adapter: string): GpuSpeed {
  const a = adapter.toLowerCase();
  if (/swiftshader|llvmpipe|lavapipe|software|\bwarp\b|cpu/.test(a)) return "software";
  if (/nvidia|amd|radeon|apple|qualcomm|adreno|\barm\b|mali/.test(a)) return "fast";
  if (/intel|integrated|iris|uhd|gen-?\d/.test(a)) return "slow";
  return "unknown";
}
const SPEED_NOTE: Record<GpuSpeed, string> = {
  fast: "",
  slow: "Runs take a few seconds on an integrated GPU.",
  software: "No GPU: this is a software renderer, runs take minutes.",
  unknown: "",
};
let speed: GpuSpeed = "unknown";
function setPill(text: string, css: "on" | "warn" | "bad", title: string): void {
  ui.gpuPill.textContent = text;
  ui.gpuPill.classList.remove("on", "warn", "bad");
  ui.gpuPill.classList.add(css);
  ui.gpuPill.title = title;
}
function describeGpu(engine: PlanarEngine): void {
  const info = (engine.device as GPUDevice & { adapterInfo?: GPUAdapterInfo }).adapterInfo;
  const parts = info ? [info.description, info.device, info.architecture, info.vendor].filter((p) => !!p) : [];
  adapterName = parts.length ? [...new Set(parts)].join(" · ") : "WebGPU";
  speed = gpuSpeed(adapterName);
  const vendor = adapterName.split(" · ").pop() ?? "";
  const detail = `${adapterName}. Runs take well under a second on a discrete GPU, a few seconds on an integrated one, and minutes on a software renderer.`;
  if (speed === "fast") setPill(`${vendor} gpu`, "on", detail);
  else if (speed === "slow") setPill("integrated gpu", "warn", detail);
  else if (speed === "software") setPill("no gpu · slow", "bad", detail);
  else setPill("gpu", "on", detail);
}
function updateReady(): void {
  if (running) return;
  if (engineError) { ui.runStatus.textContent = engineError; return; }
  if (!ready.has(model)) { ui.runStatus.textContent = imageFile ? "Loading the model…" : "Loading the model and the sample image…"; return; }
  const note = SPEED_NOTE[speed];
  ui.runStatus.textContent = (imageFile ? "Ready." : "Pick an image to start.") + (note ? ` ${note}` : "");
}
/** The loading bar: a fraction of the download, null while the model is prepared, undefined to hide. */
function setLoading(fraction: number | null | undefined, text = ""): void {
  ui.progress.hidden = fraction === undefined;
  ui.progress.classList.toggle("indeterminate", fraction === null);
  (ui.progress.firstElementChild as HTMLDivElement).style.width = fraction === null || fraction === undefined ? "" : `${(100 * fraction).toFixed(1)}%`;
  if (typeof fraction === "number") ui.progress.setAttribute("aria-valuenow", String(Math.round(100 * fraction)));
  else ui.progress.removeAttribute("aria-valuenow");
  if (text) ui.runStatus.textContent = text;
}
const MB = (b: number) => (b / 1e6).toFixed(0);
function loadEngine(kind: ModelKind): Promise<PlanarEngine> {
  const loading = !ready.has(kind);
  return getEngine(kind, (fraction, loaded, total) => {
    if (!loading || ready.has(kind)) return;
    if (fraction === null) setLoading(null, "Preparing the model…");
    else setLoading(fraction, `Loading the model… ${Math.round(100 * fraction)}% (${MB(loaded)} of ${MB(total)} MB)`);
  }).then((engine) => {
    ready.add(kind);
    setLoading(undefined);
    describeGpu(engine);
    updateReady();
    return engine;
  }).catch((e) => {
    setLoading(undefined);
    engineError = /WebGPU/i.test(String(e)) ? NEEDS_WEBGPU : `${NEEDS_WEBGPU} The engine failed to load: ${String(e)}`;
    setPill("no webgpu", "bad", NEEDS_WEBGPU);
    updateReady();
    throw e;
  });
}
if (!("gpu" in navigator)) {
  engineError = NEEDS_WEBGPU;
  setPill("no webgpu", "bad", NEEDS_WEBGPU);
} else {
  void loadEngine(model).catch(() => {});
}

function showStatus(text: string): void { ui.status.hidden = false; ui.status.textContent = text; }

// ---- showing a scene ----
function shortLabel(label: string | undefined, index: number): string {
  return (label ?? String(index)).replace(/^refinement /, "");
}
function showScene(next: Scene): void {
  // The initializer is not interesting to look at: drop it whenever refinements follow.
  if (next.states.length > 1 && next.states[0].label === "initializer") next.states = next.states.slice(1);
  const newImage = !scene || scene.name !== next.name || scene.width !== next.width || scene.height !== next.height;
  scene = next;
  sogCache.clear();
  view?.clear();
  if (view && next.width > 0) {
    view.setImageAspect(next.width / next.height);
    if (newImage) view.frontal();
  }
  ui.empty.hidden = true;
  const last = next.states.length - 1;
  ui.shownRow.hidden = next.states.length <= 1;
  ui.shown.max = String(last);
  ui.shown.value = String(last);
  ui.statN.textContent = next.count === null ? "—" : fmtInt(next.count);
  ui.exportSog.disabled = ui.exportPly.disabled = !next.states[last].splats;
  ui.status.hidden = true;
  void showState(last);
}
let showToken = 0;
async function showState(index: number): Promise<void> {
  if (!scene || !view) return;
  const s = scene.states[index];
  if (!s) return;
  ui.shown.value = String(index);
  ui.shownValue.textContent = shortLabel(s.label, index);
  ui.psnr.innerHTML = s.psnr === undefined ? "—" : `${s.psnr.toFixed(2)}<small>dB</small>`;
  const token = ++showToken;
  const centers = ui.centers.checked && !!s.splats;
  const key = `${scene.id}:${index}:${centers ? "centers" : "splats"}`;
  if (scene.file) await view.show(key, () => scene!.file!.bytes, scene.file.filename);
  else if (s.splats) await view.show(key, () => splatsToPly(centers ? centersOf(s.splats!) : s.splats!), "scene.ply");
  if (token === showToken) drawOverlay();
}

// ---- running ----
// There is no button: picking an image or changing a control runs, and a change made during a
// run queues one more run with the latest settings.
const refinements = () => Math.max(0, Number(ui.refine.value) || 0);
/** The count slider is logarithmic, -100..100 for x1/4..x4, with a dead zone at the centre that means "auto". */
function countScale(): number {
  const v = Number(ui.count.value);
  return Math.abs(v) <= 3 ? 1 : Math.pow(2, v / 50);
}
async function preparedFor(file: File): Promise<PreparedImage> {
  if (prepared?.file === file) return prepared.input;
  const input = await prepareImage(file);
  prepared = { file, input };
  return input;
}
async function run(): Promise<void> {
  if (!imageFile) return;
  if (engineError) { updateReady(); return; }
  if (running) { queued = true; return; }
  const states = refinements() + 1;
  const kind = model;
  running = true;
  ui.runStatus.textContent = ready.has(kind) ? "Running…" : "Loading the model, then running…";
  ui.live.hidden = false;
  ui.live.textContent = states > 1 ? `running ${states - 1} refinements` : "running the initializer";
  const name = imageFile.name;
  try {
    const input = await preparedFor(imageFile);
    const engine = await loadEngine(kind);
    const v = await viewReady;
    // Inference waits on GPU readbacks; a canvas redrawn every frame would stretch each wait.
    v.setRendering(false);
    let result: PlanarRunResult;
    try {
      result = await engine.run({ chw: input.chw, width: input.width, height: input.height }, { states, countScale: countScale(), keepStates: true });
    } finally {
      v.setRendering(true);
    }
    lastRun = { kind, run: result, name };
    showScene({
      id: ++sceneIds, name, kind,
      states: result.states.map((st) => ({ label: st.k === 0 ? "initializer" : `refinement ${st.k}`, splats: st.splats, render: st.render, psnr: st.psnr })),
      count: result.count, image: input.chw, density: result.density, width: result.width, height: result.height,
    });
    autoCount = result.adaptiveCount;
    syncCountLabel();
    ui.runStatus.textContent = `Done on ${adapterName}.`;
  } catch (e) {
    ui.runStatus.textContent = engineError ?? `Run failed: ${String(e)}`;
  } finally {
    running = false;
    ui.live.hidden = true;
    ui.runStatus.dataset.runs = String(Number(ui.runStatus.dataset.runs ?? 0) + 1);
    if (queued) { queued = false; void run(); }
  }
}

// ---- controls: model, refinements and count ----
function selectModel(kind: ModelKind): void {
  if (model === kind) return;
  model = kind;
  ui.modelFast.setAttribute("aria-pressed", String(kind === "fast"));
  ui.modelQuality.setAttribute("aria-pressed", String(kind === "quality"));
  if (!ready.has(kind)) void loadEngine(kind).catch(() => {});
  void run();
}
ui.modelFast.addEventListener("click", () => selectModel("fast"));
ui.modelQuality.addEventListener("click", () => selectModel("quality"));
ui.modelFast.setAttribute("aria-pressed", String(model === "fast"));
ui.modelQuality.setAttribute("aria-pressed", String(model === "quality"));
void hasModel("quality").then((ok) => {
  ui.modelQuality.disabled = !ok;
  if (!ok) ui.modelQuality.title = "The quality model is not part of this build.";
});
ui.refine.addEventListener("input", () => { ui.refineValue.textContent = ui.refine.value; });
ui.refine.addEventListener("change", () => void run());
/** The Gaussians tile shows the count that resulted; the slider shows the multiplier, "auto" at the centre. */
function syncCountLabel(): void {
  const s = countScale();
  ui.countValue.textContent = s === 1 ? "auto" : `×${s.toFixed(2)}`;
  ui.count.title = autoCount ? `${fmtInt(autoCount)} at auto` : "";
}
ui.count.addEventListener("input", syncCountLabel);
ui.count.addEventListener("change", () => { if (countScale() === 1) ui.count.value = "0"; syncCountLabel(); void run(); });

// ---- controls: the source image ----
function setImage(file: File): void {
  imageFile = file;
  const dt = new DataTransfer(); dt.items.add(file); ui.imageInput.files = dt.files;
  const url = URL.createObjectURL(file);
  ui.thumb.onload = () => {
    const w = ui.thumb.naturalWidth, h = ui.thumb.naturalHeight;
    const { width: mw, height: mh } = modelSize(w, h);
    const note = mw === w && mh === h ? "" : ` · runs at ${mw}×${mh}`;
    ui.imageDims.textContent = `${w}×${h} · ${(file.size / 1024).toFixed(0)} KB${note}`;
  };
  ui.thumb.src = url;
  ui.imageName.textContent = file.name;
  updateReady();
  void run();
}
ui.imageDrop.addEventListener("click", () => ui.imageInput.click());
ui.imageInput.addEventListener("change", () => { const f = ui.imageInput.files?.[0]; if (f) setImage(f); });
// Drop an image anywhere on the stage to run it; other files open as saved scenes.
ui.stage.addEventListener("dragover", (e) => { e.preventDefault(); ui.stage.classList.add("over"); });
ui.stage.addEventListener("dragleave", () => ui.stage.classList.remove("over"));
ui.stage.addEventListener("drop", (e) => {
  e.preventDefault(); ui.stage.classList.remove("over");
  const f = e.dataTransfer?.files?.[0];
  if (!f) return;
  if (f.type.startsWith("image/")) setImage(f);
  else void loadFiles([f]);
});

// ---- controls: Kodak samples, a strip of 24 thumbnails; one at random on page open, or ?sample=kodim07 ----
const KODAK = Array.from({ length: 24 }, (_, i) => `kodim${String(i + 1).padStart(2, "0")}`);
for (const name of KODAK) {
  const b = document.createElement("button");
  b.type = "button"; b.dataset.sample = name; b.title = name; b.setAttribute("role", "option"); b.setAttribute("aria-pressed", "false");
  const img = document.createElement("img"); img.src = `samples/kodak/thumbs/${name}.jpg`; img.alt = name; img.loading = "lazy";
  b.append(img);
  b.addEventListener("click", () => void loadSample(name));
  ui.samples.append(b);
}
async function loadSample(name: string): Promise<void> {
  for (const b of ui.samples.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.sample === name));
  try {
    const blob = await (await fetch(new URL(`samples/kodak/${name}.png`, document.baseURI).href)).blob();
    setImage(new File([blob], `${name}.png`, { type: "image/png" }));
    ui.samples.querySelector(`button[data-sample="${name}"]`)?.scrollIntoView({ block: "nearest", inline: "center" });
  } catch (e) {
    showStatus(`Could not load ${name}: ${String(e)}`); updateReady();
  }
}
const requested = new URLSearchParams(location.search).get("sample");
if (!engineError) void loadSample(requested && KODAK.includes(requested) ? requested : KODAK[Math.floor(Math.random() * KODAK.length)]);
else ui.runStatus.textContent = engineError;

// ---- saved scenes ----
async function loadFiles(files: FileList | File[]): Promise<void> {
  const f = Array.from(files).find((x) => /\.(sog|ply)$/i.test(x.name));
  if (!f) { showStatus("Drop a .sog or a .ply file."); return; }
  try {
    await viewReady;
    showScene({ id: ++sceneIds, name: f.name, kind: null, states: [{ label: f.name }], count: null, image: null, density: null, width: 0, height: 0, file: { bytes: await f.arrayBuffer(), filename: f.name } });
    ui.runStatus.textContent = `Showing ${f.name}.`;
  } catch (e) {
    showStatus(`Load failed: ${String(e)}`);
  }
}
ui.drop.addEventListener("dragover", (e) => { e.preventDefault(); e.stopPropagation(); ui.drop.classList.add("over"); });
ui.drop.addEventListener("dragleave", () => ui.drop.classList.remove("over"));
ui.drop.addEventListener("drop", (e) => { e.preventDefault(); e.stopPropagation(); ui.drop.classList.remove("over"); if (e.dataTransfer?.files) void loadFiles(e.dataTransfer.files); });
ui.pick.addEventListener("click", () => ui.file.click());
ui.file.addEventListener("change", () => { if (ui.file.files) void loadFiles(ui.file.files); });

// ---- view controls: the shown refinement, overlays, export ----
ui.shown.addEventListener("input", () => void showState(Number(ui.shown.value)));
const overlayInputs = {
  image: $<HTMLInputElement>("ov-image"),
  diff: $<HTMLInputElement>("ov-diff"),
  density: $<HTMLInputElement>("ov-density"),
};
for (const input of Object.values(overlayInputs)) {
  input.addEventListener("change", () => { if (input.checked) view?.frontal(); drawOverlay(); });
}
ui.centers.addEventListener("change", () => { if (scene) void showState(Number(ui.shown.value)); });
ui.autoRotate.addEventListener("change", () => { if (view) view.autoRotate = ui.autoRotate.checked; });
window.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement && e.target.type !== "range" && e.target.type !== "checkbox") return;
  if (e.key === "f") view?.frontal();
  if ((e.key === "[" || e.key === "]") && scene && scene.states.length > 1) {
    const next = Math.min(scene.states.length - 1, Math.max(0, Number(ui.shown.value) + (e.key === "]" ? 1 : -1)));
    void showState(next);
  }
});
new ResizeObserver(() => drawOverlay()).observe(ui.stage);

/**
 * The 2D overlays of the 2D demo, in the frontal framing (the image spans the stage height, as in
 * the model's frontal view): the source image, |render - image| x 4 of the shown state, or the
 * density map. Like the 2D demo, density wins over the difference, which wins over the image.
 */
function drawOverlay(): void {
  const mode = overlayInputs.density.checked ? "density" : overlayInputs.diff.checked ? "diff" : overlayInputs.image.checked ? "image" : null;
  const s = scene?.states[Number(ui.shown.value)];
  const data = !scene || !mode ? null
    : mode === "image" ? scene.image
    : mode === "diff" ? (scene.image && s?.render ? scene.image : null)
    : scene.density;
  if (!scene || !mode || !data) { ui.overlay.hidden = true; return; }
  const iw = scene.width, ih = scene.height;
  const hw = iw * ih;
  const px = new ImageData(iw, ih);
  const out = px.data;
  if (mode === "density") {
    const d = scene.density!;
    let max = 1e-6;
    for (let i = 0; i < hw; i++) if (d[i] > max) max = d[i];
    const a = [0.05, 0.03, 0.2], b = [0.2, 0.5, 0.9], c = [1.0, 0.9, 0.2];
    for (let i = 0; i < hw; i++) {
      const t = Math.min(1, Math.max(0, d[i] / max));
      const [p, q, u] = t < 0.5 ? [a, b, t * 2] : [b, c, (t - 0.5) * 2];
      for (let ch = 0; ch < 3; ch++) out[4 * i + ch] = 255 * (p[ch] + (q[ch] - p[ch]) * u);
      out[4 * i + 3] = 255;
    }
  } else {
    const im = scene.image!;
    const r = s?.render;
    for (let i = 0; i < hw; i++) {
      for (let ch = 0; ch < 3; ch++) {
        const v = im[ch * hw + i];
        const value = mode === "image" ? v : Math.min(1, Math.abs(Math.round(Math.min(1, Math.max(0, r![ch * hw + i])) * 255) / 255 - v) * 4);
        out[4 * i + ch] = Math.round(value * 255);
      }
      out[4 * i + 3] = 255;
    }
  }
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const W = Math.max(1, Math.round(ui.stage.clientWidth * dpr)), H = Math.max(1, Math.round(ui.stage.clientHeight * dpr));
  ui.overlay.width = W; ui.overlay.height = H;
  const ctx = ui.overlay.getContext("2d");
  if (!ctx) return;
  ctx.fillStyle = "rgb(31, 31, 33)";
  ctx.fillRect(0, 0, W, H);
  const tile = new OffscreenCanvas(iw, ih);
  tile.getContext("2d")!.putImageData(px, 0, 0);
  ctx.imageSmoothingEnabled = false;
  // The fitted frontal view: the image spans the stage height, or its width when it is wider.
  const k = Math.max(1, iw / ih / (W / H));
  const dh = Math.round(H / k), dw = Math.round((H * iw) / ih / k);
  ctx.drawImage(tile, Math.round((W - dw) / 2), Math.round((H - dh) / 2), dw, dh);
  ui.overlay.hidden = false;
}

function exportName(ext: string): string {
  const base = (scene?.name ?? "scene").replace(/\.[^.]+$/, "");
  const s = scene?.states[Number(ui.shown.value)];
  const tail = scene && scene.states.length > 1 && s ? `_${shortLabel(s.label, 0).replace(/\s+/g, "")}` : "";
  return `${base}_${scene?.kind ?? "scene"}${tail}.${ext}`;
}
ui.exportPly.addEventListener("click", () => {
  const s = scene?.states[Number(ui.shown.value)];
  if (!s?.splats) { showStatus("Nothing to export yet."); return; }
  const bytes = splatsToPly(s.splats);
  download(bytes, exportName("ply"));
  showStatus(`Exported ${fmtInt(s.splats.n)} Gaussians as .ply, ${(bytes.byteLength / 1e6).toFixed(1)} MB.`);
});
ui.exportSog.addEventListener("click", async () => {
  const s = scene?.states[Number(ui.shown.value)];
  if (!scene || !s?.splats) { showStatus("Nothing to export yet."); return; }
  const key = `${scene.id}:${ui.shown.value}`;
  ui.exportSog.disabled = true;
  try {
    let bytes = sogCache.get(key);
    if (!bytes) { bytes = await splatsToSog(s.splats); sogCache.set(key, bytes); }
    download(bytes, exportName("sog"));
    showStatus(`Exported ${fmtInt(s.splats.n)} Gaussians as .sog, ${(bytes.byteLength / 1e6).toFixed(1)} MB.`);
  } catch (e) {
    showStatus(`SOG export failed: ${String((e as Error).message ?? e)}`);
  } finally {
    ui.exportSog.disabled = false;
  }
});

// Test hook: the page's state for the browser checks.
(window as unknown as { __planarDemo: unknown }).__planarDemo = {
  get result() { return lastRun; },
  get running() { return running; },
  sogBytes: async () => (lastRun ? (await splatsToSog(lastRun.run.splats)).byteLength : 0),
};
