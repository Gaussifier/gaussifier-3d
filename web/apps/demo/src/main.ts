import "./style.css";
import { createViewer, loadNpz, loadNpyTriple, loadSplat2d, fromRunResult, sceneToSplat2d, type Scene, type Viewer } from "@gaussifier/viewer";
import { WebGpuUnavailableError, prepareInput, type PreparedInput } from "@gaussifier/inference";
import { decodeImage, decodedSize, getEngine, gpuSpeed, prepareImage, runInference, type GpuSpeed } from "./inference.js";

// ---- elements ----
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const ui = {
  canvas: $<HTMLCanvasElement>("view"),
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
  samples: $<HTMLDivElement>("samples"),
  refine: $<HTMLInputElement>("refine"),
  refineValue: $<HTMLSpanElement>("refineValue"),
  count: $<HTMLInputElement>("count"),
  countValue: $<HTMLSpanElement>("countValue"),
  statN: $<HTMLDivElement>("statN"),
  psnr: $<HTMLSpanElement>("psnr"),
  shownRow: $<HTMLDivElement>("shownRow"),
  shown: $<HTMLInputElement>("shown"),
  shownValue: $<HTMLSpanElement>("shownValue"),
  export: $<HTMLButtonElement>("export"),
  drop: $<HTMLDivElement>("drop"),
  pick: $<HTMLButtonElement>("pick"),
  file: $<HTMLInputElement>("file"),
};

// ---- state ----
const viewer: Viewer = createViewer(ui.canvas, { backend: "auto" });
let currentScene: Scene | null = null;
let currentName = "scene";
let imageFile: File | null = null;
/** The picked image converted once; the same planar array is handed to every run and to the viewer. */
let preparedImage: { file: File; input: PreparedInput } | null = null;
let autoCount: number | null = null;
let running = false;
let queued = false;

const fmtInt = (n: number) => n.toLocaleString("en-US");

// ---- engine and GPU status ----
// The engine loads as soon as the page opens (30 MB of model and runtime files, then shader
// compilation), so the first run does not pay for it. The pill says what the GPU ORT chose means
// for speed: the page runs fast only on a real GPU.
let engineReady = false, engineError: string | null = null;
let speed: GpuSpeed = "unknown";
const SPEED_NOTE: Record<GpuSpeed, string> = {
  fast: "",
  slow: "Runs take a few seconds on an integrated GPU.",
  software: "No GPU: this is a software renderer, runs take minutes.",
  unknown: "",
};
function setPill(text: string, css: "on" | "warn" | "bad", title: string): void {
  ui.gpuPill.textContent = text;
  ui.gpuPill.classList.remove("on", "warn", "bad");
  ui.gpuPill.classList.add(css);
  ui.gpuPill.title = title;
}
const NEEDS_WEBGPU = "This page needs WebGPU: a current Chrome or Edge, Safari 26, or Firefox on Windows, with a GPU.";
function updateReady(): void {
  if (running) return;
  if (engineError) { ui.runStatus.textContent = engineError; return; }
  if (!engineReady) { ui.runStatus.textContent = imageFile ? "Loading the model…" : "Loading the model and the sample image…"; return; }
  const note = SPEED_NOTE[speed];
  ui.runStatus.textContent = (imageFile ? "Ready." : "Pick an image to start.") + (note ? ` ${note}` : "");
}
viewer.ready.catch((e) => { setPill("no gpu", "bad", String(e)); showStatus(String(e)); });
getEngine().then((engine) => {
  engineReady = true;
  speed = gpuSpeed(engine.adapter);
  const vendor = engine.adapter.split(" · ").pop() ?? "";
  const detail = `${engine.adapter} (viewer: ${viewer.backendKind}). Runs take well under a second on a discrete GPU, a few seconds on an integrated one, and minutes on a software renderer.`;
  if (speed === "fast") setPill(`${vendor} gpu`, "on", detail);
  else if (speed === "slow") setPill("integrated gpu", "warn", detail);
  else if (speed === "software") setPill("no gpu · slow", "bad", detail);
  else setPill("gpu", "on", detail);
  updateReady();
}).catch((e) => {
  engineError = e instanceof WebGpuUnavailableError ? NEEDS_WEBGPU : `${NEEDS_WEBGPU} The engine failed to load: ${String(e)}`;
  setPill("no webgpu", "bad", NEEDS_WEBGPU);
  updateReady();
});

function showStatus(text: string): void { ui.status.hidden = false; ui.status.textContent = text; }

// ---- showing a scene ----
function showScene(scene: Scene, name: string): void {
  // The initializer is not interesting to look at: drop it whenever refinements follow.
  if (scene.states.length > 1 && scene.states[0].label === "initializer") scene.states = scene.states.slice(1);
  currentScene = scene; currentName = name;
  viewer.setScene(scene);
  ui.empty.hidden = true;
  const last = scene.states.length - 1;
  ui.shownRow.hidden = scene.states.length <= 1;
  ui.shown.max = String(last);
  ui.shown.value = String(last);
  viewer.setState(last);
  ui.shownValue.textContent = shortLabel(scene.states[last].label, last);
  ui.statN.textContent = fmtInt(scene.states[0].xy.length / 2);
  ui.status.hidden = true;
  updatePsnr();
}
/** The scrubber row already says "Refinement": show the number, or the label of anything else. */
function shortLabel(label: string | undefined, index: number): string {
  return (label ?? String(index)).replace(/^refinement /, "");
}
function updatePsnr(): void {
  const v = viewer.psnr();
  ui.psnr.innerHTML = v === null ? "—" : `${v.toFixed(2)}<small>dB</small>`;
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
async function preparedFor(file: File): Promise<PreparedInput> {
  if (preparedImage?.file === file) return preparedImage.input;
  const bitmap = await decodeImage(file);
  try { const input = await prepareImage(bitmap); preparedImage = { file, input }; return input; } finally { bitmap.close(); }
}
async function run(): Promise<void> {
  if (!imageFile) return;
  if (engineError) { updateReady(); return; }
  if (running) { queued = true; return; }
  const states = refinements() + 1;
  running = true;
  ui.runStatus.textContent = engineReady ? "Running…" : "Loading the model, then running…";
  ui.live.hidden = false; ui.live.textContent = states > 1 ? `running ${states - 1} refinements` : "running the initializer";
  const name = imageFile.name;
  try {
    const input = await preparedFor(imageFile);
    const result = await runInference(input, states, countScale());
    // The result carries every kept state: refinements 1..K-1, or the initializer alone at K=1.
    const scene = fromRunResult(result.run);
    scene.image = input.planar;
    showScene(scene, name);
    autoCount = result.run.adaptiveCount;
    syncCountLabel();
    ui.runStatus.textContent = `Done on ${result.adapter}.`;
  } catch (e) {
    ui.runStatus.textContent = engineError ?? `Run failed: ${String(e)}`;
  } finally {
    running = false; ui.live.hidden = true;
    ui.runStatus.dataset.runs = String(Number(ui.runStatus.dataset.runs ?? 0) + 1);
    if (queued) { queued = false; void run(); }
  }
}

// ---- controls: refinements and count ----
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
    const size = decodedSize(w, h);
    const resized = size.resized ? ` · resized to ${size.width}×${size.height}` : "";
    ui.imageDims.textContent = `${w}×${h} · ${(file.size / 1024).toFixed(0)} KB${resized}`;
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
void loadSample(requested && KODAK.includes(requested) ? requested : KODAK[Math.floor(Math.random() * KODAK.length)]);

// ---- saved scenes ----
/** A saved asset carries no pixels; the picked image stands in for PSNR and the image overlays when it has the asset's frame size. */
async function withSourceImage(scene: Scene): Promise<Scene> {
  if (scene.image || !imageFile) return scene;
  const bitmap = await decodeImage(imageFile);
  try {
    if (bitmap.width === scene.width && bitmap.height === scene.height) scene.image = prepareInput(bitmap, 1).planar;
  } finally {
    bitmap.close();
  }
  return scene;
}
async function loadFiles(files: FileList | File[]): Promise<void> {
  const list = Array.from(files);
  const npz = list.find((f) => f.name.endsWith(".npz"));
  const splat = list.find((f) => f.name.endsWith(".splat2d"));
  try {
    if (splat) { showScene(await withSourceImage(loadSplat2d(await splat.arrayBuffer())), splat.name); return; }
    if (npz) { showScene(await withSourceImage(await loadNpz(await npz.arrayBuffer())), npz.name); return; }
    const pick = (suffix: string) => list.find((f) => f.name.endsWith(suffix));
    const points = pick(".points.npy"), cov = pick(".cov.npy"), rgb = pick(".rgb.npy");
    if (points && cov && rgb) {
      const width = Number(prompt("Frame width in pixels", "512"));
      const height = Number(prompt("Frame height in pixels", "512"));
      showScene(loadNpyTriple({ points: await points.arrayBuffer(), cov: await cov.arrayBuffer(), rgb: await rgb.arrayBuffer() }, { width, height }), points.name);
      return;
    }
    showStatus("Drop a .splat2d, an .npz, or the three .points/.cov/.rgb .npy files.");
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
ui.shown.addEventListener("input", () => { viewer.setState(Number(ui.shown.value)); });
viewer.on("statechange", ({ k }) => {
  ui.shown.value = String(k);
  ui.shownValue.textContent = shortLabel(currentScene?.states[k]?.label, k);
  // Preview states carry no rendered planes; scoring them would run the CPU reference renderer per state.
  if (!running) updatePsnr();
});
for (const name of ["image", "diff", "density", "voronoi", "centers", "ellipses"] as const) {
  $<HTMLInputElement>(`ov-${name}`).addEventListener("change", (e) => viewer.setOverlay(name, (e.target as HTMLInputElement).checked));
}
ui.export.addEventListener("click", () => {
  if (!currentScene) { showStatus("Nothing to export yet."); return; }
  const blob = sceneToSplat2d(currentScene, { stateIndex: Number(ui.shown.value) || currentScene.states.length - 1 });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${currentName.replace(/\.[^.]+$/, "")}.splat2d`;
  a.click();
  const frame = currentScene.crop ?? { w: currentScene.width, h: currentScene.height };
  showStatus(`Exported ${fmtInt(currentScene.states[0].xy.length / 2)} Gaussians at ${frame.w}×${frame.h} as .splat2d, ${(blob.size / 1e6).toFixed(1)} MB.`);
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});
