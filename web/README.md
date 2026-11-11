# Web inference engine and 2DGS viewer

Browser port of the production inference profile of `gaussifier-sampler`,
plus a framework-free viewer for 2D Gaussian splat scenes. The Python side
that produces the model bundle and the goldens is documented in
[docs/web.md](../docs/web.md).

## Layout

```
web/
  packages/wgsl/               shared WGSL shaders, GPU helper, placement kernels
  packages/wasm-oversampler/   Emscripten build of the CPU oversampler (bit-exact with CUDA); prebuilt/ is committed
  packages/inference/          engine: ONNX Runtime Web + WGSL placement, head, and K-loop
  packages/viewer/             viewer: loaders, exporters, WebGPU and WebGL2 backends, <gaussifier-viewer>
  apps/demo/                   the demo page (Vite): index.html, src/main.ts, src/style.css, src/inference.ts
  packages/planar/             planar (flat 3DGS) engine: ONNX Runtime Web graphs, the shared EMV placement,
                               a WGSL port of the Faster-GS rasterizer (forward and backward), plane warp
  apps/planar-demo/            the 3D demo page (Vite, PlayCanvas viewer, SOG and PLY downloads)
  tools/                       server, Chromium launcher, browser test runner, demo check, shared cli.mjs
  test/e2e.html                end-to-end gate against the goldens
  fixtures/                    goldens and reference inputs (large files ignored by git)
```

## Commands

```bash
cd web
npm install
npm run build                             # tsc for every package
npm run build:bundle                      # packages/viewer/bundle/gaussifier-viewer.js
npm run build:demo                        # apps/demo/dist, copies runtime assets into public/
npm test                                  # vitest unit tests (Node)
npm run test:browser                      # every WebGPU test page in Chromium, sequentially
npm run check:demo                        # waits for the built demo's automatic run (--scale S, --states K rerun with a change)
npm run serve                             # serves apps/demo/dist on :8080 and https :8443, all interfaces
EMSDK=/path/to/emsdk npm run build:wasm   # only after editing the C++ oversampler source (needs Emscripten)
```

The oversampler wasm is committed under `packages/wasm-oversampler/prebuilt/`
with a `BUILD.json` that records the Emscripten version, the flags, and the
sha256 of every input. `npm test` compares those hashes with the current
sources, so editing `density_points.cpp`, `tools/transform.py`, or `build.sh`
without rebuilding fails the unit tests. The build is deterministic: two runs
of the same toolchain produce identical bytes. To rebuild, install the
recorded Emscripten once and `build.sh` finds it at `~/emsdk`:

```bash
git clone https://github.com/emscripten-core/emsdk ~/emsdk
~/emsdk/emsdk install 6.0.9 && ~/emsdk/emsdk activate 6.0.9
npm run build:wasm
```

The model bundle must exist at `apps/demo/public/models/` and the goldens at
`fixtures/<name>/` before the browser tests run; both come from the Python
scripts in [docs/web.md](../docs/web.md).

`tools/browser-run.mjs <page>` runs one page; `--all` runs the list in the
script. Chromium is found through `CHROME_PATH` or the Playwright cache. On a
Linux workstation with an NVIDIA GPU the launcher starts Xvfb and passes
`--enable-features=Vulkan --use-vulkan=native`; plain headless Chromium on
Linux only reaches SwiftShader. `WEBGPU_SOFTWARE=1` selects SwiftShader.

## Serving to another machine

WebGPU needs a secure context. `npm run serve` opens an HTTPS port with a
self-signed certificate that names this host's addresses; accept it once in
the remote browser. Certificates are cached under `tools/.certs/`. Anyone who
can reach the port can download the model bundle.

## Viewer module

`gaussifier-viewer.js` is a single ES module with no dependencies. One tag
plays an asset from a URL; `apps/demo/public/embed.html` shows it in use.

```html
<script type="module" src="gaussifier-viewer.js"></script>
<gaussifier-viewer src="scene.splat2d"></gaussifier-viewer>
```

The element takes `src`, `backend`, and `state`, exposes `viewer`, `scene`,
and `load(url)`, and dispatches `load` and `error`. The same file exports the
programmatic API: `createViewer`, `loadSceneFromUrl`, `decodeScene`, the
loaders, and the exporters. Both backends draw the production simple-sum
formula, so a Splat2D export renders at 39.6 dB against its source.

Overlays (`viewer.setOverlay(name, on, opts)`): `image`, `diff` (with
`{ gain }`), `density`, `voronoi`, `centers`, `ellipses` (with `{ sigma,
width }`). The Voronoi overlay tints every cell from a hash of its owner and
draws dark one-pixel borders; the nearest-center map is computed once per
state on the CPU with an exact grid search (about 30 ms for 84K centers at
512²) and uploaded as an integer texture. The ellipse overlay draws each
Gaussian's 1σ ellipse as an antialiased ring of constant screen width in the
fragment shader, from the same instance buffer as the splats, with a dark
halo so it reads on any background. Centers are antialiased orange discs
with the same halo, drawn the same way. All three scale their detail with
zoom: borders fade out when cells are under about 8 display pixels, rings
fade to a hint under about 7 pixels of radius, and discs shrink to a
one-pixel stipple when centers are packed closer than a few pixels, so a
zoomed-out view shows the mosaic, the coverage, and the sampling density
rather than a wall of marks. `viewer.psnr()` scores a state
against the scene image; a state without stored render planes is rendered
by the CPU reference renderer on first use, so loaded assets can be scored
when the demo attaches the picked image.

Controls: wheel or pinch to zoom about the pointer, drag to pan (one finger
on touch screens; the canvas sets `touch-action: none`), `f` fits, `1` is
actual size, `[` and `]` step through states.

## Formats

- `.splat2d`: the Splat2D format, 12-byte header `GS2D`, count, height,
  width, then float32 xy, scale, rotation, and color. The stored scale is the
  reciprocal of the effective pixel sigma, clamped at 0.01. Default export.
  A padded scene is written at the original image size with xy rescaled so
  every Gaussian keeps its pixel position (`{ frame: "padded" }` keeps the
  padded frame instead).
- `.npz`: the `save_scene` layout of the Python package, read and written by
  the viewer; the Python workflow's format.
- The C++ harness's points, cov, and rgb NPY triple, read only.

## Engine

`createEngine({ bundleUrl })` fetches `web_bundle.json`, verifies every file
hash, creates the ONNX Runtime Web sessions on the WebGPU provider, and takes
the GPUDevice from the runtime. `engine.run(image, { states, count, seed })`
returns `RunResult`. Types live in `packages/inference/src/types.ts`; the
lower-level pieces are exported under `internals`.

The correction head has two backends behind one interface: `wgsl`, the
hand-written kernels fed by `correction_head_wgsl.bin`, and `ort`, the ONNX
session with graph capture at 512. `createEngine({ headBackend })` and
`run({ headBackend })` choose; `auto` prefers `wgsl`. The head is two UNets
of different widths with the same structure, and every kernel runs both
nets' matching layers in one dispatch from shared buffers with per-net
offsets: 45 dispatches per call. The conv tiling is a generator parameter;
`packages/inference/test/browser/head.html?sweep=1` times the alternatives.

The production profile is the only profile: raw density, truncated adaptive
count, capturable-native placement semantics, fixed initial Voronoi-mass
weights at every render, and the pixel-anchored xy step `2.56 / max(H, W)`.
Run options beyond that:

- `count` pins the Gaussian count instead of the adaptive `trunc(sum(rate))`,
  and `countScale` multiplies the adaptive count once the forward map has
  produced it; `RunResult.adaptiveCount` reports the model's own count either
  way. A count above the adaptive one scales the predicted covariance by
  `adaptiveCount / N` (the `cov_shift` of `pack_static`), as the Python and
  C++ paths do; without it N Gaussians of the predicted shape render about
  `N / adaptiveCount` times too bright. A lower count leaves the map alone.
  The pinned path matches the Python reference within 0.05 dB: on
  bench_input_512 the adaptive 84,129 reach 37.7 dB, 30,000 give 31.3 dB and
  200,000 give 36.0 dB (24.4 dB before the covariance scaling), in both
  implementations.
- `keepStates` copies every state to a staging buffer inside the loop and
  reads them all back once at the end, so the GPU never stalls between
  states; `RunResult.snapshots` holds them. `onState` instead delivers each
  state as it completes, at one GPU round trip per state. `includeLatent:
  false` leaves the latent RGB out of either readback.
- A planar input marked `prepared` (the output of `prepareInput`) is used
  as is instead of being copied and clamped again.

The demo uses all three. There is no Run button: picking or dropping an
image runs it, and so does changing the Refine slider (correction steps,
0 to 11; the model is trained for 6, so K = refinements + 1 rendered
states) or the Count slider (a logarithmic scale from ×¼ to ×4 around the
model's own count, "auto" at the centre), with one more run queued if a
control changes during a run. A picked image larger than 768 px on its
longest side is downscaled to that on decode (the file card says so), so a
phone photo runs in the same time as a Kodak image. The picked image is prepared once, every refinement
is kept for the Refinement slider (each scored on first visit, Export saves
the shown one), and one of the 24 Kodak images (committed under
`apps/demo/public/samples/kodak`, 15 MB, with a thumbnail strip to pick
another; `?sample=kodim07` pins one) runs as soon as the page opens. The
engine is created and warmed at page open and the pill names the GPU: ORT
owns the device, and the engine asks for the high-performance adapter
because the default on a machine with two GPUs is often the integrated one.

## Measured on an RTX 5090 (Chromium under Xvfb, fp32)

| Run | WGSL head | ORT head | Final PSNR | Golden PSNR |
|---|---|---|---|---|
| bench_input_512, N=84129, K=7 | 66 ms | 127 ms | 37.736 dB | 37.733 dB |
| bench_input_512, N=84129, K=5 | 47 ms | 80 ms | 37.219 dB | 37.218 dB |
| bench_input_512, N=84129, K=1 | 29 ms | | | |
| head call alone at 512, GPU time | 5.3 ms | 12.4 ms | | |
| demo, click to done, K=7 | 105 ms | | | |

A submit-and-wait round trip costs about 2 ms in Chrome, so the head is
timed with ten calls per submit; at 64² the same call takes 0.7 ms, which
is the dispatch overhead, and the rest is convolution arithmetic (about
83 GFLOP per call, 18 TFLOPS achieved in fp32).

Per-stage timings in `RunResult.timings` are submission times unless
`run({ profile: true })`, which waits for the GPU after every stage. The
adaptive count differs from the CUDA reference by a few points because the
WebGPU fp32 rate map differs from the TF32 reference at the 1e-4 level; the
end-to-end gate allows 0.02% of N.

## Continuous integration

`.github/workflows/web.yml` runs on every change under `web/`: `npm ci`, the
TypeScript build, the unit tests (including the prebuilt wasm provenance
check), the viewer bundle, and the two compute test pages (oversampler,
placement) on Chromium's SwiftShader adapter. Hosted runners have no GPU, so
the ORT sessions, the render backends, and the end-to-end gate stay on the
workstation (`npm run test:browser`). Pages skip the checks whose gitignored
fixtures are absent instead of failing.

## Conventions

- Packages emit ES modules to `dist/` with explicit `.js` import extensions so
  browser test pages load them without a bundler; test pages declare an
  import map for workspace packages and for `onnxruntime-web/webgpu`.
- Each package's `index.ts` is its API; byte-level and GPU-level pieces are
  exported under `internals` so tests and tools can reach them without the
  API growing.
- The UI vocabulary is the code's: "refinements" are the correction steps
  after the initializer (the engine's K rendered states are refinements + 1),
  and the demo's element ids and variables use the same words.
- Every compute kernel uses an explicit bind group layout and at most eight
  storage buffers per stage. Never bind one buffer twice in a dispatch.
- Record a whole stage into one command encoder and await once.
- ONNX Runtime Web runs its WASM single-threaded; the multithreaded build
  stalls on worker startup under test drivers and the WebGPU provider does
  not need it.
- fp16 is out of scope: the adapters used here expose no `shader-f16`.

## Planar (flat 3DGS) engine and 3D demo

`packages/planar` runs the planar model in the browser with the call sequence of
`gaussifier_planar_infer`, at any image size whose sides divide by 16 (the demo scales pictures to a 512-pixel
short side, at most 1024 pixels on the long side): the forward map, setup, pre-render, head, and update graphs in ONNX Runtime Web; the EMV
placement of `packages/inference`; the hard point count, the 30-degree plane warp, and a WGSL port
of the Faster-GS rasterizer, with its backward pass for the quality model, whose loop reads the
residual backprojected from the frontal and a 30-degree view. `apps/planar-demo` shows the result
in PlayCanvas with an orbit camera and downloads it as SOG (splat-transform) or PLY.

Unpack the release's web bundles (`planar_fast_web.zip`, `planar_quality_web.zip`) into
`apps/planar-demo/public/models/fast` and `.../quality`, then:

```bash
npm run build && npm run build:planar-demo   # apps/planar-demo/dist
npm run serve:planar-demo                    # :8081 and https :8444
npm run check:planar-demo                    # drives the built page: run, orbit screenshots, SOG
```

Parity: the rasterizer matches Faster-GS except where primitives tie in depth (CUDA's own order is
scheduling-dependent there), its gradients sit inside CUDA's run-to-run spread, and both browser engines
reproduce the CUDA reference's frontal PSNR per iteration within about 0.05 dB.
