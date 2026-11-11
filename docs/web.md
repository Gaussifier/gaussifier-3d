# Browser engine and viewer

The `web/` tree holds a browser port of the production inference profile and a
framework-free viewer for 2D Gaussian splat scenes. The TypeScript packages,
their build and test commands, and the WebGPU conventions are documented in
[web/README.md](../web/README.md). This page covers the Python side that feeds
them.

## Export the model bundle

The engine loads single-file fp32 ONNX graphs plus a manifest validated
against [web_bundle.schema.json](../schemas/web_bundle.schema.json):

```bash
uv run --extra web python scripts/export_web_bundle.py \
  --out-dir web/apps/demo/public/models
```

Without syncing the `web` extra, the same command runs with
`uv run --with onnx --with onnxscript --with onnxruntime python scripts/export_web_bundle.py`.
The script exports `forward_map.onnx` and `correction_head.onnx` with dynamic
height and width, static 512 variants for graph capture, verifies every file
against the eager model on the CPU, and refuses to write the manifest when a
check fails. The output directory is ignored by git.

The same run writes `correction_head_wgsl.bin`, the correction-head weights as
one float32 little-endian array in `state_dict` order, and
`correction_head_wgsl.json`, which lists every tensor's name, shape, element
offset, and count plus the head architecture. They appear in the manifest with
the roles `correction_head_wgsl` and `correction_head_wgsl_layout` and feed the
engine's hand-written WGSL head, which replaces the ONNX Runtime head when the
files are present.

## Write goldens

`scripts/export_web_goldens.py` runs the production profile on CUDA with the
exact reference renderer from `gaussifier_sampler.web` and records every stage
the browser reproduces: forward-map outputs, the CUDA oversample, the merged
and polished placements, the owner map, masses and weights, the initial
attributes, and per-state points, attributes, colors, and renders.

```bash
uv run python scripts/export_web_goldens.py --image tests/fixtures/bench_input_512.png \
  --name bench_input_512 --out web/fixtures/bench_input_512
uv run python scripts/export_web_goldens.py --image tests/fixtures/bench_input_512.png \
  --name bench_128 --crop 128 --out web/fixtures/bench_128
```

Each fixture directory holds raw little-endian `.bin` arrays described by
`shapes.json`, a `scene.npz` for the viewer, and `manifest.json` with the PSNR
of every rendered state and the native placement's run-to-run envelope. Binary
outputs are ignored by git; regenerate them on a CUDA workstation.

## Viewer module and demo

`web/apps/demo` is the browser demo; `npm run build:demo` then `npm run serve`
in `web/` serves it over HTTP and HTTPS. `npm run build:bundle` produces the
single-file viewer module `gaussifier-viewer.js`, which plays a `.splat2d` or
`.npz` asset from a URL through the `<gaussifier-viewer src="...">` element;
`embed.html` in the demo shows it. The demo exports Gaussians as `.splat2d`,
the Splat2D format that the training code's `data/splat2d.py` reads, written at
the original image size (padding removed, xy rescaled), and
`gaussifier_sampler.web.load_scene` reads the viewer's NPZ export.

The viewer overlays are the source image, the difference, the density map,
the Voronoi cells of the current centers (tinted, with dark borders), the
centers, and the 1σ ellipses. The oversampler wasm ships prebuilt under
`web/packages/wasm-oversampler/prebuilt/` with a provenance record that the
web unit tests check against the C++ source, and `.github/workflows/web.yml`
runs the GPU-free web checks on every change under `web/`.

## Reference renderer

`gaussifier_sampler.web.reference_render` evaluates the production simple-sum
formula without tile culling. It accumulates each Gaussian only inside the box
where its alpha can reach the `1/255` cutoff, which is identical to a full
per-pixel loop up to summation order. `save_scene` writes the NPZ key set the
viewer's loader understands.
