# Gaussifier 3D

Inference code and weights for **Gaussifier 3D**, the planar 3D Gaussian model of Gaussifier,
an anonymous submission under double-blind review. The model turns one image into a thin slab of
3D Gaussians: a standard 3DGS scene (spherical harmonics of degree 0) that matches the image from
the front and keeps its relief when turned.

- Live browser demo (WebGPU): <https://gaussifier.github.io/demo3d/>
- Weights: [release v0.1.0](https://github.com/Gaussifier/gaussifier-3d/releases/tag/v0.1.0)
- The 2D model: <https://github.com/Gaussifier/gaussifier>

There are two models, both with seven rendered states and six correction steps:

| Model | Correction loop input | C++ (TensorRT, 512×512) |
|---|---|---|
| fast | the frontal render | 16.1 ms |
| quality | also the residual backprojected through the rasterizer from the frontal view and a 30° view | 18.0 ms |

Timings are steady-state end to end on an RTX 5090.

## Installation

Requirements: Linux, an NVIDIA GPU with a CUDA toolkit, Python 3.12, and
[uv](https://docs.astral.sh/uv/). Tested versions are listed in
[docs/compatibility.md](docs/compatibility.md).

```bash
git clone https://github.com/Gaussifier/gaussifier-3d.git
cd gaussifier-3d
uv sync

# The Faster-GS rasterizer: cloned at a pinned commit, patched for CUDA 13, installed
uv run python scripts/install_fastergs.py

# The weights: one runtime export per model (torch.export programs plus planar_runtime.json)
mkdir -p runtimes
for model in fast quality; do
  curl -L -o runtimes/planar_${model}_runtime.tar \
    https://github.com/Gaussifier/gaussifier-3d/releases/download/v0.1.0/planar_${model}_runtime.tar
  tar -xf runtimes/planar_${model}_runtime.tar -C runtimes
done
sha256sum -c release/SHA256SUMS --ignore-missing
```

The first inference call compiles the native placement kernels (Torch extensions).

## Inference

The runtime exports for Python and C++ are compiled for 512×512 images, so the command line uses
the centre square of a picture, resized. The browser engine runs at the picture's own aspect
ratio: the demo scales pictures so the short side is 512 pixels (the long side at most 1024) and
trims both sides to a multiple of 16.

### Command line

```bash
uv run gaussifier-planar --runtime runtimes/planar_quality_runtime \
  --image image.png --output scene.ply --render render.png
```

This writes the Gaussians as a standard 3DGS PLY and, with `--render`, the frontal render.
`--seed` fixes the placement seed and `--count N` the number of Gaussians.

The PLY opens upright in 3DGS viewers such as SuperSplat, facing their default camera. It is
written in the frame these viewers expect of a file: they turn the data 180° about z when they
load it. The model itself works in OpenCV axes: the image lies on the plane `z = 0` with height 1
and its centre at the origin, and the frontal camera sits at `z = -1` with a focal length of 512
pixels; the file holds that frame turned 180° about y, `(-x, y, -z)`. To get the compressed SOG
format of PlayCanvas and SuperSplat:

```bash
npx @playcanvas/splat-transform scene.ply scene.sog
```

### Python

```python
from gaussifier_sampler.image_io import load_image_tensor
from gaussifier_sampler.planar import PlanarRuntime, fit_image, save_ply

runtime = PlanarRuntime("runtimes/planar_quality_runtime")
image = fit_image(load_image_tensor("image.png"), runtime.height)
result = runtime(image)
save_ply(result.splats, "scene.ply")
print(result.count, "Gaussians, frontal PSNR", result.frontal_psnr[-1])
```

`result.splats` holds the five 3DGS tensors in the model's OpenCV frame: `means`, `log_scales`,
`quats` (w, x, y, z), `opacity_logits`, and `sh0`. `save_ply` writes them in the viewers' frame
(`to_file_frame`).

### C++ (AOT Inductor and TensorRT)

`gaussifier_planar_infer` runs the same exports compiled with AOT Inductor, with the backbone and
the correction UNet as FP16 TensorRT engines. The engines are specific to the GPU, driver, and
CUDA, PyTorch, and TensorRT versions, so build them on the machine that runs them.

```bash
uv sync --extra tensorrt
uv run python scripts/install_fastergs.py   # also provides the sources the harness compiles
bash cpp_inference/scripts/install_trt_sdk.sh   # TensorRT headers into tensorrt_sdk/, no sudo
cmake -S cpp_inference -B cpp_inference/build \
  -DCMAKE_PREFIX_PATH=$(uv run python -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake --build cpp_inference/build --target gaussifier_planar_infer -j

uv run python scripts/build_planar_runtime.py --trt \
  --export runtimes/planar_quality_runtime --out runtimes/planar_quality_cpp

# The input is a float32 (1, 3, 512, 512) array in [0, 1]
uv run python -c "
from gaussifier_sampler.image_io import load_image_tensor
from gaussifier_sampler.planar import fit_image
import numpy as np
np.save('image.npy', fit_image(load_image_tensor('image.png'), 512)[None].numpy())"
cpp_inference/build/gaussifier_planar_infer runtimes/planar_quality_cpp image.npy scene.npy
```

Without `--trt` the runtime uses AOT Inductor for every graph and needs no TensorRT. The harness
writes `scene.{means,log_scales,quats,opacity_logits,sh0}.npy`. `GAUSSIFIER_SEED`
sets the placement seed and `GAUSSIFIER_REPEAT` repeats the run for timing.

### Browser

The browser engine runs the whole model with WebGPU: ONNX Runtime Web for the networks, the same
equal-mass Voronoi placement as the 2D engine, and a WebGPU port of the Faster-GS rasterizer,
with its backward pass for the quality model. The demo shows the Gaussians in PlayCanvas and
downloads them as SOG or PLY. Unpack the web bundles of the release into the demo, then build and
serve it (needs Node.js and npm):

```bash
for model in fast quality; do
  curl -L -o planar_${model}_web.zip \
    https://github.com/Gaussifier/gaussifier-3d/releases/download/v0.1.0/planar_${model}_web.zip
  mkdir -p web/apps/planar-demo/public/models/${model}
  unzip -o planar_${model}_web.zip -d web/apps/planar-demo/public/models/${model}
done
cd web
npm ci && npm run build && npm run build:planar-demo
npm run serve:planar-demo
```

Open <http://localhost:8081/> in a browser with WebGPU (current Chrome or Edge, Safari 26, or
Firefox on Windows). WebGPU needs a secure context, so other machines should use the HTTPS
port 8444. See [web/README.md](web/README.md).
