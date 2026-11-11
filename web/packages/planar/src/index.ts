// Browser engine for the planar (flat 3DGS) model: ONNX Runtime Web graphs, the EMV placement of
// @gaussifier/inference, and a WGSL port of the Faster-GS rasterizer.
export * from "./raster/index.js";
export { loadPlanarBundle, fetchBytes, pointCount, camerasFor, frontalCamera, orbitCamera, halfThickness, GRAPH_NAMES, PlanarBundleError, type PlanarBundle, type PlanarManifest, type PlanarCamera, type PlanarGeometry, type GraphName } from "./bundle.js";
export { PlanarEngine, AbortedError, type FrontalRasterizer, type GradientBuffers as EngineGradientBuffers, type PlanarSplats, type PlanarRunOptions, type PlanarRunResult, type PlanarState, type PlanarImage } from "./engine.js";
export { PointCounter } from "./count.js";
export { tieFreeMeans, psnr } from "./final.js";
export { PlaneWarp } from "./warp.js";
export { Residual } from "./residual.js";
