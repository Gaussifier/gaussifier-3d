/**
 * Building blocks behind the engine, exported for tests, tools, and custom pipelines.
 * The stable public API is what index.ts exports directly.
 */
export { loadBundle, sha256Hex, type LoadedBundle } from "./bundle.js";
export { configureOrt, createSession, ortDevice, runGpu, disposeOutputs, gpuTensor, type OrtSession, type GpuOutput } from "./ort.js";
export { KLoop, KLoopKernels, listCapacity, type KLoopConfig } from "./kloop/kloop.js";
export { FEATURES, TILE, LARGE_TILE_THRESHOLD } from "./kloop/shaders.js";
export { TileScan, SCAN_WGSL } from "./kloop/scan.js";
export * as kloopShaders from "./kloop/shaders.js";
export { createPlacement, GpuPlacement } from "./placement/placement.js";
export { jfaSchedule, mergePlan, oversampleCount } from "./placement/schedule.js";
export { WgslHead, parseHeadWeights, type HeadLayout } from "./head/wgsl_head.js";
export { OrtHead } from "./head/ort_head.js";
export { convShader, convTile, DEFAULT_CONV, type ConvConfig } from "./head/shaders.js";
