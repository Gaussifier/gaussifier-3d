/**
 * ONNX Runtime Web setup for the WebGPU execution provider (ORT 1.30).
 *
 * The WASM side runs single-threaded: the multithreaded build stalls session creation under
 * some drivers and the WebGPU provider does not need threads. ORT owns the GPU device; a device
 * set on env.webgpu before the first session is not adopted, so the engine takes ORT's device
 * after creating its sessions and builds every other pipeline on it.
 */
import * as ort from "onnxruntime-web/webgpu";

export type OrtSession = ort.InferenceSession;

export interface OrtSetup {
  /** URL prefix for the ORT wasm files. */
  wasmPaths?: string;
  logLevel?: "verbose" | "info" | "warning" | "error" | "fatal";
  /**
   * Adapter to ask the browser for. ORT owns the device every kernel runs on, so this decides
   * which GPU does the work; default "high-performance", since the default adapter on a machine
   * with an integrated and a discrete GPU is often the integrated one.
   */
  powerPreference?: GPUPowerPreference;
}

let configured = false;

export function configureOrt(setup: OrtSetup = {}): void {
  if (configured) return;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = setup.wasmPaths ?? "/node_modules/onnxruntime-web/dist/";
  ort.env.logLevel = setup.logLevel ?? "warning";
  ort.env.webgpu.powerPreference = setup.powerPreference ?? "high-performance";
  configured = true;
}

/** Vendor, architecture, device, and description of the adapter behind a device, when the browser exposes them. */
export function describeAdapter(device: GPUDevice): string {
  const info = (device as GPUDevice & { adapterInfo?: GPUAdapterInfo }).adapterInfo;
  if (!info) return "unknown adapter";
  const parts = [info.description, info.device, info.architecture, info.vendor].filter((p) => typeof p === "string" && p.length > 0);
  return parts.length ? Array.from(new Set(parts)).join(" · ") : "unknown adapter";
}

export interface SessionOptions {
  graphCapture?: boolean;
}

export async function createSession(model: Uint8Array, options: SessionOptions = {}): Promise<OrtSession> {
  const opts: ort.InferenceSession.SessionOptions = {
    executionProviders: ["webgpu"],
    preferredOutputLocation: "gpu-buffer",
    graphOptimizationLevel: "all",
    logSeverityLevel: 2,
  };
  if (options.graphCapture) opts.enableGraphCapture = true;
  return ort.InferenceSession.create(model, opts);
}

/** The device ORT owns. Valid after the first session exists. */
export async function ortDevice(): Promise<GPUDevice> {
  const device = await ort.env.webgpu.device;
  if (!device) throw new Error("ORT did not expose a WebGPU device");
  return device as GPUDevice;
}

export function gpuTensor(buffer: GPUBuffer, dims: number[]): ort.Tensor {
  return ort.Tensor.fromGpuBuffer(buffer, { dataType: "float32", dims });
}

export interface GpuOutput {
  name: string;
  dims: readonly number[];
  buffer: GPUBuffer;
  tensor: ort.Tensor;
}

/** Run a session with one GPU input; outputs stay on the GPU. Caller copies and disposes. */
export async function runGpu(session: OrtSession, inputName: string, input: GPUBuffer, dims: number[]): Promise<GpuOutput[]> {
  const feeds: Record<string, ort.Tensor> = { [inputName]: gpuTensor(input, dims) };
  const outputs = await session.run(feeds);
  return Object.entries(outputs).map(([name, tensor]) => {
    if (tensor.location !== "gpu-buffer") throw new Error(`output ${name} is not on the GPU (${tensor.location})`);
    return { name, dims: tensor.dims, buffer: tensor.gpuBuffer as GPUBuffer, tensor };
  });
}

export function disposeOutputs(outputs: GpuOutput[]): void {
  for (const o of outputs) o.tensor.dispose();
}
