import { disposeOutputs, runGpu, type GpuOutput, type OrtSession } from "../ort.js";
import type { HeadBackend } from "./types.js";

/** ORT-session head: dynamic session, or the static 512 session with graph capture when the size matches. */
export class OrtHead implements HeadBackend {
  readonly kind = "ort" as const;
  private session: OrtSession;
  private outputs: GpuOutput[] | null = null;
  constructor(private dynamic: OrtSession, private static512: OrtSession | null, private inputName: string, private outputName: string) {
    this.session = dynamic;
  }
  prepare(width: number, height: number): void {
    this.session = width === 512 && height === 512 && this.static512 ? this.static512 : this.dynamic;
  }
  async run(features: GPUBuffer, width: number, height: number): Promise<GPUBuffer> {
    this.prepare(width, height);
    this.release();
    this.outputs = await runGpu(this.session, this.inputName, features, [1, 17, height, width]);
    const delta = this.outputs.find((o) => o.name === this.outputName) ?? this.outputs[0];
    return delta.buffer;
  }
  release(): void {
    if (this.outputs) { disposeOutputs(this.outputs); this.outputs = null; }
  }
  dispose(): void {
    this.release();
    void this.dynamic.release();
    void this.static512?.release();
  }
}
