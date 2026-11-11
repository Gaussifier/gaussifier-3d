/**
 * Placement stage contract (capturable native semantics), implemented by ./placement.ts and
 * driven by the engine. Everything is recorded into the caller's command encoder with no
 * readbacks; dispatch sizes are static because the schedule removes exactly the planned count
 * each round.
 */
export interface PlacementParams {
  width: number;
  height: number;
  /** Target point count N. */
  n: number;
  /** Oversampled capacity max(N, ceil(1.5 N)). Points buffer holds nMax rows. */
  nMax: number;
  mergeRounds: number;      // 6
  perRoundFraction: number; // 1/3
  knnK: number;             // 8
  lloydIters: number;       // 3
}

export interface PlacementInputs {
  /** f32 [H*W] unit-mean density, GPU resident. */
  density: GPUBuffer;
  /** f32 [nMax*2] normalized oversampled points from the WASM sampler, uploaded by the engine. */
  points: GPUBuffer;
}

export interface PlacementOutputs {
  /** f32 [N*2] normalized points after merge + Lloyd. */
  points: GPUBuffer;
  /** f32 [N] mass weights: mass / (sum(mass)/N). */
  weights: GPUBuffer;
  /** i32 [H*W] owner map from the final assignment (with recovery). */
  owner: GPUBuffer;
  /** f32 [N] cell masses. */
  masses: GPUBuffer;
}

export interface Placement {
  /** Allocate or reuse buffers for these params; call once per size/count. */
  prepare(params: PlacementParams): void;
  /** Record all placement work into the encoder. Returns buffers valid after the submit. */
  record(encoder: GPUCommandEncoder, inputs: PlacementInputs): PlacementOutputs;
  /** Record only a Voronoi assignment (with recovery) of `count` points into `owner`. Used by tests and the viewer overlay. */
  recordAssignment(encoder: GPUCommandEncoder, points: GPUBuffer, count: number, owner: GPUBuffer): void;
  dispose(): void;
}
