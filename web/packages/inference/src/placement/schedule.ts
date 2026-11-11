/** Pure schedule helpers for the placement stage (capturable native rules). */

/** Jump-flood steps: smallest power of two >= max(1, floor(max(W,H)/2)), halving to 1, then one extra 1. */
export function jfaSchedule(width: number, height: number): number[] {
  const cap = Math.max(1, Math.floor(Math.max(width, height) / 2));
  let step = 1;
  while (step < cap) step *= 2;
  const out: number[] = [];
  while (step >= 1) { out.push(step); step = Math.floor(step / 2); }
  out.push(1);
  return out;
}

export interface MergeRound {
  /** Active point count entering the round. */
  nIn: number;
  /** Points removed by the round; zero means the round is a no-op copy. */
  remove: number;
}

/** Static merge schedule: every round runs; only rounds with remove > 0 do work. */
export function mergePlan(nMax: number, n: number, rounds: number, perRoundFraction: number): MergeRound[] {
  const roundCount = Math.max(1, rounds);
  const plan: MergeRound[] = [];
  let nEst = nMax;
  for (let r = 0; r < roundCount; r++) {
    const excess = Math.max(0, nEst - n);
    const last = r + 1 === roundCount;
    const cap = last ? excess : excess > 0 ? Math.max(1, Math.floor(nEst * perRoundFraction)) : 0;
    const remove = Math.min(excess, cap);
    plan.push({ nIn: nEst, remove });
    nEst -= remove;
  }
  return plan;
}

/** Oversample capacity used by the capturable path. */
export function oversampleCount(n: number, factor = 1.5): number {
  return Math.max(n, Math.ceil(factor * n));
}
