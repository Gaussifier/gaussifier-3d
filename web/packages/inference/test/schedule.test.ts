import { describe, expect, it } from "vitest";
import { jfaSchedule, mergePlan, oversampleCount } from "../src/placement/schedule.js";

describe("placement schedule", () => {
  it("matches the native jump-flood schedule", () => {
    expect(jfaSchedule(512, 512)).toEqual([256, 128, 64, 32, 16, 8, 4, 2, 1, 1]);
    expect(jfaSchedule(128, 128)).toEqual([64, 32, 16, 8, 4, 2, 1, 1]);
    expect(jfaSchedule(1, 1)).toEqual([1, 1]);
  });
  it("follows the native per-round cap with the 1.5 oversample", () => {
    const n = 84129; const nMax = oversampleCount(n);
    expect(nMax).toBe(126194);
    const plan = mergePlan(nMax, n, 6, 1 / 3);
    expect(plan).toHaveLength(6);
    // floor(126194 / 3) = 42064 leaves exactly one point for round one; that is the native rule.
    expect(plan[0]).toEqual({ nIn: nMax, remove: 42064 });
    expect(plan[1]).toEqual({ nIn: nMax - 42064, remove: 1 });
    for (const r of plan.slice(2)) expect(r.remove).toBe(0);
    expect(plan.reduce((a, r) => a + r.remove, 0)).toBe(nMax - n);
  });
  it("caps removals per round and finishes on the last round", () => {
    const plan = mergePlan(100, 10, 3, 1 / 3);
    expect(plan.map((r) => r.remove)).toEqual([33, 22, 35]);
    expect(plan.reduce((a, r) => a + r.remove, 0)).toBe(90);
  });
});
