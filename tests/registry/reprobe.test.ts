/**
 * Resume re-probe (G2): pure change detection over two catalogues
 * (`windowChanges`) and the offline downgrade (`stripWindows`).
 *
 * Covers the I/O & Edge-Case Matrix rows 1–7 of spec-2-3:
 *   1. window changed (declared → observed)
 *   2. window changed (shrank)
 *   3. window unchanged
 *   4. model gains window (unknown → known)
 *   5. model "loses" window (persisted fills live — merge already handled it)
 *   6. membership change only
 *   7. embedding rows ignored
 */

import { describe, expect, it } from "vitest";
import { mergeCatalogue, stripWindows, windowChanges } from "../../src/core/window-provenance.ts";
import type { ModelDescriptor } from "../../src/core/types.ts";

const T = 1_700_000_000_000;

function model(id: string, overrides: Partial<ModelDescriptor> = {}): ModelDescriptor {
  const base: ModelDescriptor = { id, name: id, input: ["text"], reasoning: false };
  return { ...base, ...overrides };
}

function withWindow(m: ModelDescriptor, value: number, source: "declared" | "observed"): ModelDescriptor {
  return { ...m, contextWindow: value, contextWindowSource: source, contextWindowAt: T };
}

// ---------------------------------------------------------------------------
// windowChanges — per-id effective-window diff (matrix rows 1–7)
// ---------------------------------------------------------------------------

describe("windowChanges — change detection", () => {
  it("row 1: window changed (declared → observed) — one change with both values", () => {
    const prev = [withWindow(model("m"), 32768, "declared")];
    const next = [withWindow(model("m"), 234496, "observed")];
    expect(windowChanges(prev, next)).toEqual([
      { id: "m", name: "m", from: 32768, to: 234496 },
    ]);
  });

  it("row 2: window changed (shrank) — one change with both values", () => {
    const prev = [withWindow(model("m"), 234496, "observed")];
    const next = [withWindow(model("m"), 8192, "declared")];
    expect(windowChanges(prev, next)).toEqual([
      { id: "m", name: "m", from: 234496, to: 8192 },
    ]);
  });

  it("row 3: window unchanged — no change (no startup noise)", () => {
    const prev = [withWindow(model("m"), 234496, "observed")];
    const next = [withWindow(model("m"), 234496, "observed")];
    expect(windowChanges(prev, next)).toEqual([]);
  });

  it("row 3b: same value, different source label — still no change (effective window compared)", () => {
    const prev = [withWindow(model("m"), 234496, "declared")];
    const next = [withWindow(model("m"), 234496, "observed")];
    expect(windowChanges(prev, next)).toEqual([]);
  });

  it("row 4: model gains window (persisted unknown, live known) — change from undefined", () => {
    const prev = [model("m")];
    const next = [withWindow(model("m"), 234496, "observed")];
    expect(windowChanges(prev, next)).toEqual([
      { id: "m", name: "m", from: undefined, to: 234496 },
    ]);
  });

  it("row 5: model loses window (persisted fills live) — merged keeps persisted, no change", () => {
    // refreshAndRegister diffs the MERGED catalogue: the merge already filled the
    // live (windowless) model from the persisted cache, so the effective window
    // is unchanged and no notice fires.
    const prev = [withWindow(model("m"), 234496, "observed")];
    const live = [model("m")];
    const merged = mergeCatalogue(prev, live);
    expect(merged[0]?.contextWindow).toBe(234496); // merge filled from cache
    expect(windowChanges(prev, merged)).toEqual([]);
  });

  it("row 5b: true downgrade (live reports unknown, merge cannot fill) — change to undefined", () => {
    // A persisted window with no live counterpart to fill is a real downgrade:
    // diffing against the raw live list, the change is from known to unknown.
    const prev = [withWindow(model("m"), 234496, "observed")];
    const live = [model("m")];
    expect(windowChanges(prev, live)).toEqual([
      { id: "m", name: "m", from: 234496, to: undefined },
    ]);
  });

  it("row 6: membership change only — new model appears, existing windows identical, no change", () => {
    const prev = [withWindow(model("a"), 234496, "observed")];
    const next = [
      withWindow(model("a"), 234496, "observed"),
      withWindow(model("b"), 32768, "declared"),
    ];
    expect(windowChanges(prev, next)).toEqual([]);
  });

  it("row 6b: model removed (in prev, absent from next) — no change", () => {
    const prev = [
      withWindow(model("a"), 234496, "observed"),
      withWindow(model("gone"), 32768, "declared"),
    ];
    const next = [withWindow(model("a"), 234496, "observed")];
    expect(windowChanges(prev, next)).toEqual([]);
  });

  it("row 7: embedding rows differ in window — ignored by change detection", () => {
    const prev = [
      withWindow(model("chat"), 234496, "observed"),
      withWindow(model("embed", { embeddings: true }), 8192, "observed"),
    ];
    const next = [
      withWindow(model("chat"), 234496, "observed"),
      withWindow(model("embed", { embeddings: true }), 65536, "observed"),
    ];
    expect(windowChanges(prev, next)).toEqual([]);
  });

  it("name comes from the live (next) descriptor", () => {
    const prev = [withWindow(model("m", { name: "Old Name" }), 32768, "declared")];
    const next = [withWindow(model("m", { name: "New Name" }), 234496, "observed")];
    expect(windowChanges(prev, next)).toEqual([
      { id: "m", name: "New Name", from: 32768, to: 234496 },
    ]);
  });

  it("multiple models changed — one entry per changed id, next's order", () => {
    const prev = [
      withWindow(model("a"), 32768, "declared"),
      withWindow(model("b"), 65536, "observed"),
      withWindow(model("c"), 131072, "observed"),
    ];
    const next = [
      withWindow(model("a"), 131072, "observed"),
      withWindow(model("b"), 65536, "observed"), // unchanged
      withWindow(model("c"), 32768, "declared"),
    ];
    expect(windowChanges(prev, next)).toEqual([
      { id: "a", name: "a", from: 32768, to: 131072 },
      { id: "c", name: "c", from: 131072, to: 32768 },
    ]);
  });

  it("accepts an undefined persisted catalogue and does not mutate its inputs", () => {
    const prev = [withWindow(model("m"), 32768, "declared")];
    const next = [withWindow(model("m"), 234496, "observed")];
    const before = JSON.stringify({ prev, next });
    expect(windowChanges(undefined, next)).toEqual([]); // nothing in both → no change
    windowChanges(prev, next);
    expect(JSON.stringify({ prev, next })).toBe(before);
  });
});

// ---------------------------------------------------------------------------
// stripWindows — the offline downgrade
// ---------------------------------------------------------------------------

describe("stripWindows — offline downgrade", () => {
  it("drops the window triplet (window, source, timestamp) from every descriptor", () => {
    const models = [
      withWindow(model("a"), 234496, "observed"),
      withWindow(model("b"), 32768, "declared"),
      model("c"), // already unknown
    ];
    const stripped = stripWindows(models);
    for (const m of stripped) {
      expect(m).not.toHaveProperty("contextWindow");
      expect(m).not.toHaveProperty("contextWindowSource");
      expect(m).not.toHaveProperty("contextWindowAt");
    }
  });

  it("keeps trainedMax and all other fields (informational, never budgeted)", () => {
    const models = [
      {
        ...withWindow(model("a"), 234496, "observed"),
        trainedMax: 262144,
        maxTokens: 8192,
        input: ["text", "image"] as ("text" | "image")[],
        tools: true,
        loaded: true,
      },
    ];
    const [stripped] = stripWindows(models);
    expect(stripped).toEqual({
      id: "a",
      name: "a",
      input: ["text", "image"],
      reasoning: false,
      trainedMax: 262144,
      maxTokens: 8192,
      tools: true,
      loaded: true,
    });
  });

  it("returns a new array and new objects — inputs are not mutated", () => {
    const models = [withWindow(model("a"), 234496, "observed")];
    const stripped = stripWindows(models);
    expect(stripped).not.toBe(models);
    expect(stripped[0]).not.toBe(models[0]);
    // The original cache is untouched (still paired, still usable for the merge ladder).
    expect(models[0]?.contextWindow).toBe(234496);
    expect(models[0]?.contextWindowSource).toBe("observed");
  });

  it("empty input yields an empty array", () => {
    expect(stripWindows([])).toEqual([]);
  });
});
