/**
 * Window provenance — the (window, source) contract for context-window reporting.
 *
 * Every context window crossbar knows about carries a source label — `declared` or `observed`
 * (see `WindowProvenance` in `types.ts`) — and a trained ceiling never becomes a serviceable
 * window. `ModelDescriptor.contextWindow` is set iff `contextWindowSource` is set; the
 * conformance suite enforces the pairing for every adapter.
 *
 * The Pi boundary: Pi's model type REQUIRES a numeric `contextWindow`, so an unknown window
 * cannot be omitted at registration. Every adapter's `toPiModel` emits the uniform,
 * most-conservative legacy value {@link UNKNOWN_CONTEXT_WINDOW} for unknown windows — the
 * descriptor and registry still report unknown, only the mandatory boundary number is filled.
 */

import type { ModelDescriptor } from "./types.ts";

/**
 * The single uniform context window `toPiModel` emits when the effective window is unknown.
 * 8192 is the most conservative value among the previous per-adapter fallbacks (it was the
 * default in 5 of 7 adapters), so the switch from per-adapter values is safe-direction
 * only — understated, never overstated.
 */
export const UNKNOWN_CONTEXT_WINDOW = 8192 as const;

/** True when the descriptor carries a well-formed (window, source) pair. */
export function hasProvenancedWindow(model: ModelDescriptor): boolean {
  return (
    typeof model.contextWindow === "number" &&
    Number.isSafeInteger(model.contextWindow) &&
    model.contextWindow > 0 &&
    (model.contextWindowSource === "declared" || model.contextWindowSource === "observed")
  );
}

/**
 * The context window a `toPiModel` must emit: the descriptor's window (positive safe integer,
 * paired or legacy-bare) when present, else {@link UNKNOWN_CONTEXT_WINDOW}. Pairing is
 * enforced on adapter OUTPUT (listModels) and on cache migration — this boundary accepts
 * what the descriptor holds and never invents a different number.
 */
export function piContextWindow(model: ModelDescriptor): number {
  const window = model.contextWindow;
  return typeof window === "number" && Number.isSafeInteger(window) && window > 0
    ? window
    : UNKNOWN_CONTEXT_WINDOW;
}

/**
 * One per-model context-window change detected by the resume re-probe.
 *
 * `from`/`to` are the effective windows in the persisted (previous) and live (next)
 * catalogues respectively — `undefined` means unknown (no serviceable window).
 * `name` is taken from the live (next) descriptor so the notice shows the current name.
 */
export interface WindowChange {
  id: string;
  name: string;
  from: number | undefined;
  to: number | undefined;
}

/**
 * Diff the effective context window per model id between two catalogues — the
 * resume re-probe's change detection.
 *
 * Compares `contextWindow` for chat models (`!embeddings`) whose id is present in
 * BOTH catalogues: a model that appears or disappears is a membership change
 * (`catalogueChanged`'s concern, never a window-change notice). `name` comes from
 * the live (next) descriptor.
 *
 * Callers pass the MERGED catalogue as `next`, so a live model the merge filled
 * from the persisted cache compares equal and produces no false notice (the merge
 * already handles that direction).
 *
 * PURE — no I/O, no mutation of the inputs.
 */
export function windowChanges(
  prev: ModelDescriptor[] | undefined,
  next: ModelDescriptor[],
): WindowChange[] {
  const prevById = new Map<string, ModelDescriptor>();
  for (const p of prev ?? []) {
    if (typeof p === "object" && p !== null && typeof p.id === "string") {
      prevById.set(p.id, p);
    }
  }

  const changes: WindowChange[] = [];
  for (const n of next) {
    if (n.embeddings) continue; // embedding windows are never budgeted — ignored
    const p = prevById.get(n.id);
    if (p === undefined) continue; // not in both — membership change, not a window change
    if (p.contextWindow !== n.contextWindow) {
      changes.push({ id: n.id, name: n.name, from: p.contextWindow, to: n.contextWindow });
    }
  }
  return changes;
}

/**
 * The offline downgrade: drop the serviceable window triplet
 * (`contextWindow`/`contextWindowSource`/`contextWindowAt`) from every descriptor so a
 * cached catalogue registers in the unknown state — the Pi boundary then emits the
 * uniform {@link UNKNOWN_CONTEXT_WINDOW} instead of budgeting a frozen number.
 * `trainedMax` and all other fields survive (informational, never budgeted).
 *
 * Satisfies the pairing invariant (window set iff source set) in the unknown state.
 * PURE — returns a new array; inputs are not mutated.
 */
export function stripWindows(models: ModelDescriptor[]): ModelDescriptor[] {
  return models.map(({ contextWindow: _w, contextWindowSource: _s, contextWindowAt: _t, ...rest }) => rest);
}

/**
 * Merge a persisted (cached) catalogue with a freshly listed one, applying the
 * precedence ladder per model.
 *
 * Ladder: live evidence beats persisted evidence, and persisted evidence fills models the
 * live list reports no window for. "Live evidence" is whatever the adapter's own chain
 * produced — each adapter already applies declared > observed internally (e.g. llama.cpp's
 * pinned `-c` beats the loaded runtime `n_ctx`) — so a live descriptor always outranks a
 * persisted one regardless of the persisted source.
 *
 * Models are returned in the live list's order and membership: a model absent from the live
 * list is never resurrected (registration only ever reflects the live catalogue), and a
 * model present live keeps its live `id`/`name`/caps — only the window fields (and, when the
 * live list is silent, `trainedMax`) may come from the cache.
 *
 * PURE — no I/O, no mutation of the inputs.
 */
export function mergeCatalogue(
  persisted: ModelDescriptor[] | undefined,
  live: ModelDescriptor[],
): ModelDescriptor[] {
  const persistedById = new Map<string, ModelDescriptor>();
  for (const p of persisted ?? []) {
    if (typeof p === "object" && p !== null && typeof p.id === "string") {
      persistedById.set(p.id, p);
    }
  }

  return live.map((m) => {
    if (hasProvenancedWindow(m)) return m;

    const p = persistedById.get(m.id);
    const pWindow = p?.contextWindow;
    const pSource = p?.contextWindowSource;
    if (
      p &&
      typeof pWindow === "number" &&
      Number.isSafeInteger(pWindow) &&
      pWindow > 0 &&
      (pSource === "declared" || pSource === "observed")
    ) {
      const out: ModelDescriptor = {
        ...m,
        contextWindow: pWindow,
        contextWindowSource: pSource,
      };
      if (p.contextWindowAt !== undefined) out.contextWindowAt = p.contextWindowAt;
      if (out.trainedMax === undefined && p.trainedMax !== undefined) {
        out.trainedMax = p.trainedMax;
      }
      return out;
    }
    if (p && m.trainedMax === undefined && p.trainedMax !== undefined) {
      return { ...m, trainedMax: p.trainedMax };
    }
    return m;
  });
}
