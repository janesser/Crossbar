/**
 * Factory-level integration: session-start re-probe honesty (G2).
 *
 * Calls the REAL default-exported `crossbar(pi)` factory with:
 *  - a fake `pi` (captured `on` handlers, `registerProvider` spy,
 *    `unregisterProvider`/`registerCommand`/`setStatus`/`ui.register` no-ops),
 *  - a fake `ctx` (hasUI, `ui.notify` spy, `ui.setStatus` no-op, `ui.theme` passthrough),
 *  - an isolated tmp agent dir (`PI_CODING_AGENT_DIR`) whose `crossbar.json` holds one
 *    enabled llamacpp server, and
 *  - a real `node:http` mock llama.cpp server on 127.0.0.1:0 serving `/v1/models` + `/props`.
 *
 * Covers the I/O & Edge-Case Matrix rows 8–12: offline with cache (row 8), offline with
 * empty cache (row 9), the multi-change notice cap (row 10), headless (row 11), and the
 * change notice against the persisted catalogue (rows 1/3/12).
 *
 * Note: `session_start` also runs `discoverLocalhost`, which on a dev machine may find
 * REAL local backends (e.g. ollama on 11434). Every assertion is scoped to the fixture
 * server's provider id / label — never on total call counts — and the fixture config sets
 * `autoRegisterLocalhost: false` so real backends are never registered or re-probed here.
 */

import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createServer, type Server } from "node:http";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import type { AddressInfo } from "node:net";
import type { ExtensionAPI, ExtensionContext, ProviderConfig } from "@earendil-works/pi-coding-agent";
import crossbar from "../src/index.ts";
import { serverId } from "../src/registry/ids.ts";
import type { ModelDescriptor, ServerRecord } from "../src/core/types.ts";

const T = 1_700_000_000_000;
const MODEL_ID = "qwen3.8-27b";
const LABEL = "fixture-llamacpp";

function withWindow(id: string, value: number, source: "declared" | "observed"): ModelDescriptor {
  return {
    id,
    name: id,
    input: ["text"],
    reasoning: false,
    contextWindow: value,
    contextWindowSource: source,
    contextWindowAt: T,
  };
}

interface Registration {
  id: string;
  config: ProviderConfig;
}

interface Notice {
  message: string;
  type: "info" | "warning" | "error" | undefined;
}

function makeFakePi(): {
  handlers: Map<string, (event: unknown, ctx: unknown) => unknown>;
  registrations: Registration[];
  pi: ExtensionAPI;
} {
  const handlers = new Map<string, (event: unknown, ctx: unknown) => unknown>();
  const registrations: Registration[] = [];
  const pi = {
    on: (event: string, handler: (event: unknown, ctx: unknown) => unknown) => {
      handlers.set(event, handler);
    },
    registerProvider: (id: string, config: ProviderConfig) => {
      registrations.push({ id, config });
    },
    unregisterProvider: () => {},
    registerCommand: () => {},
    setStatus: () => {},
    ui: { register: () => {} },
  } as unknown as ExtensionAPI;
  return { handlers, registrations, pi };
}

function makeFakeCtx(hasUI: boolean): { ctx: ExtensionContext; notices: Notice[] } {
  const notices: Notice[] = [];
  const ctx = {
    hasUI,
    mode: "tui",
    cwd: process.cwd(),
    model: undefined,
    signal: undefined,
    ui: {
      notify: (message: string, type?: "info" | "warning" | "error") => {
        notices.push({ message, type });
      },
      setStatus: () => {},
      theme: { fg: (_token: string, text: string) => text },
    },
  } as unknown as ExtensionContext;
  return { ctx, notices };
}

// ---------------------------------------------------------------------------
// Mock llama.cpp server (real node:http on 127.0.0.1:0)
// ---------------------------------------------------------------------------

async function startMockLlamacpp(
  modelsData: Array<Record<string, unknown>>,
  props: Record<string, unknown>,
): Promise<{ server: Server; baseUrl: string }> {
  const server = createServer((req, res) => {
    if (req.url === "/v1/models") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ data: modelsData }));
    } else if (req.url === "/props") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify(props));
    } else {
      res.writeHead(404, { "content-type": "application/json" });
      res.end("{}");
    }
  });
  await new Promise<void>((resolve) => {
    server.listen(0, "127.0.0.1", () => resolve());
  });
  const { port } = server.address() as AddressInfo;
  return { server, baseUrl: `http://127.0.0.1:${port}` };
}

/** A port that is (almost certainly) not listening — for the offline cases. */
function findDeadPort(): Promise<number> {
  return new Promise((resolve) => {
    const s = createServer();
    s.listen(0, "127.0.0.1", () => {
      const { port } = s.address() as AddressInfo;
      s.close(() => resolve(port));
    });
  });
}

// ---------------------------------------------------------------------------
// Fixture + runner
// ---------------------------------------------------------------------------

let agentDir: string;
let prevAgentDir: string | undefined;
const openServers: Server[] = [];

beforeEach(() => {
  agentDir = mkdtempSync(join(tmpdir(), "crossbar-reprobe-"));
  prevAgentDir = process.env.PI_CODING_AGENT_DIR;
  process.env.PI_CODING_AGENT_DIR = agentDir;
});

afterEach(async () => {
  for (const s of openServers.splice(0)) {
    await new Promise<void>((resolve) => s.close(() => resolve()));
  }
  if (prevAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
  else process.env.PI_CODING_AGENT_DIR = prevAgentDir;
  rmSync(agentDir, { recursive: true, force: true });
});

/** Write crossbar.json (one enabled llamacpp server, auto-register off) and return the record. */
function writeFixture(agentDir: string, baseUrl: string, lastKnownModels: ModelDescriptor[]): ServerRecord {
  const record: ServerRecord = {
    id: serverId("llamacpp", baseUrl),
    kind: "llamacpp",
    baseUrl,
    label: LABEL,
    auth: "none",
    enabled: true,
    addedAt: T,
    lastKnownModels,
  };
  const cfg = {
    version: 1,
    servers: [record],
    // Keep real local backends (found by discoverLocalhost) out of registration so
    // the suite is deterministic on any dev machine.
    settings: { autoRegisterLocalhost: false },
  };
  writeFileSync(join(agentDir, "crossbar.json"), JSON.stringify(cfg, null, 2), "utf8");
  return record;
}

async function runSessionStart(opts: {
  baseUrl: string;
  lastKnownModels: ModelDescriptor[];
  hasUI?: boolean;
}): Promise<{ record: ServerRecord; registrations: Registration[]; notices: Notice[] }> {
  const record = writeFixture(agentDir, opts.baseUrl, opts.lastKnownModels);
  const fake = makeFakePi();
  const { ctx, notices } = makeFakeCtx(opts.hasUI ?? true);

  await crossbar(fake.pi); // factory: preload + handler registration
  const start = fake.handlers.get("session_start");
  expect(start).toBeDefined();
  await start!(undefined, ctx);

  return { record, registrations: fake.registrations, notices };
}

/** The last registration for a provider id (session_start's re-registration is authoritative). */
function lastRegistration(registrations: Registration[], id: string): Registration | undefined {
  const all = registrations.filter((r) => r.id === id);
  return all[all.length - 1];
}

function fixtureNotices(notices: Notice[], label: string): Notice[] {
  return notices.filter((n) => n.message.includes(`Crossbar: ${label}`));
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

describe("session-start re-probe (factory-level, mock llama.cpp server)", () => {
  it("live window differs from persisted (32768/declared → 234496/observed): one change notice, live re-registration", async () => {
    const { server, baseUrl } = await startMockLlamacpp(
      [{ id: MODEL_ID, meta: { n_ctx: 234496, n_ctx_train: 262144 } }],
      { default_generation_settings: { n_ctx: 32768 }, build_info: {} },
    );
    openServers.push(server);

    const persisted = [withWindow(MODEL_ID, 32768, "declared")];
    const { record, registrations, notices } = await runSessionStart({ baseUrl, lastKnownModels: persisted });

    const changed = notices.filter((n) => n.message.includes("context window changed"));
    expect(changed).toHaveLength(1);
    expect(changed[0]?.message).toBe(
      `Crossbar: ${LABEL} — context window changed — ${MODEL_ID}: 32768 → 234496 ctx.`,
    );
    expect(changed[0]?.type).toBe("info");

    // The frozen persisted number is no longer what Pi budgets against.
    const reg = lastRegistration(registrations, record.id);
    expect(reg?.config.models).toHaveLength(1);
    expect(reg?.config.models?.[0]?.contextWindow).toBe(234496);
    expect(fixtureNotices(notices, LABEL).filter((n) => n.message.includes("unreachable"))).toHaveLength(0);
  });

  it("live window identical to persisted: no window-change notice (no startup noise)", async () => {
    const { server, baseUrl } = await startMockLlamacpp(
      [{ id: MODEL_ID, meta: { n_ctx: 234496, n_ctx_train: 262144 } }],
      { default_generation_settings: { n_ctx: 32768 }, build_info: {} },
    );
    openServers.push(server);

    const persisted = [withWindow(MODEL_ID, 234496, "observed")];
    const { record, registrations, notices } = await runSessionStart({ baseUrl, lastKnownModels: persisted });

    expect(notices.filter((n) => n.message.includes("context window changed"))).toHaveLength(0);
    const reg = lastRegistration(registrations, record.id);
    expect(reg?.config.models?.[0]?.contextWindow).toBe(234496);
  });

  it("backend offline with cache: model stays registered at the 8192 Pi boundary + one offline notice", async () => {
    const deadPort = await findDeadPort();
    const persisted = [withWindow(MODEL_ID, 234496, "observed")];
    const { record, registrations, notices } = await runSessionStart({
      baseUrl: `http://127.0.0.1:${deadPort}`,
      lastKnownModels: persisted,
    });

    // Browsable/selectable, but windowless at the descriptor level: the Pi boundary
    // (toPiModel) emits the uniform 8192 — not the frozen 234496 from the cache.
    const reg = lastRegistration(registrations, record.id);
    expect(reg).toBeDefined();
    expect(reg?.config.models).toHaveLength(1);
    expect(reg?.config.models?.[0]?.contextWindow).toBe(8192);

    const offline = fixtureNotices(notices, LABEL).filter((n) => n.message.includes("unreachable"));
    expect(offline).toHaveLength(1);
    expect(offline[0]?.message).toBe(
      `Crossbar: ${LABEL} unreachable — 1 model(s) kept from cache with unknown window (8192 fallback) until it's back.`,
    );
    expect(offline[0]?.type).toBe("info");
  });

  it("backend offline with empty cache: nothing registered, no notice (early return 0)", async () => {
    const deadPort = await findDeadPort();
    const { record, registrations, notices } = await runSessionStart({
      baseUrl: `http://127.0.0.1:${deadPort}`,
      lastKnownModels: [],
    });

    expect(registrations.filter((r) => r.id === record.id)).toHaveLength(0);
    expect(fixtureNotices(notices, LABEL)).toHaveLength(0);
  });

  it("7 windows changed on one server: single notice with the first 5 changes + '+2 more'", async () => {
    const liveData = Array.from({ length: 7 }, (_, i) => ({
      id: `model-${i}`,
      meta: { n_ctx: 32768 * (i + 1) },
    }));
    const { server, baseUrl } = await startMockLlamacpp(liveData, {
      default_generation_settings: { n_ctx: 8192 },
      build_info: {},
    });
    openServers.push(server);

    const persisted = Array.from({ length: 7 }, (_, i) => withWindow(`model-${i}`, 8192, "declared"));
    const { record, registrations, notices } = await runSessionStart({ baseUrl, lastKnownModels: persisted });

    const changed = notices.filter((n) => n.message.includes("context window changed"));
    expect(changed).toHaveLength(1);
    expect(changed[0]?.message).toBe(
      `Crossbar: ${LABEL} — context window changed — ` +
        [
          "model-0: 8192 → 32768 ctx",
          "model-1: 8192 → 65536 ctx",
          "model-2: 8192 → 98304 ctx",
          "model-3: 8192 → 131072 ctx",
          "model-4: 8192 → 163840 ctx",
        ].join("; ") + " (+2 more).",
    );
    expect(changed[0]?.type).toBe("info");

    // All 7 re-registered with their live windows.
    const reg = lastRegistration(registrations, record.id);
    expect(reg?.config.models).toHaveLength(7);
    expect(reg?.config.models?.[0]?.contextWindow).toBe(32768);
    expect(reg?.config.models?.[6]?.contextWindow).toBe(32768 * 7);
  });

  it("headless (hasUI false): identical registration, no notices", async () => {
    const { server, baseUrl } = await startMockLlamacpp(
      [{ id: MODEL_ID, meta: { n_ctx: 234496, n_ctx_train: 262144 } }],
      { default_generation_settings: { n_ctx: 32768 }, build_info: {} },
    );
    openServers.push(server);

    const persisted = [withWindow(MODEL_ID, 32768, "declared")];
    const { record, registrations, notices } = await runSessionStart({
      baseUrl,
      lastKnownModels: persisted,
      hasUI: false,
    });

    expect(notices).toHaveLength(0);
    const reg = lastRegistration(registrations, record.id);
    expect(reg?.config.models?.[0]?.contextWindow).toBe(234496);
  });
});
