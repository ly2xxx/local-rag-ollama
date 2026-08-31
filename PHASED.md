# Phased Delivery Log

Running record of what has actually been built against [`DESIGN.md`](DESIGN.md).
Each entry states what shipped, whether the phase's Definition of Done is met,
what the tests prove, and where the implementation deviates from the design.

**Rule for this file:** a checkbox is ticked only when something was run and
observed. Anything believed-but-unverified is written as a deviation, not a tick.

| Phase | Theme                                                     | Status         |
| ----- | --------------------------------------------------------- | -------------- |
| 0     | Hardening                                                 | ✅ Complete    |
| 1     | FastAPI core engine                                       | ✅ Complete    |
| 2     | LLM layer: streaming, structured output, function calling | ⬜ Not started |
| 3     | RAG core: Qdrant, rerank, citations                       | ⬜ Not started |
| 4     | LangGraph StateGraph + human-in-the-loop                  | ⬜ Not started |
| 5     | Enterprise infrastructure                                 | ⬜ Not started |
| 6     | Complete system on Rancher                                | ⬜ Not started |

---

## Phase 0 — Hardening ✅

Closed the defects that would fail a first code review, before adding tiers on top.

| Fix                | What changed                                                                                                                                  | Evidence                                                                                                                                                             |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Calculator sandbox | `eval` → `safe_eval_expression`, a strict AST whitelist with caps on length, node count and exponent                                     | The previous`{"__builtins__": {}}` sandbox leaked **4902 classes** via `(1).__class__.__base__.__subclasses__()`; all 8 known escape payloads now rejected |
| Tenant isolation   | Global`doc_manager` → `DocumentStoreRegistry`, one Chroma collection per namespace, bound through `contextvars` in `AgentCore.run`   | The LLM cannot name a namespace — it is not a tool argument                                                                                                         |
| Durability         | Collections persist to`CHROMA_PERSIST_DIR`; deterministic chunk IDs make re-ingest an upsert; file listing derived from collection metadata | Survives restart; re-ingest holds at 58 chunks instead of doubling                                                                                                   |
| Answer extraction  | `_extract_final_answer` scans backwards past tool-call messages; flattens list-of-blocks content                                            | Pre-tool reasoning text no longer overwrites the answer                                                                                                              |
| File tool scoping  | `read_local_file` confined to `FILE_TOOL_ROOTS`                                                                                           | Ingested documents are untrusted input; the tool can no longer walk the host                                                                                         |

Originally verified by a scratchpad script. **Phase 1 promoted these into permanent
regression tests** at `tests/unit/test_phase0_hardening.py` (DESIGN §6.5).

---

## Phase 1 — FastAPI core engine ✅

**Capability:** async, Pydantic v2 schemas, JSON validation, SSE plumbing, tenancy.
**Cluster component (Phase 6):** `agent-api` Deployment + ClusterIP Service.

### What shipped

#### API tier — `agentic_rag/api/`

| Module                  | Responsibility                                                                     |
| ----------------------- | ---------------------------------------------------------------------------------- |
| `main.py`             | App factory, lifespan, middleware wiring, four exception handlers                  |
| `routes/chat.py`      | `POST /api/v1/chat`, `POST /api/v1/chat/stream`                                |
| `routes/documents.py` | `POST` / `GET` / `DELETE /api/v1/documents`                                  |
| `routes/health.py`    | `GET /healthz` (liveness), `GET /readyz` (readiness)                           |
| `schemas.py`          | Pydantic v2 request/response models                                                |
| `errors.py`           | The DESIGN §5.4 taxonomy: code → status, retryability, envelope                  |
| `sse.py`              | DESIGN §5.2 event envelope, framing, chunking, preview truncation                 |
| `service.py`          | `AgentService` protocol + `AgenticRAGService`; thread offload; tenancy scoping |
| `deps.py`             | API-key auth,`Principal`, settings/service accessors                             |
| `middleware.py`       | `trace_id` binding and access logging                                            |
| `tracing.py`          | Sortable trace ids, inbound`X-Request-Id` handling                               |

`agentic_rag/settings.py` adds a typed `Settings` (pydantic-settings) for the new tier.

#### Test scaffolding — `tests/`

`conftest.py` (fixtures + the outbound-network guard), `fakes/` (fake service,
fake helper, deterministic embeddings, scripted tool-calling chat model), and
`unit/` · `contract/` · `integration/` suites. Markers, `asyncio_mode = auto`,
and the coverage gate live in `pyproject.toml`.

#### Supporting

- `scripts/dump_openapi.py` — regenerates the OpenAPI contract snapshot
- `.github/workflows/ci.yml` — fast suite → integration → coverage gate → snapshot freshness → Streamlit import check
- `.env.example` — every new setting documented

### Key decisions made during implementation

**Tenancy is a composite scope, not just a thread id.** `scope_id(tenant, thread)`
produces `"tenant_a::session_1"`, used as *both* the vector-store namespace and
the LangGraph checkpoint `thread_id`. Without the tenant in the checkpoint key,
two tenants both using `session_1` would have shared conversation state — a leak
the DESIGN's data-isolation rule implies but does not spell out.
`tests/unit/test_service.py::test_same_thread_id_across_tenants_yields_different_scopes`
pins it.

**Auth fails closed.** No configured `API_KEYS` means every authenticated
endpoint returns 401. Startup logs a loud warning so a developer does not
mistake it for a broken build. Keys are compared with `hmac.compare_digest`
against every candidate, so timing cannot reveal a valid prefix.

**Validation errors are 400, not FastAPI's default 422.** The taxonomy says 400;
a `RequestValidationError` handler enforces it. 422 stays reserved for
`GUARDRAIL_BLOCKED` in Phase 5.

**Routes are thin on purpose.** They validate, resolve the principal, and call
the service. Phase 4 replaces the agent with an explicit `StateGraph` behind the
same `AgentService` protocol — no HTTP code should need to change.

**The sync core is offloaded, not rewritten.** `anyio.to_thread.run_sync` keeps
the event loop free and copies the context, which the Phase 0 namespace binding
depends on. Phase 4 makes the graph natively async.

### Definition of Done

| DoD item                                                                     | Status | Evidence                                                                                                                                                                                               |
| ---------------------------------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `uvicorn agentic_rag.api.main:app` serves all endpoints; `/docs` renders | ✅     | Live smoke against real uvicorn:`/healthz`, `/readyz`, `/docs` (Swagger), `/openapi.json` all 200                                                                                              |
| `curl -N .../chat/stream` prints tokens incrementally                      | ✅     | Real socket, 11 reads, first frame at**5–46 ms**, last at **~0.4 s** while the agent worked for 0.4 s — measured across sync/async clients and `iter_text`/`iter_raw`/`iter_bytes` |
| Malformed body → 400 with`INVALID_REQUEST`, no stack trace                | ✅     | `test_blank_query_is_400_not_422`, `test_internal_error_never_leaks_traceback`                                                                                                                     |
| Missing/bad API key → 401; tenant A cannot read tenant B's documents        | ✅     | `test_chat_is_401_without_api_key`, `test_tenant_cannot_read_other_tenant_documents`, and through real Chroma in `test_documents_are_isolated_between_tenants_through_http`                      |
| Unhandled exception → 500 with a`trace_id` that appears in the logs       | ✅     | `test_internal_error_never_leaks_traceback` — asserts the secret string and `Traceback` are both absent from the body                                                                             |
| `/readyz` 503 when Redis is stopped, 200 when it returns                   | ✅     | `test_readyz_detects_unreachable_redis` (dead port) and `test_readyz_ok_against_live_redis` (live Redis) — see deviation D-3                                                                      |
| Streamlit tab 3 still works unchanged                                        | ✅     | `app.py` untouched this phase; compiles and `agentic_rag` imports clean                                                                                                                            |
| CI green; coverage ≥ 80 %; OpenAPI snapshot committed                       | ✅     | **147 tests pass. Coverage 91.3 % (fast suite) / 97.2 % (combined).** Snapshot at `tests/contract/openapi_snapshot.json`                                                                       |

### Test inventory

| Suite                   |         Tests | Runtime | Needs                                 |
| ----------------------- | ------------: | ------- | ------------------------------------- |
| `unit` + `contract` |           135 | ~24 s   | Nothing — fully hermetic             |
| `integration`         |            12 | ~46 s   | Real Redis, real Chroma, real uvicorn |
| **Total**         | **147** |         |                                       |

```
tests/
├── conftest.py                              fixtures + outbound-network guard
├── fakes/                                   service, helper, embeddings, tool-calling model
├── unit/
│   ├── test_schemas.py                      blank query, thread-id traversal, extra="forbid"
│   ├── test_errors.py                       full taxonomy table, no-leak assertion
│   ├── test_sse.py                          framing, newline escaping, chunking, truncation
│   ├── test_service.py                      scoping, failure classification, degradation
│   └── test_phase0_hardening.py             8 sandbox escapes, isolation, idempotency, restart
├── contract/
│   ├── test_chat_endpoint.py                auth, tenancy, envelope, trace propagation
│   ├── test_stream_endpoint.py              event order, heartbeats, in-band errors
│   ├── test_documents_endpoint.py           upload limits, filename stripping, scoping
│   ├── test_health.py                       liveness independence, readiness transitions
│   └── test_openapi_snapshot.py             contract freeze
└── integration/
    ├── test_lifespan.py                     live Redis probe, embedding cache, lifespan
    ├── test_api_end_to_end.py               real Chroma isolation through HTTP
    └── test_streaming_live.py               real uvicorn, real socket, incremental arrival
```

Coverage after the combined run:

```
agentic_rag\api\deps.py            95%     agentic_rag\api\schemas.py     100%
agentic_rag\api\errors.py         100%     agentic_rag\api\service.py      96%
agentic_rag\api\main.py            96%     agentic_rag\api\sse.py          97%
agentic_rag\api\middleware.py     100%     agentic_rag\api\tracing.py     100%
agentic_rag\api\routes\chat.py     96%     agentic_rag\settings.py         95%
agentic_rag\api\routes\documents.py 100%   ---------------------------------
agentic_rag\api\routes\health.py  100%     TOTAL                           97%
```

### Deviations from DESIGN.md

**D-1 · SSE is framing-complete, not generation-streaming.**
Phase 1 runs the agent to completion, then emits `node` → `tool_call` →
`observation` → chunked `token` → `done`, with heartbeats during the wait. Frames
genuinely reach the client incrementally (measured above), but the *text* is
chunked from a finished answer. Phase 2 replaces the body of `_stream_turn` with
`astream_events`; **the wire format does not change**, so no client or test needs
updating. Flagged because "streaming" could otherwise be over-claimed.

**D-2 · Two error codes added to the §5.4 taxonomy.**
`NOT_FOUND` (404) and `PAYLOAD_TOO_LARGE` (413). Needed by the documents
endpoints and by unknown-route handling. `tests/unit/test_errors.py` restates the
whole table so the doc and the code cannot drift apart silently.
*Please confirm this amendment.*

**D-3 · `test_redis_reconnect_after_bounce` replaced.**
Bouncing a container from inside a test is unreliable and slow. Substituted two
tests that prove the same property — that the probe is live rather than a cached
startup value: `test_readyz_detects_unreachable_redis` (dead port → 503) and
`test_readyz_ok_against_live_redis` (real Redis → 200), plus
`test_readyz_recovers_when_dependency_returns` at the contract layer.

**D-4 · Incremental-arrival test moved out of the contract suite.**
`httpx.ASGITransport` buffers the whole response body, so it *cannot* observe
streaming — the originally-planned contract test would have passed or failed for
reasons unrelated to the server. Replaced by an assertion on the async generator
itself (fast, deterministic) plus `tests/integration/test_streaming_live.py`,
which runs real uvicorn on a real socket. This is the honest home for that DoD item.

**D-5 · Coverage gate scoped to the new tier.**
`agentic_rag/api/*` and `settings.py`, not the whole package. The legacy modules
(`evaluation.py`, `judge/`, `graph_display.py`, and the placeholders) have no
tests, and a package-wide 80 % gate would have failed on day one for code Phase 1
did not touch. Phases 3–5 rewrite most of it; the gate widens as they land.

**D-6 · `config.py` constants kept alongside `Settings`.**
DESIGN §5.6 wants one settings object and no `os.getenv` outside `config.py`.
`Settings` covers the API tier and reads the same env vars, but `tools.py`,
`core.py` and `agent.py` still import the module-level constants. Unifying them
is a refactor of Phase 0 code with no test coverage behind it yet — deferred to
Phase 2, when `llm/provider.py` needs the same values.

**D-7 · `trace_id` is ULID-*like*, not a ULID.**
`{ms timestamp:012x}{random:16}` — lexicographically sortable by arrival time,
which is the property that matters for log grepping, without adding a dependency.

**D-8 · Readiness vs. degradation tension, made configurable.**
DESIGN §5.5 says Redis loss is degradation; the Phase 1 DoD says `/readyz` must
fail while Redis is down. Both are honoured via `readyz_require_redis` (default
`true`). Phase 5 should settle which is right once rate limiting and the
semantic cache also depend on Redis.

### Fixed during implementation

- **Windows event-loop vs. the network guard.** `ProactorEventLoop` builds its
  self-pipe with `socket.socketpair()`, which falls back to a real loopback
  `connect()` — the guard was failing 45 tests before a test ran. The guard now
  exempts `socketpair` specifically rather than whitelisting loopback, which
  would have defeated it (Redis and Ollama are both on localhost in dev).
- **Lifespan not exercised by `ASGITransport`.** It skips the lifespan entirely,
  so `app.state.service` stayed `None`. The integration test now enters
  `app.router.lifespan_context(app)` explicitly.

### Running it

Start the API:

```bash
uv run uvicorn agentic_rag.api.main:app --reload --port 8000
```

Fast inner loop (hermetic, ~24 s):

```bash
uv run pytest
```

Integration suite (needs Redis):

```bash
uv run pytest -m integration -o addopts=""
```

Watch a stream arrive token by token:

```bash
curl -N -X POST http://127.0.0.1:8000/api/v1/chat/stream -H "X-API-Key: dev-key-local" -H "Content-Type: application/json" -d "{\"query\":\"what is in my documents?\",\"thread_id\":\"session_1\"}"
```

Regenerate the OpenAPI contract after an intentional change:

```bash
uv run python scripts/dump_openapi.py
```

### Dependencies added

Runtime: `fastapi`, `uvicorn[standard]`, `python-multipart`, `pydantic-settings`.
Dev: `pytest`, `pytest-asyncio`, `pytest-cov`, `httpx`, `fakeredis`.

`fakeredis` is unused so far — it lands here because DESIGN §6.2 lists it in the
shared fake set and Phase 5's rate-limiter tests need it.

### Open items carried into Phase 2

1. Confirm deviation **D-2** (two added error codes).
2. Decide **D-8** (readiness vs. degradation for Redis) before Phase 5 adds two
   more Redis-dependent features.
3. Fold `config.py` into `Settings` (**D-6**) when `llm/provider.py` arrives.
4. Replace `classify_agent_failure`'s string matching with typed provider
   exceptions — it is a deliberate stopgap and is marked as such in the code.
5. The five questions in DESIGN §15 are still open; **§15 Q3 (who writes the
   50–100 golden Q/A pairs, against which corpus) is on the critical path for
   Phase 3** and should be settled before Phase 2 finishes.
