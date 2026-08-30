# Enterprise Agentic RAG — Design & Delivery Plan

**Status:** Draft for review · **Owner:** @ly2xxx · **Target:** Rancher-managed k3d cluster (`rancher-cluster`, UI at `https://rancher.localhost:8443/`)

This document is the plan of record for growing `local-rag-ollama` from a single-process
Streamlit demo into a deployed, observable, multi-tier agent platform. It is **design only** —
no code in this repo has been changed by it.

Every phase carries a **Definition of Done** and a **planned pytest suite**. A phase is not
finished because the feature works in the UI; it is finished when its DoD checklist is green
and its tests pass in CI.

---

## Table of Contents

1. [Goals and non-goals](#1-goals-and-non-goals)
2. [Where we are today](#2-where-we-are-today)
3. [Target architecture](#3-target-architecture)
4. [Target repository layout](#4-target-repository-layout)
5. [Cross-cutting contracts](#5-cross-cutting-contracts)
6. [Test strategy](#6-test-strategy)
7. [Phase 1 — FastAPI core engine](#phase-1--fastapi-core-engine)
8. [Phase 2 — LLM layer: streaming, structured output, function calling](#phase-2--llm-layer-streaming-structured-output-function-calling)
9. [Phase 3 — RAG core: Qdrant, rerank, citations](#phase-3--rag-core-qdrant-rerank-citations)
10. [Phase 4 — LangGraph StateGraph and human-in-the-loop](#phase-4--langgraph-stategraph-and-human-in-the-loop)
11. [Phase 5 — Enterprise infrastructure](#phase-5--enterprise-infrastructure)
12. [Phase 6 — Complete system on Rancher](#phase-6--complete-system-on-rancher)
13. [Deployment topology](#13-deployment-topology)
14. [Demo script](#14-demo-script)
15. [Decision register and open questions](#15-decision-register-and-open-questions)

---

## 1. Goals and non-goals

### Goals

| # | Goal | How it is judged |
|---|---|---|
| G1 | A vertical business agent, not a chat demo | An end-to-end flow with retrieval, tool use, approval, and audit |
| G2 | Every answer is traceable to a source | Citation coverage measured, not asserted |
| G3 | Quality is a number, not an opinion | Retrieval + judge metrics tracked per release; CI blocks regressions |
| G4 | It runs as a real deployment | Multi-pod on Rancher, with probes, limits, secrets, ingress, dashboards |
| G5 | Failure is designed for | Typed error taxonomy, bounded retries, graceful degradation on every dependency |

### Non-goals

- Multi-agent supervisor topologies. Deliberately deferred — see [§15 D-7](#15-decision-register-and-open-questions).
- Training or fine-tuning models.
- Multi-cluster, HA, or production-grade security hardening (single-node k3d is the target).
- Replacing the existing `RAG` and `AGENT` Streamlit tabs. They stay as teaching examples.

---

## 2. Where we are today

### Done — Phase 0 (hardening), complete

| Fix | Outcome |
|---|---|
| Calculator sandbox | `eval` replaced with an AST-whitelist evaluator (`safe_eval_expression`). The old `{"__builtins__": {}}` sandbox leaked 4902 classes; all known escapes now rejected. |
| Tenant isolation | Global `doc_manager` replaced by `DocumentStoreRegistry`; one Chroma collection per namespace, bound via `contextvars` in `AgentCore.run` so the LLM cannot select a namespace. |
| Durability | Collections persist to `CHROMA_PERSIST_DIR`; deterministic chunk IDs make re-ingest an upsert; file listing derived from collection metadata. |
| Answer extraction | `_extract_final_answer` scans backwards past tool-call messages; handles list-of-blocks content. |
| File tool scoping | `read_local_file` confined to `FILE_TOOL_ROOTS`. |

### Baseline inventory

| Area | State |
|---|---|
| Runtime | Single Streamlit process, `app.py`, 3 tabs |
| Agent | `create_react_agent` prebuilt; no custom graph |
| Vector store | Chroma, local persisted directory |
| Memory | Redis checkpointer (`RedisSaver`) with `MemorySaver` fallback — works |
| Evaluation | `EvaluationManager` + `OllamaJudge`: heuristics, judge scoring, sliding-window drift, golden-dataset benchmark — **wired into the UI and working** |
| Placeholders | `planning.py`, `action.py`, `observation.py`, `orchestration.py`, `memory/long_term.py`, `engineering/guardrails.py`, `engineering/observability.py` — exported but never instantiated in the runtime path |
| Tests | **None.** No `tests/`, no `conftest.py`, no CI |
| Packaging | No `Dockerfile`, no manifests, no `.github/` |
| Versions | `langgraph` 1.2.10, `langchain-core` 1.5.3 installed. `pyproject` pins `langgraph>=0.2.0` — tighten to `>=1.2` in Phase 4. `fastapi` and `pytest` not yet dependencies. |

> **The asset to build around.** `engineering/evaluation.py` is the rarest thing in this repo.
> Most portfolio agents have no evaluation at all. The plan below treats measured quality as the
> headline deliverable and the LangGraph wiring as the supporting act.

---

## 3. Target architecture

```
                        [ Browser ]
                             │  https://agent.localhost
                             ▼
                   ┌───────────────────┐
                   │ Traefik Ingress   │  (k3d default; SSE buffering disabled)
                   └─────────┬─────────┘
                             │
        ┌────────────────────┴─────────────────────┐
        ▼                                          ▼
┌──────────────────┐                  ┌────────────────────────────┐
│  UI tier         │  HTTP + SSE      │  FastAPI Core Engine       │
│  Streamlit       │ ───────────────► │  • async, Pydantic v2      │
│  • streamed md   │                  │  • SSE token streaming     │
│  • citation rail │                  │  • API-key auth + tenancy  │
│  • approve/deny  │                  │  • error boundary + trace  │
└──────────────────┘                  └─────────────┬──────────────┘
                                                    │
                                    ┌───────────────┴────────────────┐
                                    ▼                                ▼
                        ┌───────────────────────┐        ┌──────────────────────┐
                        │ LangGraph StateGraph  │        │ Ingestion worker     │
                        │ • route / retrieve    │        │ (RQ on Redis)        │
                        │ • grade / rewrite ↺   │        │ • parse, chunk, embed│
                        │ • generate / verify   │        └──────────┬───────────┘
                        │ • HIL interrupt       │                   │
                        └───────┬───────────────┘                   │
                                │                                   │
        ┌───────────────────────┼───────────────────┬───────────────┘
        ▼                       ▼                   ▼
┌───────────────┐     ┌──────────────────┐   ┌──────────────┐   ┌──────────────┐
│ Qdrant        │     │ Reranker         │   │ Redis        │   │ Postgres     │
│ StatefulSet   │     │ BGE cross-enc.   │   │ • checkpoint │   │ • audit log  │
│ • dense+sparse│     │ (sidecar svc)    │   │ • rate limit │   │ • eval runs  │
│ • PVC         │     └──────────────────┘   │ • sem. cache │   └──────────────┘
└───────────────┘                            │ • job queue  │
                                             └──────────────┘
        ┌──────────────────────────────────────────────────────┐
        │ Ollama Service (in-cluster or host-mapped)            │
        └──────────────────────────────────────────────────────┘
        ┌──────────────────────────────────────────────────────┐
        │ Rancher Monitoring: Prometheus + Grafana + log viewer │
        └──────────────────────────────────────────────────────┘
```

### Phase-to-component map

| Phase | Capability | Cluster component |
|---|---|---|
| 1 | Async, FastAPI, Pydantic, JSON validation | `agent-api` Deployment; ClusterIP Service |
| 2 | Streaming, structured output, function calling | `agent-api` → Ollama Service; SSE through Traefik |
| 3 | Chunking, embeddings, vector search, rerank, citations | `qdrant` StatefulSet + PVC; `reranker` Deployment |
| 4 | State graph, conditional edges, tool calling, HIL | StateGraph inside `agent-api`; Redis checkpointer |
| 5 | Redis, logging, rate limiting, exceptions, metrics | `redis` Deployment; `postgres` StatefulSet; ServiceMonitor → Rancher Monitoring |
| 6 | End-to-end enterprise knowledge agent | Full manifest set, Ingress, Secrets/ConfigMaps, dashboards |

---

## 4. Target repository layout

```
local-rag-ollama/
├── agentic_rag/                    # existing package, extended in place
│   ├── api/                        # NEW — Phase 1
│   │   ├── main.py                 # FastAPI app factory, lifespan, middleware
│   │   ├── routes/                 # chat.py, documents.py, threads.py, admin.py
│   │   ├── schemas.py              # Pydantic v2 request/response models
│   │   ├── sse.py                  # SSE event envelope + serializer
│   │   ├── deps.py                 # auth, tenant resolution, rate limit deps
│   │   └── errors.py               # error taxonomy → HTTP mapping
│   ├── llm/                        # NEW — Phase 2
│   │   ├── provider.py             # ChatModel factory, timeouts, retries
│   │   └── structured.py           # structured-output helpers + repair
│   ├── rag/                        # NEW — Phase 3 (supersedes tools.py internals)
│   │   ├── chunking.py  ├── store.py      (Qdrant)
│   │   ├── hybrid.py    ├── rerank.py
│   │   └── citations.py
│   ├── graph/                      # NEW — Phase 4 (replaces core.py internals)
│   │   ├── state.py     ├── nodes/        ├── build.py
│   ├── engineering/                # existing; guardrails + observability get wired
│   ├── memory/  judge/  tools.py  profile.py  config.py
│   └── workers/ingest.py           # NEW — Phase 5
├── ui/streamlit_app.py             # NEW — API client; app.py kept for the legacy tabs
├── tests/                          # NEW — Phase 1 onward
│   ├── conftest.py
│   ├── fakes/                      # fake chat model, fake reranker, fake store
│   ├── unit/  contract/  integration/  e2e/
│   └── data/golden/                # golden dataset + labelled retrieval set
├── deploy/                         # NEW — Phase 6
│   ├── base/                       # namespace, configmap, secret template
│   ├── qdrant/ redis/ postgres/ ollama/ api/ ui/ reranker/
│   ├── ingress.yaml  servicemonitor.yaml
│   └── grafana/agentic-rag-dashboard.json
├── docker/                         # NEW — Phase 6: Dockerfile.api, Dockerfile.ui, Dockerfile.worker
├── scripts/                        # run_ci_eval.py, build_and_import.sh, seed_kb.py
└── .github/workflows/              # ci.yml (Phase 1), eval-gate.yml (Phase 3)
```

**Principle:** new tiers are added beside the existing package, not on top of it. `app.py` keeps
working throughout so there is never a broken commit on `main`.

---

## 5. Cross-cutting contracts

These are settled once, in Phase 1, and every later phase conforms.

### 5.1 Tenancy and scoping

| Concept | Source | Used for |
|---|---|---|
| `tenant_id` | Derived from the API key (never client-supplied) | Qdrant collection, audit rows, rate-limit bucket |
| `thread_id` | Client-supplied, validated `^[A-Za-z0-9_-]{1,64}$` | LangGraph checkpoint, conversation scope |
| `trace_id` | Server-generated ULID per request | Logs, spans, SSE `done` event, audit row |

A client can never select another tenant's data: `tenant_id` comes from the credential, and the
knowledge-base namespace is bound to the execution context (the Phase 0 `contextvars` mechanism),
not passed as a tool argument.

### 5.2 SSE event envelope

One event type per line, `data` is always a JSON object:

| `event` | `data` | Notes |
|---|---|---|
| `token` | `{"delta": "..."}` | Incremental answer text |
| `node` | `{"name": "...", "status": "start\|end", "ms": 42}` | Drives the live graph view |
| `tool_call` | `{"id","name","args"}` | |
| `observation` | `{"id","name","preview"}` | Preview truncated to 2 KB |
| `citation` | `{"n","source","page","chunk_id","score"}` | Emitted before the tokens that cite it |
| `interrupt` | `{"interrupt_id","reason","action","payload"}` | Stream ends; client must call the resume endpoint |
| `done` | `{"answer","citations","usage","latency_ms","trace_id"}` | Terminal |
| `error` | `{"code","message","trace_id","retryable"}` | Terminal |

Heartbeat comment (`: ping`) every 15 s so idle proxies do not drop the connection.

### 5.3 Citation contract

Established in Phase 3, consumed by Phase 4's verifier and the UI.

```
Citation = {
  n: int,                # 1-based marker matching [n] in the answer text
  chunk_id: str,         # stable sha1(source|page|index|content) — already the Phase 0 scheme
  source: str,           # display filename
  page: int | None,
  score: float,          # post-rerank relevance
  span: {start, end} | None   # char offsets into the chunk, when the verifier can localise
}
```

**Rule:** an answer that makes a factual claim without at least one resolvable citation is a
verification failure, not a stylistic preference. Phase 4's `verify_citations` node enforces this.

### 5.4 Error taxonomy

| Code | HTTP | Retryable | Raised when |
|---|---|---|---|
| `INVALID_REQUEST` | 400 | no | Pydantic validation failure |
| `UNAUTHENTICATED` | 401 | no | Missing/unknown API key |
| `RATE_LIMITED` | 429 | yes | Token bucket exhausted; sets `Retry-After` |
| `GUARDRAIL_BLOCKED` | 422 | no | Input or output guardrail refused |
| `RETRIEVAL_UNAVAILABLE` | 503 | yes | Qdrant unreachable after retries |
| `LLM_UNAVAILABLE` | 503 | yes | Ollama unreachable / timed out |
| `LLM_TIMEOUT` | 504 | yes | Generation exceeded the deadline |
| `GRAPH_INTERRUPTED` | 200 | — | Not an error; surfaced as an SSE `interrupt` event |
| `INTERNAL` | 500 | no | Anything unmapped; never leaks a stack trace to the client |

Every response body carries `{"error": {"code","message","trace_id","retryable"}}`.

### 5.5 Degradation ladder

Nothing hard-fails on a single dependency loss.

| Dependency down | Behaviour |
|---|---|
| Redis | Checkpointer falls back to `MemorySaver`; rate limiting fails **open** with a logged warning; semantic cache disabled. Response header `X-Degraded: redis`. |
| Qdrant | Retrieval node returns `RETRIEVAL_UNAVAILABLE`; the router may still answer non-retrieval intents. |
| Reranker | Falls back to fusion order; `X-Degraded: reranker`; eval records the degraded run. |
| Postgres | Audit writes buffer in memory (bounded, 1000 rows) and drop with a counter increment. Never blocks the request. |
| Ollama | `LLM_UNAVAILABLE`; readiness probe fails so the pod leaves the load-balancer rotation. |

### 5.6 Configuration

All settings move to a single Pydantic `Settings` object (`pydantic-settings`), sourced from env.
`.env` for local, ConfigMap for non-secret cluster values, Secret for keys and DSNs. No new
`os.getenv` calls outside `config.py` after Phase 1.

---

## 6. Test strategy

### 6.1 Layers

| Layer | Marker | Dependencies | Runs in | Target |
|---|---|---|---|---|
| Unit | `unit` | None; everything faked | Every commit, < 30 s | Node logic, schemas, chunking, guardrails, error mapping |
| Contract | `contract` | None; ASGI transport only | Every commit | HTTP surface, SSE framing, status codes, OpenAPI stability |
| Integration | `integration` | Real Redis + Qdrant (compose or testcontainers) | Every commit in CI | Checkpoint resume, vector round-trip, rate limiter, worker |
| Eval | `eval` | Real LLM (Ollama) | Nightly + pre-release | Golden dataset, judge scores, drift |
| E2E | `e2e` | Deployed cluster | Pre-release, manual | Ingress → UI → API → answer with citations |

`pytest.ini` registers all markers; default run is `-m "unit or contract"` so the inner loop
stays fast. CI adds `integration`. `eval` and `e2e` are opt-in jobs.

### 6.2 Fakes — built once in Phase 1, reused everywhere

| Fake | Replaces | Why |
|---|---|---|
| `FakeToolCallingModel` | `ChatOllama` | Scripted list of `AIMessage`s (with `tool_calls`) so graph paths are deterministic. `langchain_core`'s `GenericFakeChatModel` does not script tool calls, so this is written locally. |
| `FakeEmbeddings` | `HuggingFaceEmbeddings` | Deterministic hash-based vectors; keeps unit tests off the 90 MB model |
| `FakeVectorStore` | Qdrant | In-memory, returns a fixed ranked list |
| `FakeReranker` | BGE service | Identity or scripted reordering |
| `fakeredis` | Redis | Rate limiter and cache unit tests |
| `frozen clock` | `time.monotonic` | Deterministic token-bucket and latency assertions |

### 6.3 Determinism rules

1. No unit or contract test may call a real network endpoint. Enforced by an autouse fixture that
   patches `socket.socket` to raise, with an opt-out marker.
2. LLM-dependent assertions use fakes. Real-LLM behaviour is asserted **statistically** in the
   `eval` layer against thresholds, never as exact string equality.
3. Every integration test creates a uniquely-named collection/keyspace and tears it down.

### 6.4 Quality gates

| Gate | Threshold | Enforced from |
|---|---|---|
| Line coverage, `agentic_rag/` | ≥ 80 % | Phase 1 |
| Branch coverage, `agentic_rag/graph/` | ≥ 90 % | Phase 4 |
| New modules with zero tests | 0 | Phase 1 |
| Retrieval hit@5 on the labelled set | ≥ 0.80, no drop vs `main` | Phase 3 |
| Judge faithfulness mean on golden set | ≥ 0.85, no drop > 0.03 vs `main` | Phase 3 |
| Citation coverage on golden set | ≥ 0.95 | Phase 3 |
| p95 latency, cached retrieval path | ≤ 3 s | Phase 5 |

### 6.5 Retrofit obligation

`tests/unit/test_phase0_hardening.py` lands with Phase 1, porting the existing scratchpad
verification script (sandbox escapes, tenant isolation, restart persistence, answer extraction).
Phase 0 is the only phase whose tests are written retroactively.

---

## Phase 1 — FastAPI core engine

**Capability:** async, Pydantic schemas, JSON validation, SSE plumbing.
**Cluster component:** `agent-api` Deployment + ClusterIP Service.

### Deliverables

1. `agentic_rag/api/` — app factory, lifespan (warm the embedding model, open Redis, close cleanly),
   routers, Pydantic v2 schemas, error boundary middleware, `trace_id` middleware.
2. Endpoints:
   - `POST /api/v1/chat` — non-streaming, returns the full `ChatResponse`
   - `POST /api/v1/chat/stream` — SSE, envelope per [§5.2](#52-sse-event-envelope)
   - `POST /api/v1/documents` — upload, returns `202` + `job_id` (synchronous until Phase 5)
   - `GET /api/v1/documents` · `DELETE /api/v1/documents`
   - `GET /healthz` (liveness, no deps) · `GET /readyz` (checks Redis + store)
3. API-key auth dependency; `tenant_id` resolved from the key.
4. `tests/` scaffolding: `conftest.py`, `fakes/`, markers, coverage config.
5. `pytest`, `pytest-asyncio`, `pytest-cov`, `httpx`, `fastapi`, `uvicorn`, `pydantic-settings`
   added to `pyproject`; dev deps in a `dev` group.
6. `.github/workflows/ci.yml` — lint, unit, contract, integration; coverage gate.

### Design decisions

- **Thin routes.** Routes validate, resolve tenancy, and delegate to `AgenticRAGHelper`. No
  business logic in the API layer — this keeps Phase 4's graph swap from touching HTTP code.
- **Sync core behind async routes.** The existing helper is synchronous and the embedding model
  releases the GIL poorly. Route handlers offload to a thread via `anyio.to_thread.run_sync`
  until Phase 4 makes the graph natively async. Contextvars propagate correctly across
  `to_thread`, which the Phase 0 namespace binding depends on.
- **SSE, not WebSockets.** One-way token streaming, works through Traefik with no upgrade
  handshake, trivially curl-able in a demo.
- **OpenAPI is a contract.** The generated schema is snapshotted; a diff fails the build unless
  the snapshot is intentionally updated.

### Definition of Done

- [ ] `uvicorn agentic_rag.api.main:app` serves all endpoints; `/docs` renders
- [ ] `curl -N .../chat/stream` prints tokens incrementally, not one block at the end
- [ ] Malformed body → `400` with the `INVALID_REQUEST` envelope; no stack trace in the response
- [ ] Missing/bad API key → `401`; a valid key for tenant A cannot read tenant B's documents
- [ ] Unhandled exception → `500` with a `trace_id` that appears in the logs
- [ ] `/readyz` returns `503` when Redis is stopped, `200` when it returns
- [ ] Streamlit tab 3 still works unchanged (no regression on `app.py`)
- [ ] CI green; coverage ≥ 80 %; OpenAPI snapshot committed

### Planned pytest

```
tests/conftest.py                      # app fixture, ASGI client, fake settings, no-network autouse
tests/fakes/{chat_model,embeddings,store}.py

tests/unit/test_schemas.py
    test_chat_request_rejects_blank_query
    test_thread_id_pattern_rejects_path_traversal          # "../../etc"
    test_chat_response_serialises_citations_stably
tests/unit/test_errors.py
    test_each_taxonomy_code_maps_to_expected_status         # parametrised over the table
    test_internal_error_never_leaks_traceback
tests/unit/test_sse.py
    test_event_serialisation_is_valid_sse_framing          # "event: x\ndata: {...}\n\n"
    test_multiline_payload_is_escaped
tests/unit/test_phase0_hardening.py                        # retrofit; see §6.5
    test_sandbox_rejects_dunder_escape[8 payloads]
    test_namespaces_are_isolated
    test_reingest_is_idempotent
    test_final_answer_skips_pre_tool_reasoning

tests/contract/test_chat_endpoint.py
    test_chat_returns_answer_and_trace_id
    test_chat_is_401_without_api_key
    test_tenant_cannot_read_other_tenant_documents
tests/contract/test_stream_endpoint.py
    test_stream_emits_tokens_then_done
    test_stream_emits_error_event_when_llm_unavailable
    test_stream_sends_heartbeat_when_idle
tests/contract/test_openapi_snapshot.py
    test_openapi_schema_matches_snapshot
tests/contract/test_health.py
    test_healthz_is_dependency_free
    test_readyz_reports_503_when_redis_down

tests/integration/test_lifespan.py
    test_embedding_model_loaded_once_across_requests
    test_redis_reconnect_after_bounce
```

### Risks

| Risk | Mitigation |
|---|---|
| Blocking call in an async route stalls the event loop | `to_thread` offload + a test asserting concurrent requests overlap |
| Contextvar namespace lost across the thread hop | Explicit integration test asserting tenant isolation through the HTTP layer |

---

## Phase 2 — LLM layer: streaming, structured output, function calling

**Capability:** token streaming, structured output, function/tool calling.
**Cluster component:** `agent-api` → `ollama` Service (in-cluster) or host-mapped endpoint.

### Deliverables

1. `agentic_rag/llm/provider.py` — a `get_chat_model(purpose)` factory with per-purpose config
   (`chat`, `judge`, `router`, `grader`), explicit timeouts, bounded retry with jitter, and a
   `LLM_UNAVAILABLE` / `LLM_TIMEOUT` mapping.
2. True token streaming from `graph.astream_events` through the SSE `token` event.
3. `agentic_rag/llm/structured.py` — `with_structured_output` wrappers for the Pydantic models
   later used by the router, document grader, and citation verifier, plus one repair retry on
   schema-validation failure.
4. Tool-calling verification against the configured model, with a documented fallback if the
   model's tool support is unreliable.
5. Token/latency accounting surfaced in the SSE `done` event.

### Design decisions

- **Provider-agnostic by interface, Ollama by default.** `provider.py` returns a LangChain
  `BaseChatModel`; swapping to a hosted provider is a config change. No provider SDK is imported
  outside this module.
- **Structured output is validated twice.** Once by the model adapter, once by an explicit
  `model_validate`. A malformed structure is a typed failure the graph can branch on, not an
  exception that kills the turn.
- **Purpose-scoped models.** The router and grader want a small, fast, low-temperature model; the
  generator wants the strong one. Separating them now is what makes the Phase 5 cost numbers
  interesting.
- **Model capability probe.** Startup logs whether the configured model actually honours tool
  calls and structured output, so a silent capability downgrade is visible in the pod logs.

### Definition of Done

- [ ] First token reaches the browser in < 1.5 s on a warm path (measured, recorded)
- [ ] A structured-output call returns a validated Pydantic object; a deliberately corrupted
      response triggers exactly one repair attempt, then a typed failure
- [ ] A tool call round-trips: model → `tool_call` event → observation → final answer
- [ ] Ollama stopped mid-stream → SSE `error` event with `LLM_UNAVAILABLE`, connection closed
      cleanly, no hung request
- [ ] Timeouts are enforced and configurable; no unbounded LLM call exists in the codebase
- [ ] `done` event carries prompt/completion token counts and latency

### Planned pytest

```
tests/unit/test_provider.py
    test_purpose_returns_configured_model_and_temperature
    test_retry_stops_after_max_attempts
    test_connection_error_maps_to_llm_unavailable
    test_timeout_maps_to_llm_timeout
tests/unit/test_structured_output.py
    test_valid_payload_parses_to_model
    test_malformed_payload_triggers_single_repair_then_typed_failure
    test_repair_is_not_attempted_twice
tests/unit/test_streaming_adapter.py
    test_astream_events_map_to_token_events
    test_tool_call_chunk_maps_to_tool_call_event
    test_stream_cancellation_closes_upstream          # client disconnect
tests/contract/test_stream_token_order.py
    test_tokens_arrive_before_done
    test_citation_events_precede_the_tokens_that_cite_them

tests/eval/test_model_capabilities.py      # marker: eval, needs real Ollama
    test_model_emits_wellformed_tool_calls
    test_model_honours_structured_output_schema
```

### Risks

| Risk | Mitigation |
|---|---|
| The configured model handles tool calls unreliably | Capability probe at startup + an `eval`-marked test; documented fallback to prompt-based JSON extraction |
| Client disconnect leaks an upstream generation | Cancellation test; `anyio` cancel scope around the stream |

---

## Phase 3 — RAG core: Qdrant, rerank, citations

**Capability:** chunking, embeddings, vector search, reranking, citations.
**Cluster component:** `qdrant` StatefulSet + PVC; `reranker` Deployment.

This is the phase that produces the numbers the whole demo rests on.

### Deliverables

1. `rag/chunking.py` — structure-aware splitting (headings, paragraph and table boundaries,
   page-span metadata) replacing blind 1024/100.
2. `rag/store.py` — Qdrant backend behind the existing `DocumentRetrieverManager` interface;
   one collection per tenant; named vectors for dense + sparse. Chroma retained behind a config
   flag for local dev.
3. `rag/hybrid.py` — BM25/sparse + dense retrieval fused with Reciprocal Rank Fusion.
4. `rag/rerank.py` — BGE cross-encoder over the top-30 → top-5, called as an in-cluster HTTP
   service with a local in-process fallback.
5. `rag/citations.py` — the [§5.3](#53-citation-contract) contract: chunk-ID propagation,
   marker assignment, answer-to-source resolution.
6. `tests/data/golden/` — **50–100 labelled Q/A pairs** with relevant chunk IDs, hard negatives,
   unanswerable questions, and prompt-injection documents.
7. `scripts/run_ci_eval.py` + `.github/workflows/eval-gate.yml` — the pipeline already specified
   in `IMPLEMENTATION.md` §6, actually built.
8. A recorded ablation table.

### The ablation table (the deliverable that matters)

| Configuration | hit@5 | MRR | Faithfulness | Citation coverage | p95 latency |
|---|---|---|---|---|---|
| Baseline: fixed chunks, dense-only, k=3 | — | — | — | — | — |
| + structure-aware chunking | | | | | |
| + hybrid (BM25 + dense, RRF) | | | | | |
| + cross-encoder rerank | | | | | |
| + citation verification | | | | | |

Filled in by `scripts/run_ci_eval.py`, committed as `docs/ablation.md`, and rendered in the UI's
Ops tab in Phase 5. **Every row must be reproducible by one command.**

### Design decisions

- **Qdrant over pgvector.** Native sparse-vector and named-vector support makes hybrid retrieval
  a single query rather than an application-side join. Postgres still arrives in Phase 5 for the
  audit trail, so the demo shows both stores used for what they are good at.
- **Reranker as a separate Deployment.** A cross-encoder is a different resource profile from the
  API (CPU-heavy, bursty). Separating it makes the Rancher pod-metrics view genuinely
  interesting — you can watch one pod spike while the API stays flat.
- **Chunk IDs stay `sha1(source|page|index|content)`.** Already implemented in Phase 0, already
  idempotent, and it is exactly the key the citation contract needs. No change.
- **Golden dataset is version-controlled and reviewed.** It is the closest thing this project has
  to a spec. Adding a question requires the same review as adding code.
- **Migration is additive.** Qdrant lands behind `VECTOR_BACKEND=qdrant|chroma` with a
  `scripts/migrate_chroma_to_qdrant.py`. Both backends pass the same test suite, parametrised.

### Definition of Done

- [ ] Qdrant and Chroma both pass the shared `VectorStore` contract test suite
- [ ] Tenant A's query never returns tenant B's chunks — asserted at the store layer *and* through HTTP
- [ ] Hybrid + rerank beats dense-only on hit@5 by a **measured** margin, recorded in `docs/ablation.md`
- [ ] Every answer on the golden set carries ≥ 1 resolvable citation; coverage ≥ 0.95
- [ ] Clicking a citation in the UI opens the source and page
- [ ] `eval-gate.yml` runs on PR, publishes `baseline_metrics.json`, and **demonstrably fails**
      a PR that degrades retrieval (prove it with a throwaway PR; keep the link)
- [ ] Reranker pod deleted → answers still return, `X-Degraded: reranker` set, eval records it
- [ ] Ingesting a 300-page PDF does not time out the request (sync is acceptable here; Phase 5 queues it)

### Planned pytest

```
tests/unit/test_chunking.py
    test_splits_on_heading_boundaries
    test_preserves_page_span_metadata
    test_table_rows_are_not_split_mid_row
    test_chunk_id_is_stable_across_runs
    test_chunk_id_changes_when_content_changes
tests/unit/test_hybrid_fusion.py
    test_rrf_ranks_doc_present_in_both_lists_highest
    test_fusion_is_stable_for_tied_scores
    test_empty_sparse_result_degrades_to_dense_order
tests/unit/test_rerank.py
    test_reranker_reorders_by_score
    test_reranker_failure_falls_back_to_fusion_order
    test_topk_truncation_applied_after_rerank
tests/unit/test_citations.py
    test_markers_are_assigned_in_first_use_order
    test_answer_without_citation_is_flagged
    test_citation_resolves_to_source_and_page
    test_hallucinated_chunk_id_is_rejected

tests/contract/test_vector_store_contract.py        # parametrised: [chroma, qdrant]
    test_upsert_then_search_roundtrip
    test_reingest_does_not_duplicate
    test_delete_collection_empties_namespace
    test_metadata_survives_roundtrip
    test_tenant_isolation

tests/integration/test_qdrant.py                    # real Qdrant
    test_hybrid_query_returns_named_vector_results
    test_collection_survives_restart
    test_concurrent_ingest_is_consistent
tests/integration/test_ingest_large_pdf.py
    test_300_page_pdf_completes_within_budget

tests/eval/test_golden_dataset.py                   # marker: eval
    test_hit_at_5_meets_threshold
    test_faithfulness_mean_meets_threshold
    test_citation_coverage_meets_threshold
    test_unanswerable_questions_are_refused_not_fabricated
    test_injection_documents_do_not_alter_agent_behaviour
```

### Risks

| Risk | Mitigation |
|---|---|
| Golden dataset too small to be meaningful | Minimum 50 pairs is a DoD item, not a suggestion; report confidence intervals |
| Reranker latency dominates p95 | Measured in the ablation table; cap candidates at 30; degradation path tested |
| Qdrant migration silently loses metadata | Shared contract suite runs against both backends |

---

## Phase 4 — LangGraph StateGraph and human-in-the-loop

**Capability:** explicit state graph, conditional edges, tool calling, HIL approval.
**Cluster component:** StateGraph inside `agent-api`, checkpointed to Redis.

### Deliverables

1. `graph/state.py` — a typed state schema (messages, plan, query, candidates, citations,
   grade, retry counters, pending approval, degradation flags).
2. `graph/nodes/` — one module per node, each independently unit-testable with no graph involved.
3. `graph/build.py` — assembles and compiles the graph with the Redis checkpointer.
4. HIL via `interrupt()` + `Command(resume=...)`, plus `POST /api/v1/threads/{id}/resume`.
5. `planning.py` and `observation.py` implemented for real; `orchestration.py` and
   `memory/long_term.py` either implemented or **deleted**. No more exported placeholders.
6. Live node-status rendering in the UI, driven by the SSE `node` event.

### Graph topology

```
        ┌──────────┐
        │ guard_in │──blocked──► GUARDRAIL_BLOCKED
        └────┬─────┘
             ▼
        ┌──────────┐   direct    ┌──────────┐
        │  route   │────────────►│ generate │
        └────┬─────┘             └────┬─────┘
     retrieve│         tool           │
             ▼          └──►┌──────────────┐
       ┌──────────┐         │  act (tools) │──► observe ──┐
       │ retrieve │         └──────────────┘              │
       └────┬─────┘                ▲                      │
            ▼                      │ needs_approval       │
       ┌──────────┐                │                      │
       │  rerank  │           ┌────┴─────┐                │
       └────┬─────┘           │ hil_wait │◄── interrupt() │
            ▼                 └──────────┘                │
       ┌──────────┐  insufficient   ┌──────────┐          │
       │  grade   │────────────────►│ rewrite  │──↺ (max 2)
       └────┬─────┘                 └──────────┘          │
   sufficient│                                            │
            ▼                                             │
       ┌──────────┐                                       │
       │ generate │◄──────────────────────────────────────┘
       └────┬─────┘
            ▼
    ┌────────────────┐  unsupported   ┌──────────┐
    │verify_citations│───────────────►│ rewrite  │──↺ (max 1)
    └────┬───────────┘                └──────────┘
         ▼
    ┌──────────┐
    │ guard_out│──► answer
    └──────────┘
```

### Design decisions

- **Every loop has a hard cap.** `rewrite` from `grade` is capped at 2, from `verify_citations`
  at 1. On exhaustion the agent returns a partial answer flagged `low_confidence` — it never
  spins, and it never pretends. This is asserted by test, not by convention.
- **`interrupt()` over `interrupt_before`.** Approval is data-dependent (only side-effecting or
  low-confidence actions need it), which static `interrupt_before` cannot express. Requires
  `langgraph>=1.2` — already installed at 1.2.10; the `pyproject` pin is tightened here.
- **Nodes are pure functions of state.** `def node(state) -> dict` returning a partial update.
  No node reaches for a global. This is what makes the 90 % branch-coverage gate achievable.
- **The router is a real decision, and it is measured.** Direct-answer vs retrieve vs tool is
  logged per turn and charted in Phase 5, so "did routing help?" is answerable with data.
- **Resume is idempotent.** Resuming an already-resumed interrupt returns the existing result
  rather than re-executing the action — a double-clicked approve button must not double-fire.

### Definition of Done

- [ ] `create_react_agent` is gone from the request path; `build_graph()` compiles the explicit graph
- [ ] Every conditional edge has a test that forces it
- [ ] Bad retrieval triggers exactly one rewrite loop and recovers; a permanently-bad query
      terminates at the cap with a `low_confidence` answer
- [ ] A side-effecting tool halts at `hil_wait`; the SSE stream emits `interrupt` and closes;
      `POST /resume` with `approve` completes the action, `reject` produces a refusal answer
- [ ] Interrupted state survives an API pod restart (checkpointed in Redis) and resumes correctly
- [ ] Double-resume does not double-execute
- [ ] Zero placeholder modules remain exported from `agentic_rag/__init__.py`
- [ ] `IMPLEMENTATION.md` diagram matches the compiled graph — verified by a test that renders
      the real topology and compares node/edge names
- [ ] Branch coverage on `graph/` ≥ 90 %

### Planned pytest

```
tests/unit/nodes/test_route.py
    test_smalltalk_routes_direct
    test_document_question_routes_retrieve
    test_calculation_routes_tool
    test_ambiguous_intent_defaults_to_retrieve
tests/unit/nodes/test_grade.py
    test_relevant_context_marks_sufficient
    test_offtopic_context_marks_insufficient
    test_empty_candidates_marks_insufficient
tests/unit/nodes/test_rewrite.py
    test_rewrite_increments_attempt_counter
    test_rewrite_refuses_past_cap
tests/unit/nodes/test_verify_citations.py
    test_claim_without_citation_is_unsupported
    test_citation_to_unretrieved_chunk_is_rejected
tests/unit/nodes/test_hil.py
    test_side_effecting_tool_requires_approval
    test_readonly_tool_does_not_require_approval

tests/unit/test_graph_topology.py
    test_all_nodes_reachable_from_start
    test_no_node_lacks_an_outbound_edge
    test_compiled_topology_matches_documented_diagram

tests/unit/test_graph_paths.py                 # FakeToolCallingModel scripts each path
    test_direct_answer_path_skips_retrieval
    test_happy_retrieval_path_emits_citations
    test_bad_then_good_retrieval_recovers_after_one_rewrite
    test_persistently_bad_retrieval_stops_at_cap_with_low_confidence
    test_guardrail_block_short_circuits_before_llm_call
    test_reranker_outage_still_produces_an_answer

tests/integration/test_hil_resume.py           # real Redis
    test_interrupt_persists_across_new_graph_instance
    test_approve_completes_the_pending_action
    test_reject_produces_refusal_without_side_effect
    test_double_resume_is_idempotent
    test_resume_unknown_interrupt_id_returns_404
tests/integration/test_multiturn_memory.py
    test_second_turn_sees_first_turn_context
    test_threads_do_not_leak_into_each_other
```

### Risks

| Risk | Mitigation |
|---|---|
| Graph complexity outruns test coverage | 90 % branch gate; one test file per node; topology test |
| Interrupt/resume state corruption | Idempotency test; unique `interrupt_id`; integration test across instances |
| Diagram drifts from the code again | Automated topology-vs-docs test — the specific failure mode this repo already had |

---

## Phase 5 — Enterprise infrastructure

**Capability:** Redis caching and rate limiting, structured logging, metrics, exception boundaries.
**Cluster component:** `redis` Deployment, `postgres` StatefulSet, ServiceMonitor → Rancher Monitoring.

### Deliverables

1. **Rate limiting** — Redis token bucket per tenant and per IP; `429` + `Retry-After`; fails open.
2. **Semantic cache** — embedding-similarity lookup on `(tenant, normalised query)` above a
   threshold; TTL'd; reports hit rate and tokens saved.
3. **Async ingestion** — RQ worker on Redis. `POST /documents` → `202` + `job_id`;
   `GET /documents/jobs/{id}` for progress. Large PDFs stop blocking the API.
4. **Structured logging** — JSON lines, `trace_id` on every record, secrets redacted.
5. **Observability wired for real** — `ObservabilityManager` and `GuardrailManager` finally
   instantiated in the request path. OpenTelemetry spans per node; Prometheus `/metrics`.
6. **Audit trail** — Postgres: turn, tenant, thread, query, chosen route, cited chunk IDs, judge
   scores, tokens, cost, latency, approval decisions.
7. **Ops tab in the UI** — score distributions, drift chart, latency percentiles, cost per query,
   cache hit rate, guardrail trips, route mix.
8. **Load test** — Locust profile and a recorded concurrency number.

### Metrics catalogue

| Metric | Type | Labels |
|---|---|---|
| `agent_turn_duration_seconds` | histogram | `route`, `outcome` |
| `agent_node_duration_seconds` | histogram | `node` |
| `agent_llm_tokens_total` | counter | `purpose`, `direction` |
| `agent_llm_cost_usd_total` | counter | `purpose` |
| `agent_retrieval_hits` | histogram | `backend`, `reranked` |
| `agent_cache_lookups_total` | counter | `result` (hit/miss) |
| `agent_guardrail_trips_total` | counter | `stage`, `rule` |
| `agent_rate_limit_rejections_total` | counter | `tenant` |
| `agent_hil_pending` | gauge | — |
| `agent_judge_score` | histogram | `metric` |
| `agent_degraded_total` | counter | `dependency` |

### Design decisions

- **Fail open on rate limiting, fail closed on auth.** Availability matters more than perfect
  quota enforcement in a demo; identity does not. Both directions are tested explicitly.
- **Semantic cache keyed per tenant.** Cross-tenant cache hits would be a data leak. This is a
  test case, not a note.
- **Audit in Postgres, not logs.** Logs rotate; an audit trail must be queryable. It also gives
  the demo a second, correctly-chosen datastore.
- **Guardrails become real.** Input: length, injection heuristics, known override phrases.
  Output: PII redaction, refusal-consistency check. Both stages get counters, so "how often does
  it trip?" is answerable.
- **Cost is tracked even for a local model.** Priced at a configured rate per 1K tokens. The
  point is demonstrating cost-awareness, and it makes the Phase 4 router decision measurable.

### Definition of Done

- [ ] Exceeding the limit returns `429` with `Retry-After`; Redis stopped → requests still served,
      `X-Degraded: redis` set, warning logged
- [ ] Repeated identical query is served from cache with a measured latency drop; **cache never
      crosses tenants**
- [ ] A 300-page PDF upload returns `202` in < 500 ms; the job completes in the background and
      progress is pollable
- [ ] Every log line is valid JSON with a `trace_id`; grepping one `trace_id` reconstructs the
      whole turn end-to-end
- [ ] No API key, DSN, or document content appears in any log line (asserted by test)
- [ ] `/metrics` exposes the full catalogue; a Grafana dashboard renders latency, cost, and drift
- [ ] The audit table contains one complete row per turn, including cited chunk IDs and approvals
- [ ] Locust: sustained concurrency target met with p95 ≤ 3 s on the cached path; number recorded
- [ ] Every dependency in the [§5.5](#55-degradation-ladder) ladder has a test that kills it and
      asserts the documented behaviour

### Planned pytest

```
tests/unit/test_rate_limit.py                  # fakeredis + frozen clock
    test_bucket_allows_burst_then_throttles
    test_bucket_refills_over_time
    test_limits_are_per_tenant_not_global
    test_redis_outage_fails_open_and_warns
tests/unit/test_semantic_cache.py
    test_near_identical_query_hits
    test_dissimilar_query_misses
    test_cache_is_tenant_scoped                # data-leak regression
    test_ttl_expiry_evicts
tests/unit/test_logging.py
    test_every_record_is_json_with_trace_id
    test_api_key_is_redacted
    test_document_content_is_not_logged
tests/unit/test_guardrails.py
    test_overlong_input_blocked
    test_injection_phrases_flagged[parametrised corpus]
    test_pii_redacted_from_output
    test_guardrail_trip_increments_counter
tests/unit/test_metrics.py
    test_turn_duration_recorded_with_route_label
    test_degradation_counter_increments_per_dependency

tests/integration/test_ingest_worker.py        # real Redis + RQ
    test_upload_returns_202_immediately
    test_job_completes_and_documents_become_searchable
    test_failed_job_surfaces_error_status
    test_worker_crash_does_not_lose_the_job
tests/integration/test_audit_trail.py          # real Postgres
    test_one_row_per_turn_with_citations
    test_approval_decision_recorded
    test_postgres_outage_does_not_fail_the_request
tests/integration/test_degradation_matrix.py   # parametrised over the ladder
    test_dependency_outage_matches_documented_behaviour[redis|qdrant|reranker|postgres]

tests/e2e/locustfile.py                        # not pytest; recorded in docs/load-test.md
```

### Risks

| Risk | Mitigation |
|---|---|
| Semantic cache returns a stale or wrong-tenant answer | Tenant-scoped key is a test; similarity threshold tuned on the golden set |
| Observability overhead inflates latency | Measure with tracing on and off; record both in the load test |
| Postgres becomes a hard dependency by accident | Explicit outage test asserting the request still succeeds |

---

## Phase 6 — Complete system on Rancher

**Capability:** the whole thing, deployed, observed, and demonstrable.
**Cluster component:** full manifest set managed and observed through the Rancher UI.

### Deliverables

1. `docker/Dockerfile.api`, `.ui`, `.worker` — multi-stage, non-root, pinned base images,
   `HEALTHCHECK`.
2. `deploy/` manifests: Deployments, StatefulSets, Services, PVCs, ConfigMap, Secret template,
   Ingress (`agent.localhost`), ServiceMonitor, resource requests/limits, liveness/readiness/
   startup probes.
3. `scripts/build_and_import.sh` — build + `k3d image import <tag> -c rancher-cluster`.
4. Rancher Monitoring enabled; the Grafana dashboard from Phase 5 imported.
5. `docs/RUNBOOK.md` — deploy, upgrade, roll back, read the dashboards, common failures.
6. `docs/DEMO.md` — the [§14](#14-demo-script) walkthrough.
7. E2E smoke suite runnable against the deployed ingress.

### Design decisions

- **Plain manifests with kustomize overlays, not Helm.** Fewer layers between the reader and the
  Kubernetes objects — better for a demo whose point is showing you understand them. A Helm chart
  can wrap this later.
- **`imagePullPolicy: IfNotPresent` + `k3d image import`.** No registry needed for local work.
  Images are tagged with the git SHA, never `latest`, so rollbacks are real.
- **Startup probe separate from liveness.** The embedding model takes tens of seconds to load;
  without a startup probe, k8s kills the pod in a restart loop. This is the single most likely
  first-deploy failure and the manifests are written to pre-empt it.
- **Secrets are templates.** `deploy/base/secret.example.yaml` is committed; the real Secret is
  created out-of-band. No credential ever enters git.
- **SSE through Traefik needs care.** Response buffering must be off and read timeouts raised, or
  streaming silently degrades to one big chunk at the end. Explicit ingress annotations, and an
  E2E test that asserts incremental arrival through the ingress — not just against the pod.

### Definition of Done

- [ ] `kubectl apply -k deploy/overlays/local` brings up every pod to `Ready` from a clean cluster
- [ ] `https://agent.localhost` serves the UI; a question returns a cited, streamed answer
- [ ] Streaming is still incremental **through the ingress**, verified by E2E test
- [ ] Every pod has requests, limits, and all three probes; none is in a restart loop after 10 min
- [ ] Qdrant and Postgres data survive `kubectl delete pod` (PVC-backed)
- [ ] Config comes from ConfigMap; secrets from Secret; no credentials in any manifest or image
- [ ] Rancher UI shows pod metrics, logs, and events for the whole stack
- [ ] Grafana dashboard renders live latency, cost, drift, and cache hit rate
- [ ] Rolling update with zero failed requests under light Locust load
- [ ] `docs/RUNBOOK.md` is accurate enough that a rollback can be performed by following it
- [ ] `README.md` front page: architecture diagram, ablation table, metrics screenshot, and a
      link to the eval-gate PR that failed then passed

### Planned pytest

```
tests/e2e/conftest.py                          # BASE_URL from env; skip if unreachable
tests/e2e/test_smoke.py                        # marker: e2e
    test_healthz_reachable_through_ingress
    test_readyz_reports_all_dependencies
    test_ask_returns_answer_with_resolvable_citations
    test_stream_arrives_incrementally_through_ingress   # asserts inter-token gaps
tests/e2e/test_hil_flow.py
    test_approval_required_action_pauses_and_resumes
tests/e2e/test_resilience.py
    test_api_pod_restart_preserves_conversation
    test_qdrant_pod_restart_preserves_documents
    test_rolling_update_serves_every_request
tests/e2e/test_isolation.py
    test_two_tenants_cannot_see_each_others_documents

tests/integration/test_manifests.py            # static validation, no cluster needed
    test_every_container_sets_resource_limits
    test_every_deployment_defines_all_three_probes
    test_no_manifest_contains_a_literal_secret
    test_images_are_sha_tagged_not_latest
```

### Risks

| Risk | Mitigation |
|---|---|
| Embedding model load exceeds the probe window | Startup probe with a generous failure threshold; bake the model into the image |
| Traefik buffers SSE and streaming appears broken | Ingress annotations + an E2E test asserting incremental arrival |
| Image size makes the loop slow | Multi-stage build; model in a separate cached layer; measure and record build time |
| k3d resource exhaustion on one node | Modest requests; Ollama may stay host-mapped rather than in-cluster |

---

## 13. Deployment topology

| Workload | Kind | Replicas | Storage | Notes |
|---|---|---|---|---|
| `agent-api` | Deployment | 2 | — | Stateless; state lives in Redis. Rolling update. |
| `agent-ui` | Deployment | 1 | — | Streamlit client of the API |
| `agent-worker` | Deployment | 1 | — | RQ ingestion consumer |
| `reranker` | Deployment | 1 | — | CPU-heavy, separate profile |
| `qdrant` | StatefulSet | 1 | PVC 10 Gi | Vector index |
| `redis` | Deployment | 1 | emptyDir | Checkpoints, rate limit, cache, queue. Loss = degraded, not fatal. |
| `postgres` | StatefulSet | 1 | PVC 5 Gi | Audit trail, eval history |
| `ollama` | Deployment or host-mapped | 1 | PVC (models) | See [§15 D-6](#15-decision-register-and-open-questions) |

Namespace `agentic-rag`. Ingress `agent.localhost` → `agent-ui`; `agent.localhost/api` → `agent-api`.
Rancher Monitoring in `cattle-monitoring-system` scrapes via ServiceMonitor.

---

## 14. Demo script

Five minutes, in this order. Each beat maps to a phase.

1. **Rancher UI** — the workload list, all green. "This is a deployment, not a notebook." (P6)
2. **Ask a question** — tokens stream in, node badges light up in sequence, citations appear in
   the sidebar. Click one; it opens the source page. (P2, P3, P4)
3. **Ask a question the documents cannot answer** — the agent refuses and says why, instead of
   fabricating. Show the `verify_citations` node firing in the trace. (P4)
4. **Trigger an approval** — the stream pauses on an `interrupt`; the approve/reject buttons
   appear. Delete the API pod. Approve anyway; it resumes from the Redis checkpoint. (P4, P5)
5. **Ops tab** — judge scores, drift chart, p95 latency, cost per query, cache hit rate, route
   mix. (P5)
6. **The ablation table and the eval-gate PR** — "here is what each retrieval change was worth,
   and here is CI blocking a regression." (P3)
7. **Grafana in Rancher** — the same numbers, live, at the infrastructure layer. (P6)

Beats 3, 5, and 6 are the ones that separate this from every other LangGraph portfolio project.
Beat 6 is the strongest single artefact in the whole build.

---

## 15. Decision register and open questions

### Decisions

| # | Decision | Rationale |
|---|---|---|
| D-1 | Qdrant as the primary vector store | Native hybrid/sparse support; pgvector would need app-side fusion |
| D-2 | Keep Chroma behind a config flag | Zero-dependency local dev; gives the store contract suite two implementations |
| D-3 | SSE over WebSockets | One-way streaming, ingress-friendly, curl-demonstrable |
| D-4 | Plain manifests + kustomize, not Helm | Legibility for a demo about understanding Kubernetes |
| D-5 | Postgres for audit, Redis for ephemeral state | Correct tool per job; also demonstrates the distinction |
| D-6 | Ollama placement deferred | In-cluster is cleaner architecturally but may exhaust a single k3d node. Decide by measurement in Phase 6. |
| D-7 | Multi-agent orchestration deferred, possibly permanently | "Built a multi-agent system with LangGraph" is the most common line on every competing résumé. A single well-instrumented agent with published numbers is the rarer and stronger claim. If it is built, it ships **with** a single-vs-supervisor comparison on the same eval set including cost — otherwise `orchestration.py` gets deleted rather than left as a placeholder. |
| D-8 | Every phase lands behind a flag with `app.py` still working | No broken `main`; each phase is independently demonstrable |

### Open questions for review

1. **Vertical scenario.** The plan assumes a policy/compliance knowledge agent — must cite, must
   refuse when unsupported, escalates to human. It forces citations, branching, HIL, and audit
   naturally. Confirm, or name the domain you would rather demo. This choice shapes the golden
   dataset, so it should be settled before Phase 3.
2. **Which tool needs approval?** HIL needs at least one genuinely side-effecting tool to be
   convincing. Candidates: send-summary-email, write-to-ticket, export-report. A stubbed
   integration is fine, but it must have a real side effect to approve.
3. **Golden dataset source.** Who writes the 50–100 labelled Q/A pairs, and against which corpus?
   This is the critical path for Phase 3 and the most easily underestimated item in the plan.
4. **Is Phase 2 partly redundant?** Streaming and structured output could fold into Phases 1 and 4.
   Kept separate here because it maps to the roadmap and gives the provider abstraction its own
   DoD — say if you would rather compress it.
5. **Cost model.** For a local Ollama model, is a synthetic price per 1K tokens acceptable for the
   cost dashboard, or should the metric be tokens and latency only?

---

## Appendix A — Phase summary

| Phase | Theme | Key artefact | Test focus | Gate |
|---|---|---|---|---|
| 0 ✅ | Hardening | Sandbox, tenancy, persistence | Retrofit unit tests in P1 | — |
| 1 | FastAPI core | `api/` + `tests/` scaffolding | Unit + contract | Coverage ≥ 80 % |
| 2 | LLM layer | Provider abstraction, SSE tokens | Streaming adapter, structured output | First token < 1.5 s |
| 3 | RAG core | **`docs/ablation.md`** | Store contract, eval suite | hit@5 ≥ 0.80, citations ≥ 0.95 |
| 4 | StateGraph + HIL | Explicit graph, approval flow | Per-node + path tests | Branch coverage ≥ 90 % |
| 5 | Enterprise infra | Metrics, audit, Ops tab | Degradation matrix | p95 ≤ 3 s |
| 6 | Rancher deploy | Manifests + runbook | E2E through ingress | All pods ready, zero-downtime update |

## Appendix B — Test count estimate

| Layer | Approx. tests | Runtime |
|---|---|---|
| Unit | ~120 | < 30 s |
| Contract | ~30 | < 20 s |
| Integration | ~35 | 2–4 min |
| Eval | ~10 (over 50–100 golden pairs) | 5–15 min |
| E2E | ~12 | 3–5 min |

Inner loop (`-m "unit or contract"`) stays under a minute — the property that determines whether
the suite actually gets run.
