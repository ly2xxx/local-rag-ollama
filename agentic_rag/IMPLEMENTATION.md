# Agentic RAG Implementation: Complete Agent Architecture Pattern

This document tracks the technical implementation of the **Agentic RAG** subsystem in `local-rag-ollama`, designed according to **Section 3: Complete Agent Architecture** in [`Agent.md`](Agent.md).

---

## 1. Architecture Overview & Component Mapping

All agentic code is strictly isolated inside the [`agentic_rag/`](agentic_rag/) package. The table below illustrates how each architectural element from `Agent.md` is realized:

```mermaid
flowchart TD
    subgraph CoreAgent ["Agent Core (agentic_rag/core.py)"]
        AgentCore(("🤖 Agent Core<br/>LangGraph ReAct Loop"))
    end

    Profile["1. Profile<br/>(agentic_rag/profile.py)"] --> AgentCore
    Planning["2. Planning [Placeholder]<br/>(agentic_rag/planning.py)"] --> AgentCore
    MemoryST["3. Short-term Memory (Redis Checkpointer)<br/>(agentic_rag/memory/short_term.py)"] <--> AgentCore
    MemoryLT["3. Long-term Memory [Placeholder]<br/>(agentic_rag/memory/long_term.py)"] <--> AgentCore
    AgentCore --> Tools["4. Tools (RAG, Files, Math)<br/>(agentic_rag/tools.py)"]
    AgentCore --> Action["5. Action [Placeholder]<br/>(agentic_rag/action.py)"]
    Action --> Observation["6. Observation [Placeholder]<br/>(agentic_rag/observation.py)"]
    Observation --> AgentCore
    Orchestration["7. Orchestration [Placeholder]<br/>(agentic_rag/orchestration.py)"] -.-> AgentCore

    subgraph EngineeringEssentials ["Engineering Essentials"]
        Obs["🔭 Observability<br/>(agentic_rag/engineering/observability.py)"]
        Guard["🛡️ Guardrails<br/>(agentic_rag/engineering/guardrails.py)"]
        Eval["📊 Evaluation & Drift<br/>(agentic_rag/engineering/evaluation.py)"]
    end
    EngineeringEssentials -.-> CoreAgent
```

---

## 2. Directory Structure

```
local-rag-ollama/
├── agentic_rag/
│   ├── __init__.py                     # Package entry point exposing all components
│   ├── agent.py                        # High-level AgenticRAGHelper facade for Streamlit
│   ├── core.py                         # [IMPLEMENTED] AgentCore with LangGraph ReAct loop
│   ├── profile.py                      # [IMPLEMENTED] 1. Profile (Personas & Prompt constraints)
│   ├── planning.py                     # [PLACEHOLDER] 2. Planning (Task decomposition & reflection)
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── short_term.py               # [IMPLEMENTED] 3. Short-term Memory (RedisSaver checkpointer)
│   │   └── long_term.py                # [PLACEHOLDER] 3. Long-term Memory (Episodic & Summary store)
│   ├── tools.py                        # [IMPLEMENTED] 4. Tools (RAG vector query, file read, math)
│   ├── action.py                       # [PLACEHOLDER] 5. Action (Outbound dispatch & formatting)
│   ├── observation.py                  # [PLACEHOLDER] 6. Observation (Feedback parsing & reflection)
│   ├── orchestration.py                # [PLACEHOLDER] 7. Orchestration (Multi-agent supervisor graphs)
│   ├── judge/
│   │   └── ollama_judge.py             # [IMPLEMENTED] DeepEval LLM-as-a-Judge backed by Ollama
│   └── engineering/
│       ├── __init__.py
│       ├── observability.py            # [IMPLEMENTED] Observability (Step tracing & metrics)
│       ├── guardrails.py               # [IMPLEMENTED] Guardrails (Input/output safety & policies)
│       └── evaluation.py               # [IMPLEMENTED] Evaluation & Drift Detection (Judge scoring, baseline drift)
├── app.py                              # Streamlit app with 3 tabs: RAG, AGENT, Agentic RAG
├── pyproject.toml                      # Updated with langgraph, langgraph-checkpoint-redis, redis
├── README.md                           # Updated with Docker Redis deployment guide
├── Agent.md                            # Reference AI Agent architecture cheat sheet
└── IMPLEMENTATION.md                   # This implementation tracking document
```

---

## 3. Implemented Components Deep Dive

### A. Agent Core (`agentic_rag/core.py`)
- Built on **LangGraph** `create_react_agent` state machine.
- Binds `ChatOllama` (default `glm-5.2:cloud`), the toolset, system prompt profile, and the short-term memory checkpointer.
- Executes the `Thought -> Action (Tool Call) -> Observation (Tool Result) -> Answer` cycle.

### B. 4. Tools (`agentic_rag/tools.py`)
- `query_document_knowledge_base(query: str)`: Searches Chroma vector store containing ingested PDF chunks with cosine similarity.
- `read_local_file(file_path: str)`: Directly inspects local files (PDF text extraction via `pypdf` or raw text).
- `calculate_expression(expression: str)`: Computes mathematical formulas safely.
- `DocumentRetrieverManager`: Manages chunking, embedding (`all-MiniLM-L6-v2`), and Chroma indexing.

### C. 3. Short-term Memory & Redis Checkpointing (`agentic_rag/memory/short_term.py`)
- Integrates `langgraph-checkpoint-redis` with `RedisSaver` (and `AsyncRedisSaver`).
- Persists full multi-turn dialogue state and intermediate agent scratchpads keyed by `thread_id`.
- **Fault Tolerant**: Performs non-blocking ping checks on initialization (`redis://localhost:6379`). If Redis is not currently active, it automatically falls back to `MemorySaver` without failing.

### D. Helper Facade (`agentic_rag/agent.py`)
- Provides `AgenticRAGHelper` with clean methods:
  - `ingest_document(path)`: PDF ingestion into knowledge base.
  - `ask(query, thread_id)`: Executes ReAct loop and extracts final answer + structured scratchpads.
  - `get_redis_status()`: Inspects Redis connection state for UI rendering.
  - `clear_documents()`: Resets knowledge base.

### E. Engineering Essentials & Quality Evaluation (`agentic_rag/engineering/`)
- **Observability** ([`observability.py`](agentic_rag/engineering/observability.py)): Traces step execution latencies, input/output structures, and trajectory breakdowns.
- **Guardrails** ([`guardrails.py`](agentic_rag/engineering/guardrails.py)): Pre-flight input length/injection checks and post-flight response sanitation.
- **Evaluation & Drift Detection** ([`evaluation.py`](agentic_rag/engineering/evaluation.py)):
  - Plugs into [`OllamaJudge`](agentic_rag/judge/ollama_judge.py) for scoring RAG metrics (faithfulness, answer relevancy).
  - Implements sliding-window statistical drift tracking against established benchmark baselines.
  - Supports deterministic golden-dataset batch regression benchmarking.

---

## 4. UI Integration in `app.py`

Tab 3 (**"Agentic RAG"**) provides:
1. **Redis Status Indicator**: Live badge reflecting active Redis connection or in-memory fallback.
2. **Document Ingestion**: Multi-file PDF uploader dedicated to Agentic RAG.
3. **Session / Thread ID Selector**: Configurable `thread_id` to demonstrate multi-turn persistence across sessions.
4. **Interactive Chat & Scratchpad Expander**: Shows conversation messages alongside collapsible **"🔍 Inspect Scratchpad & Tool Executions"** showing intermediate tool calls, inputs, and observations.
5. **Reset Button**: Resets current thread state and document stores.

---

## 5. Hosting Local Redis with Docker

As documented in `README.md`, start a local Redis container with:

```bash
# Standard lightweight Redis
docker run -d --name local-redis -p 6379:6379 redis:alpine

# Or Redis Stack with RedisInsight UI on port 8001
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

---

## 6. CI/CD Baseline Collection & Quality Gate (GitHub Actions)

In production LLMOps, the baseline distributions used for live model drift detection **must never be hardcoded in application code**. Instead, baselines are generated deterministically in pre-deployment CI/CD pipelines by running a curated **Golden Test Dataset** through LLM-as-a-Judge and saving the resulting artifact to [`baseline_metrics.json`](agentic_rag/engineering/baseline_metrics.json).

### A. CI/CD Pre-Deployment Flow

```mermaid
flowchart LR
    subgraph CI_Pipeline ["GitHub Actions CI Pipeline"]
        Push["1. Git Push / PR"] --> StartOllama["2. Start Ollama Service / API"]
        StartOllama --> RunEval["3. Run Golden Dataset Benchmark<br/>(scripts/run_ci_eval.py)"]
        RunEval --> CheckGate{"4. Mean Score >= 0.85 &<br/>No Regressions?"}
        CheckGate -- Fail --> Block["❌ Fail CI & Block Merge"]
        CheckGate -- Pass --> ExportArtifact["5. Export baseline_metrics.json"]
        ExportArtifact --> Deploy["6. Deploy Container to Production"]
    end

    subgraph Runtime_Deployment ["Live Production Application"]
        Deploy --> Streamlit["Streamlit app.py"]
        ExportArtifact -.->|Loaded on App Boot| EvalMgr["EvaluationManager<br/>(Live Drift Detection)"]
    end
```

### B. Example GitHub Actions Workflow (`.github/workflows/eval-gate.yml`)

```yaml
name: Agentic RAG Evaluation & Baseline Gate

on:
  pull_request:
    branches: [ main ]
  push:
    branches: [ main ]

jobs:
  evaluate-baseline:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4

      - name: Install uv & Python
        uses: astral-sh/setup-uv@v2
        with:
          python-version: "3.11"

      - name: Install Dependencies
        run: uv sync

      - name: Launch Background Ollama Runner
        run: |
          docker run -d --name ollama-ci -p 11434:11434 ollama/ollama:latest
          sleep 5
          docker exec ollama-ci ollama pull glm-5.2:cloud || true

      - name: Run Golden Benchmark & Export Baseline Artifact
        env:
          OLLAMA_BASE_URL: "http://localhost:11434/v1"
        run: |
          uv run python -m agentic_rag.engineering.evaluation

      - name: Upload Baseline Artifact
        uses: actions/upload-artifact@v4
        with:
          name: baseline-metrics
          path: agentic_rag/engineering/baseline_metrics.json
          retention-days: 30
```

### C. Runtime Artifact Ingestion
When [`app.py`](app.py) boots up in staging or production, it calls `eval_mgr.load_baseline_from_file("agentic_rag/engineering/baseline_metrics.json")`. The live conversation turns are then compared against the exact benchmark distribution captured during the successful CI/CD release.
