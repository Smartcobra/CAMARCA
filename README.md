# CAMARCA

Production-oriented root cause analysis (RCA) for microservice incidents using logs, traces, and metrics.

## What It Does

- Parses natural-language fault windows.
- Loads windowed observability data (with cache + partition support).
- Builds a dynamic service dependency graph.
- Fuses graph + modality confidence signals for root-cause ranking.
- Emits competition-style RCA output JSON.

## Architecture Diagram

```mermaid
flowchart LR
    U["User / CLI Prompt"] --> C["`camarca/cli.py`"]
    C --> P["`camarca/pipeline.py`<br/>Hybrid RCA Orchestrator"]
    C --> OAI["`camarca/rca_openai.py`<br/>OpenAI RCA Runner"]

    subgraph Inputs["Input Data Sources (`data/`)"]
        L["Logs"]
        T["Traces"]
        M["Metrics"]
        G["Ground Truth"]
    end

    subgraph Ingestion["Ingestion + Windowing"]
        I["`camarca/ingest.py`<br/>Load + normalize + bucket"]
        PARQ["Parquet / Partitioned Parquet"]
        CACHE["Window Cache<br/>`outputs/cache/windows/`"]
    end

    subgraph Reasoning["Analysis + RCA Reasoning"]
        A["`camarca/analysis.py`<br/>Feature extraction + anomalies"]
        DG["Dynamic Service Dependency Graph"]
        F["Confidence Fusion<br/>trace + metric + log + graph causality"]
        PP["`camarca/prompt_pack.py`<br/>Log/Trace/Metric/Graph/Supervisor agents"]
    end

    subgraph Output["Outputs (`outputs/`)"]
        J["RCA JSON<br/>component + reason + reasoning trace"]
        CH["Charts / Debug artifacts"]
    end

    L --> I
    T --> I
    M --> I
    G --> I
    PARQ --> I
    I --> CACHE
    I --> P
    P --> A
    A --> DG
    DG --> F
    A --> F
    F --> J
    F --> CH
    OAI --> PP
    PP --> J
```

Animated HTML version: `docs/animated_architecture.html`

## Repository Layout

- `camarca/` - application package
  - `cli.py` - main CLI entrypoint
  - `pipeline.py` - hybrid RCA pipeline orchestration
  - `analysis.py` - graph/anomaly/fusion logic
  - `ingest.py` - data loading and time normalization
  - `prompt_pack.py` - LLM multi-agent prompt pack
  - `rca_openai.py` - OpenAI-backed RCA run flow
- `data/` - input datasets
- `outputs/` - generated outputs and caches
- `docs/` - project documentation
- `tests/` - test suite
- `main.py` - compatibility wrapper to `camarca.cli`

## Quick Start

```bash
uv sync
cp .env.example .env
uv run camarca --prompt "A fault occurred from 2022-05-09T06:40:17Z to 2022-05-09T06:50:17Z. Please identify the root cause."
```

## Common Commands

```bash
# Hybrid RCA
uv run camarca --prompt "A fault occurred from 2022-05-09T06:17:02Z to 2022-05-09T06:18:40Z. Please identify the root cause."

# Force predicted labels instead of GT-aligned labels
uv run camarca --prompt "..." --component-granularity predicted

# Conversion helpers
uv run camarca-convert
uv run camarca-convert-partitioned

# OpenAI-backed run
uv run camarca-rca-openai --prompt "A fault occurred from 2022-05-09T06:40:17Z to 2022-05-09T06:50:17Z. Please identify the root cause."
```

## FastAPI + Streamlit Live UI

Run backend API (includes SSE stream endpoint):

```bash
uv run uvicorn camarca.web_api:app --host 0.0.0.0 --port 8001 --reload
```

Run Streamlit UI in another terminal:

```bash
uv run streamlit run streamlit_app.py
```

Endpoints:
- `GET /health`
- `GET /api/rca?prompt=...`
- `GET /api/stream?prompt=...` (SSE)
