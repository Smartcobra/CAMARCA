# CAMARCA

> This file is kept for backward compatibility.  
> Use `README.md` as the primary project documentation.

CAMARCA is an observability root-cause analysis pipeline that combines:
- logs
- traces
- metrics

It builds time-bucketed features, extracts log templates, builds a trace call graph,
detects anomalies on metric aggregates, and ranks likely root-cause services.

## Project Structure

- `main.py` - compatibility wrapper to hybrid CLI
- `camarca/pipeline.py` - hybrid RCA orchestration
- `camarca/cli.py` - hybrid RCA command-line interface
- `camarca/ingest.py` - CSV/Parquet loading and time normalization
- `camarca/analysis.py` - feature extraction, graph building, anomaly + RCA
- `camarca/convert_to_parquet.py` - one-time CSV -> Parquet conversion helper

## Requirements

- Python 3.13+
- `uv` installed

## Setup

From the project root:

```bash
uv sync
```

Create env file:

```bash
cp .env.example .env
```

Then edit `.env` and set `OPENAI_API_KEY`.

## Run Steps

### 1) (Optional but recommended) Convert CSV files to Parquet

This improves load speed for large datasets.

```bash
uv run python -m camarca.convert_to_parquet
```

Or via script:

```bash
uv run camarca-convert
```

Create hour-partitioned parquet for faster fault-window reads:

```bash
uv run camarca-convert-partitioned
```

Preview conversion only:

```bash
uv run python -m camarca.convert_to_parquet --dry-run
```

### 2) Run hybrid RCA

```bash
uv run camarca-hybrid --prompt "A fault occurred from 2022-05-09T06:40:17Z to 2022-05-09T06:50:17Z. Please identify the root cause."
```

Or via the default package script alias:

```bash
uv run camarca --prompt "A fault occurred from 2022-05-09T06:40:17Z to 2022-05-09T06:50:17Z. Please identify the root cause."
```

### 3) Optional output controls

Use predicted service labels:

```bash
uv run camarca-hybrid --prompt "A fault occurred from 2022-05-09T06:40:17Z to 2022-05-09T06:50:17Z. Please identify the root cause." --component-granularity predicted
```

Use GT-aligned component granularity (default):

```bash
uv run camarca-hybrid --prompt "A fault occurred from 2022-05-09T06:40:17Z to 2022-05-09T06:50:17Z. Please identify the root cause." --component-granularity gt-aligned
```

## What the run prints

- competition-style RCA JSON (`uuid`, `component`, `reason`, `reasoning_trace`)
- optional debug print (`--output-format debug`) with rows, top fused services, GT match

## Notes

- Hybrid window loading is implemented in `camarca/pipeline.py`.
- Parquet is preferred automatically when available (`camarca/config.py`).
- For very large data, main bottlenecks are log-template regex grouping and trace parent-service enrichment.

## LLM Prompt Pack (Production)

The project includes a production-oriented multi-agent prompt pack in `camarca/prompt_pack.py`.

Agent flow:

`Input Data -> LogAgent/TraceAgent/MetricAgent -> GraphAgent -> SupervisorAgent`

It provides:
- hardened prompts with guardrails
- confidence rubric (0.0-1.0)
- strict supervisor JSON contract
- output parsing + validation
- fallback policy (`insufficient_evidence` when confidence is low)
- LangGraph-friendly node factory (`RCAAgentPack`)

### OpenAI client integration

Set environment variable:

```bash
export OPENAI_API_KEY="your_key_here"
```

Or set it once in `.env` (recommended for local development).

Run OpenAI-backed multi-agent RCA:

```bash
uv run camarca-rca-openai --prompt "A fault occurred from 2022-05-09T06:40:17Z to 2022-05-09T06:50:17Z. Please identify the root cause." --model gpt-4o-mini
```

This writes final JSON to `outputs/rca_openai.json`.

## Hybrid Pipeline (MicroRCA + RC-LLM Inspired)

Implemented hybrid flow:
- Dynamic service dependency graph with weighted edges:
  - latency anomaly
  - error propagation
  - call frequency
- Multi-agent workflow:
  - LogAgent
  - TraceAgent
  - MetricAgent
  - GraphReasoningAgent
  - SupervisorAgent
- Residual confidence fusion:
  - `w1*trace_conf + w2*metric_conf + w3*log_conf + w4*graph_causality`
- Iterative reasoning loop:
  - hypothesis -> validate -> refine -> converge

Run hybrid on a natural language fault prompt:

```bash
uv run camarca-hybrid --prompt "A fault occurred from 2022-05-09T06:40:17Z to 2022-05-09T06:50:17Z. Please identify the root cause."
```

With LLM iterative loop:

```bash
uv run camarca-hybrid --with-llm --model gpt-4o-mini --prompt "A fault occurred from 2022-05-09T06:40:17Z to 2022-05-09T06:50:17Z. Please identify the root cause."
```

Performance acceleration used by hybrid mode:
- Reads from `data/partitioned/*/dt=YYYY-MM-DD/hour=HH/*.parquet` when available
- Caches per-window modality slices in `outputs/cache/windows/`

- **Multi-fault windows:** current output returns single top-1 `component` and `reason`.  
  For intervals containing multiple ground-truth events, add multi-component output support:
  - `components`: top-k ranked components (e.g., top 2 or top 3)
  - per-component `reasoning_trace`
  - optional `ground_truth_matches` against all events in the selected window
- **Schema mode:** add output switch for:
  - compact single-fault mode (current)
  - multi-fault mode (top-k)


##Input and output comparision--

now user passed a prompt "A fault occurred from2022-05-09T06:40:17.000Z to 2022-05-09T06:50:17.000Z . Please identify the root cause."
---productcatalogservice
--1652078717,service,productcatalogservice,Kubernetes Container Network Resource Packet Corruption



now user passed a prompt "A fault occurred from 2022-05-09T06:17:02.000Z to 2022-05-09T06:18:40.000Z . Please identify the root cause."
--2022-05-09T06:18:03.000Z
---1652077083,service,cartservice,Kubernetes Container Network Resource Packet Corruption


--2022-05-09T10:52:24.000Z
1652093544,pod,paymentservice-2,Kubernetes Container Process Termination
now user passed a prompt "A fault occurred from 022-05-09T10:50:24.000Z to 022-05-09T10:53:24.000Z . Please identify the root cause."

--1652087378,node,node-3,Node Disk Write I/O Consumption
--2022-05-09T09:09:38.000Z


--1652046023,service,emailservice,Kubernetes Container Write I/O Load
2022-05-08T21:40:23.000Z
now user passed a prompt "A fault occurred from 2022-05-08T21:38:23.000Z to 2022-05-08T21:41:23.000Z . Please identify the root cause."

--1652102061,pod,cartservice-2,Kubernetes Container Network Resource Packet Corruption
---2022-05-09T13:14:21.000Z
now user passed a prompt "A fault occurred from 2022-05-09T13:12:21.000Z to 2022-05-09T13:15:21.000Z . Please identify the root cause."

--1652031460,service,shippingservice,Kubernetes Container Read I/O Load
--2022-05-08T17:37:40.000Z
now user passed a prompt "A fault occurred from 2022-05-08T17:35:40.000Z to 2022-05-08T17:38:40.000Z . Please identify the root cause."

--1652078717,service,productcatalogservice,Kubernetes Container Network Resource Packet Corruption
--2022-05-09T06:45:17.000Z
now user passed a prompt "A fault occurred from 2022-05-09T06:45:16.000Z to 2022-05-09T06:45:18.000Z . Please identify the root cause."


2022-05-08T22:36:07.000Z
1652049367,service,productcatalogservice,Kubernetes Container Network Packet Loss
2022-05-08T23:06:40.000Z
1652051200,pod,recommendationservice-2,Kubernetes Container Memory Load

now user passed a prompt " multiple fault occurred from 2022-05-08T22:34:07.000Z to 2022-05-08T23:07:40.000Z . Please identify the root cause."

## Future Implementation Notes

