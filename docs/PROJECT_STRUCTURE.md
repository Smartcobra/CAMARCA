# Project Structure

CAMARCA is organized as a single Python package with operational helpers around it.

## Top-Level

- `camarca/` - application source code
- `data/` - input datasets (`logs`, `metrics`, `traces`, `graound_truth`)
- `outputs/` - generated artifacts (charts, RCA JSON, caches)
- `docs/` - design and operational docs
- `tests/` - automated tests

## Package Modules

- `cli.py`: user-facing CLI for hybrid RCA execution.
- `pipeline.py`: orchestration for window parsing, loading, graphing, fusion.
- `analysis.py`: feature extraction, anomaly signals, graph scoring, fusion heuristics.
- `ingest.py`: CSV/Parquet loading, timestamp normalization, filtering, bucketing.
- `prompt_pack.py`: multi-agent prompts and OpenAI chat contract.
- `rca_openai.py`: OpenAI-backed RCA flow for prompt-pack execution.
- `logging_utils.py`: shared logging bootstrap.
- `config.py`: tunable constants and fusion weights.

## Runtime Data Flow

1. CLI parses prompt and options.
2. Pipeline derives the time window.
3. Ingest loads and filters telemetry for that window.
4. Analysis builds dependency graph and fused ranking.
5. Output formatter emits RCA JSON.
