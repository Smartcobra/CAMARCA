# Research Papers vs CAMARCA Proposal

This document compares the implemented CAMARCA RCA workflow with the two referenced papers:

- **MicroRCA-Agent**
- **RC-LLM**

It focuses on architecture, reasoning/output design, evidence handling, and operational behavior.

## 1) High-Level Objective Alignment

| Area | Paper Direction | CAMARCA Implementation | Status |
|---|---|---|---|
| Multi-source RCA | Use traces + metrics + logs together | Uses all three modalities with fusion in `camarca/pipeline.py` and `camarca/analysis.py` | Aligned |
| Structured RCA output | Return machine-readable diagnosis and reasoning trace | Competition JSON with `component`, `reason`, `reasoning_trace` | Aligned |
| Reasoning workflow | Ordered analysis steps across modalities | Step sequence: metrics -> trace -> logs with top evidence sections | Aligned |
| Practical deployment | Robust to sparse or noisy telemetry | Added cache fallback, halo window retrieval, service normalization | Partially aligned (improving) |

## 2) Architecture Comparison

### MicroRCA-Agent style

Paper-style characteristics (as used in this project context):

- Explicit step-wise diagnosis chain.
- Evidence retrieval per modality.
- JSON output with clear reasoning steps.

CAMARCA mapping:

- `reasoning_trace` uses explicit `step/action/observation`.
- Each step now includes evidence count style observations.
- Top evidence snippets are attached:
  - `top_metric_evidence` (max 3)
  - `top_trace_evidence` (max 3)
  - `top_log_evidence` (max 3)

### RC-LLM style

Paper-style characteristics (as used in this project context):

- Confidence-aware multimodal reasoning.
- Graph-informed service causality.
- Iterative refinement and residual fusion.

CAMARCA mapping:

- Dynamic service dependency graph in `analysis.py`.
- Confidence fusion via modality + graph terms.
- Iterative reasoning support via prompt pack and OpenAI flow (`prompt_pack.py`, `rca_openai.py`).

## 3) Output Format Design Comparison

## Paper-style output intent

- A strict JSON contract.
- Stepwise explanations with compact observations.
- Evidence-backed diagnosis, not just a final label.

## CAMARCA current output

- JSON schema:
  - `uuid`
  - `component`
  - `reason`
  - `reasoning_trace`
- `reasoning_trace` now includes:
  - human-readable evidence counts (`X ... matched`)
  - structured top evidence arrays for interpretability.

## Key update made

To better mirror paper-style readability:

- Observations were changed from heuristic prose to count-style phrasing.
- Long log evidence strings were moved into structured arrays.
- Top evidence is capped to reduce payload noise.

## 4) Evidence Handling Comparison

| Capability | Paper expectation | CAMARCA behavior | Notes |
|---|---|---|---|
| Metric evidence retrieval | Retrieve meaningful nearby metric context | Strict window + progressive halo fallback | Improved for short windows |
| Service identity consistency | Consistent service naming across modalities | Added metric service normalization (`-grpc`, `-http`, pod index stripping) | Improved |
| Trace explainability | Show strongest dependency paths | `top_trace_evidence` with top incoming edges | Aligned |
| Log explainability | Show most representative matched logs | `top_log_evidence` top 3 templates + counts | Aligned |

## 5) Where CAMARCA Differs From Papers

1. **No full LLM-only diagnosis loop by default**
   - Default CLI path is deterministic/rule + graph fusion.
   - LLM loop is optional (`--with-llm` paths).

2. **Competitive output constraints**
   - Output is shaped for benchmark/competition schema.
   - Some paper-specific rich metadata is intentionally omitted for compactness.

3. **Operational shortcuts for latency**
   - Caching and partitioned loading are prioritized for responsiveness.
   - This may differ from pure research prototype pipelines.

## 6) Strengths of the Proposed Solution

- Production-leaning reliability improvements:
  - empty-cache metric fallback,
  - progressive halo retrieval for short windows,
  - explicit modality-level evidence counts.
- Better explainability than a plain ranked label:
  - top metric/trace/log evidence exposed in output.
- Works with very heterogeneous metric folders (`container`, `istio`, `jvm`, `node`, `service`) through recursive loading.

## 7) Remaining Gaps and Recommended Next Steps

1. **Metric evidence granularity**
   - Add `top_metric_evidence` ranking by anomaly-first then value magnitude.

2. **Trace evidence richness**
   - Include optional edge attributes (latency/error/call frequency) behind a debug flag.

3. **Log readability**
   - Further clean template artifacts (punctuation collapse and token post-processing).

4. **Paper-faithful mode switch**
   - Add a `--reasoning-style {paper,compact}` flag to support strict research-style traces versus production compact output.

## 8) Summary

CAMARCA is now closely aligned with the referenced paper intent at the **reasoning-output and multimodal fusion level**, while adding production-oriented handling for sparse windows and inconsistent telemetry labels. The main divergence is that CAMARCA defaults to a deterministic hybrid flow and uses LLM stages optionally, which is practical for repeatability and latency.
