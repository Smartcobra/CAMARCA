"""
Production prompt pack for LLM-based RCA orchestration (LangGraph-friendly).

This module provides:
- Agent prompts with strict output contracts
- Confidence calibration rubric
- Fallback and uncertainty policy
- JSON parsing/validation for supervisor output
- Simple node factories usable in a LangGraph workflow
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, TypedDict

from camarca.settings import OpenAISettings

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SHARED_GUARDRAILS = """
Rules:
1) Use only the provided input. Do not invent services, metrics, or failures.
2) If evidence is weak or conflicting, explicitly say so.
3) Keep output concise and structured.
4) If no clear signal exists, return "insufficient_evidence".
"""


LOG_PROMPT = """
You are a log analysis expert.

Given log templates and error counts:
{log_data}

Identify:
- Error patterns
- Affected services
- Severity

Output format (strict JSON object):
{{
  "error_patterns": ["..."],
  "affected_services": ["..."],
  "severity": "low|medium|high|insufficient_evidence",
  "summary": "..."
}}
{guardrails}
"""


TRACE_PROMPT = """
You are a distributed tracing expert.

Given trace graph and anomalies:
{trace_data}

Identify:
- Slow services
- Failed calls
- Propagation path

Output format (strict JSON object):
{{
  "slow_services": ["..."],
  "failed_calls": ["serviceA->serviceB ..."],
  "propagation_path": ["service1", "service2"],
  "summary": "..."
}}
{guardrails}
"""


METRIC_PROMPT = """
You are a system performance expert.

Given metrics:
{metric_data}

Identify:
- CPU spikes
- Latency anomalies
- Resource bottlenecks

Output format (strict JSON object):
{{
  "cpu_spikes": ["..."],
  "latency_anomalies": ["..."],
  "resource_bottlenecks": ["..."],
  "impacted_services": ["..."],
  "summary": "..."
}}
{guardrails}
"""


GRAPH_PROMPT = """
You are a root cause analysis expert.

Given:
- Trace insights: {trace_summary}
- Log insights: {log_summary}
- Metric insights: {metric_summary}
- Ranked services: {rca_rank}

Determine:
- Most probable root cause
- Why it caused failure
- How it propagated

Output format (strict JSON object):
{{
  "candidate_root_cause": "...",
  "propagation": "...",
  "reasoning": "...",
  "alternatives": ["..."],
  "evidence_strength": "low|medium|high|insufficient_evidence"
}}
{guardrails}
"""


CONFIDENCE_RUBRIC = """
Confidence rubric (0.0-1.0):
- 0.85-1.00: all three modalities align + top-ranked service corroborated.
- 0.65-0.84: two modalities align with minor conflicts.
- 0.40-0.64: single strong modality, weak corroboration.
- 0.00-0.39: conflicting/insufficient evidence.
"""


SUPERVISOR_PROMPT = """
You are the final RCA decision engine.

Combine all insights:
{graph_reasoning}

Apply this confidence rubric:
{confidence_rubric}

Output strict JSON only:
{{
  "root_cause": "...",
  "confidence": 0.0,
  "reasoning": "...",
  "evidence": {{
    "trace": "...",
    "metrics": "...",
    "logs": "..."
  }}
}}

If evidence is insufficient, output:
{{
  "root_cause": "insufficient_evidence",
  "confidence": <= 0.39,
  "reasoning": "...",
  "evidence": {{
    "trace": "...",
    "metrics": "...",
    "logs": "..."
  }}
}}
{guardrails}
"""


# ---------------------------------------------------------------------------
# State + validation
# ---------------------------------------------------------------------------


class RCAState(TypedDict, total=False):
    """Minimal LangGraph-compatible shared state."""

    log_data: str
    trace_data: str
    metric_data: str
    rca_rank: str
    log_summary: str
    trace_summary: str
    metric_summary: str
    graph_reasoning: str
    final_json: dict[str, Any]


class SupervisorOutput(TypedDict):
    root_cause: str
    confidence: float
    reasoning: str
    evidence: dict[str, str]


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return text[start : end + 1]


def parse_supervisor_output(text: str) -> SupervisorOutput:
    """Parse and validate strict supervisor JSON."""
    raw = json.loads(_extract_json_object(text))
    required_top = {"root_cause", "confidence", "reasoning", "evidence"}
    missing = required_top - set(raw.keys())
    if missing:
        raise ValueError(f"Supervisor output missing keys: {sorted(missing)}")
    if not isinstance(raw["confidence"], (int, float)):
        raise ValueError("confidence must be numeric")
    confidence = float(raw["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    evidence = raw["evidence"]
    if not isinstance(evidence, dict):
        raise ValueError("evidence must be object")
    for k in ("trace", "metrics", "logs"):
        if k not in evidence:
            raise ValueError(f"evidence missing '{k}'")
    return {
        "root_cause": str(raw["root_cause"]),
        "confidence": confidence,
        "reasoning": str(raw["reasoning"]),
        "evidence": {
            "trace": str(evidence["trace"]),
            "metrics": str(evidence["metrics"]),
            "logs": str(evidence["logs"]),
        },
    }


def apply_fallback_policy(out: SupervisorOutput, rca_rank_top1: str | None) -> SupervisorOutput:
    """
    Conservative fallback:
    - If confidence < 0.4, force insufficient_evidence
    - If root_cause is empty but confidence is high, use top-ranked service hint
    """
    fixed = dict(out)
    root = str(fixed.get("root_cause", "")).strip()
    conf = float(fixed.get("confidence", 0.0))
    if conf < 0.4:
        fixed["root_cause"] = "insufficient_evidence"
    elif not root and rca_rank_top1:
        fixed["root_cause"] = rca_rank_top1
        fixed["reasoning"] = (
            str(fixed.get("reasoning", ""))
            + " Root cause defaulted to top-ranked service due to missing label."
        ).strip()
    return fixed  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_log_prompt(log_data: str) -> str:
    return LOG_PROMPT.format(log_data=log_data, guardrails=SHARED_GUARDRAILS)


def build_trace_prompt(trace_data: str) -> str:
    return TRACE_PROMPT.format(trace_data=trace_data, guardrails=SHARED_GUARDRAILS)


def build_metric_prompt(metric_data: str) -> str:
    return METRIC_PROMPT.format(metric_data=metric_data, guardrails=SHARED_GUARDRAILS)


def build_graph_prompt(
    trace_summary: str,
    log_summary: str,
    metric_summary: str,
    rca_rank: str,
) -> str:
    return GRAPH_PROMPT.format(
        trace_summary=trace_summary,
        log_summary=log_summary,
        metric_summary=metric_summary,
        rca_rank=rca_rank,
        guardrails=SHARED_GUARDRAILS,
    )


def build_supervisor_prompt(graph_reasoning: str) -> str:
    return SUPERVISOR_PROMPT.format(
        graph_reasoning=graph_reasoning,
        confidence_rubric=CONFIDENCE_RUBRIC,
        guardrails=SHARED_GUARDRAILS,
    )


# ---------------------------------------------------------------------------
# LangGraph-style node factories
# ---------------------------------------------------------------------------

LLMCall = Callable[[str], str]


@dataclass
class RCAAgentPack:
    """
    Agent-node factory using a simple callable `llm(prompt) -> text`.

    You can use these node callables directly in a LangGraph state graph.
    """

    llm: LLMCall

    def log_agent(self, state: RCAState) -> RCAState:
        prompt = build_log_prompt(state.get("log_data", ""))
        return {"log_summary": self.llm(prompt)}

    def trace_agent(self, state: RCAState) -> RCAState:
        prompt = build_trace_prompt(state.get("trace_data", ""))
        return {"trace_summary": self.llm(prompt)}

    def metric_agent(self, state: RCAState) -> RCAState:
        prompt = build_metric_prompt(state.get("metric_data", ""))
        return {"metric_summary": self.llm(prompt)}

    def graph_agent(self, state: RCAState) -> RCAState:
        prompt = build_graph_prompt(
            trace_summary=state.get("trace_summary", ""),
            log_summary=state.get("log_summary", ""),
            metric_summary=state.get("metric_summary", ""),
            rca_rank=state.get("rca_rank", ""),
        )
        return {"graph_reasoning": self.llm(prompt)}

    def supervisor_agent(self, state: RCAState) -> RCAState:
        prompt = build_supervisor_prompt(state.get("graph_reasoning", ""))
        raw = self.llm(prompt)
        parsed = parse_supervisor_output(raw)
        top1 = None
        if state.get("rca_rank"):
            top1 = str(state["rca_rank"]).split(",")[0].strip()
        final = apply_fallback_policy(parsed, top1)
        return {"final_json": final}


# ---------------------------------------------------------------------------
# OpenAI client integration
# ---------------------------------------------------------------------------


@dataclass
class OpenAIChatClient:
    """
    Callable OpenAI client adapter: ``client(prompt) -> text``.

    Environment:
    - OPENAI_API_KEY (required unless passed explicitly)
    - OPENAI_MODEL (optional, default: gpt-4o-mini)
    """

    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    api_key: str | None = None

    def __post_init__(self) -> None:
        from openai import OpenAI  # lazy import for optional runtime dependency

        env = OpenAISettings.from_env()
        if self.model == "gpt-4o-mini":
            self.model = env.model
        if self.temperature == 0.0:
            self.temperature = env.temperature
        key = self.api_key or env.api_key
        if not key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIChatClient.")
        self._client = OpenAI(api_key=key)

    def __call__(self, prompt: str) -> str:
        resp = self._client.responses.create(
            model=self.model,
            temperature=self.temperature,
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
        )
        return getattr(resp, "output_text", "").strip()


def run_openai_rca(
    *,
    log_data: str,
    trace_data: str,
    metric_data: str,
    rca_rank: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> SupervisorOutput:
    """
    End-to-end prompt-pack orchestration using OpenAI.

    Order:
    LogAgent + TraceAgent + MetricAgent -> GraphAgent -> SupervisorAgent
    """
    llm = OpenAIChatClient(model=model, temperature=temperature)
    pack = RCAAgentPack(llm=llm)

    state: RCAState = {
        "log_data": log_data,
        "trace_data": trace_data,
        "metric_data": metric_data,
        "rca_rank": rca_rank,
    }
    state.update(pack.log_agent(state))
    state.update(pack.trace_agent(state))
    state.update(pack.metric_agent(state))
    state.update(pack.graph_agent(state))
    state.update(pack.supervisor_agent(state))
    return state["final_json"]  # type: ignore[return-value]
