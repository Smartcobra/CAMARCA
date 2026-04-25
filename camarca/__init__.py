"""Observability RCA package (hybrid RCA + LLM prompt-pack helpers)."""

__version__ = "0.1.0"

from camarca.logging_utils import setup_logging
from camarca.pipeline import (
    HybridRCAResult,
    compare_with_ground_truth,
    default_agent_payloads,
    format_competition_output,
    iterative_reasoning,
    parse_fault_prompt_window,
    parse_uuid_from_prompt,
    run_hybrid_from_prompt,
    run_hybrid_window,
)
from camarca.prompt_pack import OpenAIChatClient, RCAAgentPack, RCAState, run_openai_rca

__all__ = [
    "HybridRCAResult",
    "OpenAIChatClient",
    "RCAAgentPack",
    "RCAState",
    "compare_with_ground_truth",
    "default_agent_payloads",
    "format_competition_output",
    "iterative_reasoning",
    "parse_fault_prompt_window",
    "parse_uuid_from_prompt",
    "run_hybrid_from_prompt",
    "run_hybrid_window",
    "run_openai_rca",
    "setup_logging",
    "__version__",
]
