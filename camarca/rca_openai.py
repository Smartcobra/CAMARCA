"""Run RCA prompt-pack with OpenAI from hybrid pipeline outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camarca.logging_utils import setup_logging
from camarca.pipeline import default_agent_payloads, run_hybrid_from_prompt
from camarca.prompt_pack import run_openai_rca
from camarca.settings import OpenAISettings, load_environment


def _to_agent_inputs(*, prompt: str) -> dict[str, str]:
    r = run_hybrid_from_prompt(prompt)
    log_data, trace_data, metric_data = default_agent_payloads(r)
    rca_rank = ", ".join([svc for svc, _ in r.fused_ranking[:10]])
    return {
        "log_data": log_data,
        "trace_data": trace_data,
        "metric_data": metric_data,
        "rca_rank": rca_rank,
    }


def main() -> None:
    load_environment()
    parser = argparse.ArgumentParser(description="Run OpenAI-backed RCA prompt pack.")
    parser.add_argument(
        "--prompt",
        required=True,
        help='Example: "A fault occurred from 2022-05-09T06:40:17Z to 2022-05-09T06:50:17Z. Please identify the root cause."',
    )
    env = OpenAISettings.from_env()
    parser.add_argument("--model", default=env.model, help="OpenAI model name.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=env.temperature,
        help="LLM temperature.",
    )
    parser.add_argument(
        "--output",
        default="outputs/rca_openai.json",
        help="Write final supervisor JSON to this path.",
    )
    args = parser.parse_args()

    setup_logging()
    payload = _to_agent_inputs(prompt=args.prompt)
    final = run_openai_rca(
        log_data=payload["log_data"],
        trace_data=payload["trace_data"],
        metric_data=payload["metric_data"],
        rca_rank=payload["rca_rank"],
        model=args.model,
        temperature=args.temperature,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    print(json.dumps(final, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
