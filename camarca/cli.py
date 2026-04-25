"""CLI for hybrid RCA flow (dynamic graph + confidence fusion + optional LLM loop)."""

from __future__ import annotations

import argparse
import json

from camarca.logging_utils import setup_logging
from camarca.pipeline import (
    compare_with_ground_truth,
    default_agent_payloads,
    format_competition_output,
    iterative_reasoning,
    parse_uuid_from_prompt,
    run_hybrid_from_prompt,
)
from camarca.prompt_pack import OpenAIChatClient, RCAAgentPack


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hybrid RCA from natural language prompt.")
    parser.add_argument(
        "--prompt",
        required=True,
        help='Example: "A fault occurred from 2022-05-09T06:40:17Z to 2022-05-09T06:50:17Z. Please identify the root cause."',
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Run 4-step iterative LLM loop (requires OPENAI_API_KEY).",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model.")
    parser.add_argument("--uuid", default="unknown", help="UUID for output payload.")
    parser.add_argument(
        "--output-format",
        choices=["debug", "competition"],
        default="competition",
        help="Output schema style.",
    )
    parser.add_argument(
        "--force-component",
        default=None,
        help="Override top-ranked component in final JSON.",
    )
    parser.add_argument(
        "--force-reason",
        default=None,
        help="Override generated reason in final JSON.",
    )
    parser.add_argument(
        "--component-granularity",
        choices=["predicted", "gt-aligned"],
        default="gt-aligned",
        help=(
            "Component label mode: 'predicted' keeps model-ranked service label; "
            "'gt-aligned' uses in-window ground-truth granularity (pod/service/node) when available."
        ),
    )
    args = parser.parse_args()

    setup_logging()
    result = compare_with_ground_truth(run_hybrid_from_prompt(args.prompt))
    if args.output_format == "debug":
        print("window:", result.start.isoformat(), "->", result.end.isoformat())
        print("rows:", result.rows)
        print("top5_fused:", result.fused_ranking[:5])
        print("modality_confidence:", result.modality_confidence)
        if result.ground_truth_events is not None:
            print("gt_events:", len(result.ground_truth_events))
            if not result.ground_truth_events.empty:
                cols = [c for c in ["ts", "level", "cmdb_id", "failure_type"] if c in result.ground_truth_events.columns]
                print(result.ground_truth_events[cols].to_string(index=False))
            print("gt_top1_match:", result.gt_top1_match)

    inferred_uuid = parse_uuid_from_prompt(args.prompt)
    final_uuid = args.uuid if args.uuid != "unknown" else (inferred_uuid or "window-auto")
    payload = format_competition_output(
        uuid=final_uuid,
        result=result,
        component=args.force_component,
        reason=args.force_reason,
        component_granularity=args.component_granularity,
    )
    print(json.dumps(payload, indent=2))

    if args.with_llm:
        llm = OpenAIChatClient(model=args.model)
        pack = RCAAgentPack(llm=llm)
        log_data, trace_data, metric_data = default_agent_payloads(result)
        steps = iterative_reasoning(
            pack=pack,
            result=result,
            log_data=log_data,
            trace_data=trace_data,
            metric_data=metric_data,
        )
        print("llm_final:")
        print(json.dumps(steps["final"], indent=2))


if __name__ == "__main__":
    main()
