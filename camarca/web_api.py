"""FastAPI app exposing CAMARCA RCA with SSE streaming progress."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from camarca.pipeline import compare_with_ground_truth, format_competition_output, parse_uuid_from_prompt, run_hybrid_from_prompt

app = FastAPI(title="CAMARCA API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/api/rca")
def rca(prompt: str = Query(..., min_length=10), component_granularity: str = "gt-aligned") -> JSONResponse:
    try:
        result = compare_with_ground_truth(run_hybrid_from_prompt(prompt))
        uuid = parse_uuid_from_prompt(prompt) or "window-auto"
        payload = format_competition_output(
            uuid=uuid,
            result=result,
            component_granularity=component_granularity,
        )
        return JSONResponse(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"RCA execution failed: {exc}") from exc


@app.get("/api/stream")
async def stream(prompt: str = Query(..., min_length=10), component_granularity: str = "gt-aligned") -> StreamingResponse:
    async def event_gen() -> AsyncGenerator[str]:
        try:
            yield _sse("status", {"step": "received", "message": "Prompt accepted"})
            await asyncio.sleep(0.1)
            yield _sse("status", {"step": "parse", "message": "Parsing incident window"})
            await asyncio.sleep(0.1)
            yield _sse("status", {"step": "ingest", "message": "Loading logs, traces, metrics"})
            await asyncio.sleep(0.1)
            yield _sse("status", {"step": "analysis", "message": "Running graph and anomaly analysis"})

            result = compare_with_ground_truth(run_hybrid_from_prompt(prompt))

            yield _sse("status", {"step": "fuse", "message": "Fusing modality confidence"})
            uuid = parse_uuid_from_prompt(prompt) or "window-auto"
            output = format_competition_output(
                uuid=uuid,
                result=result,
                component_granularity=component_granularity,
            )
            yield _sse(
                "result",
                {
                    "component": output.get("component"),
                    "reason": output.get("reason"),
                    "payload": output,
                },
            )
            yield _sse("done", {"ok": True})
        except ValueError as exc:
            yield _sse("error", {"detail": str(exc)})
        except Exception as exc:  # pragma: no cover
            yield _sse("error", {"detail": f"RCA execution failed: {exc}"})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
