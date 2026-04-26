"""Streamlit UI for CAMARCA FastAPI SSE endpoint."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse
import urllib.parse
import urllib.request
from typing import Any

import streamlit as st
import streamlit.components.v1 as components


DEFAULT_PROMPT = (
    "A fault occurred from 2022-05-09T06:40:17Z to 2022-05-09T06:50:17Z. "
    "Please identify the root cause."
)


def _read_sse(url: str):
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
        },
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        event = "message"
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
                continue
            if line.startswith("data:"):
                payload = line.split(":", 1)[1].strip()
                yield event, json.loads(payload)


def _normalize_api_base(raw_url: str) -> str:
    text = raw_url.strip()
    if not text:
        return "http://127.0.0.1:8000"
    if "://" not in text:
        text = f"http://{text}"
    parsed = urlparse(text)
    host = parsed.netloc or parsed.path
    return f"{parsed.scheme}://{host}".rstrip("/")


def _health_check(api_base: str) -> tuple[bool, str]:
    health_url = f"{api_base}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=8) as response:
            if response.status == 200:
                return True, health_url
            return False, f"{health_url} returned status {response.status}"
    except Exception as exc:
        return False, f"{health_url} failed: {exc}"


def _verify_stream_route(api_base: str) -> tuple[bool, str]:
    openapi_url = f"{api_base}/openapi.json"
    try:
        with urllib.request.urlopen(openapi_url, timeout=8) as response:
            if response.status != 200:
                return False, f"{openapi_url} returned status {response.status}"
            raw = response.read().decode("utf-8")
            doc = json.loads(raw)
            title = doc.get("info", {}).get("title", "Unknown API")
            paths = doc.get("paths", {})
            if "/api/stream" not in paths:
                return False, f"Connected API is '{title}', but '/api/stream' is missing"
            return True, f"Connected to '{title}'"
    except Exception as exc:
        return False, f"{openapi_url} failed: {exc}"


def _ui_style() -> None:
    st.markdown(
        """
        <style>
        .main-card {
            background: linear-gradient(135deg, #0e172f 0%, #1a2750 100%);
            padding: 1rem 1.25rem;
            border-radius: 14px;
            border: 1px solid #2f4888;
            margin-bottom: 1rem;
        }
        .title-big {
            font-size: 1.5rem;
            font-weight: 700;
            color: #e9f0ff;
            margin-bottom: 0.25rem;
        }
        .subtitle {
            color: #a9bddf;
            font-size: 0.95rem;
        }
        .result-card {
            background: linear-gradient(135deg, #0b1732 0%, #1d2f66 100%);
            border: 1px solid #3f5eb0;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.8rem;
        }
        .result-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #f2f7ff;
            margin-bottom: 0.3rem;
        }
        .result-chip {
            display: inline-block;
            background: #0f244f;
            border: 1px solid #4e72d0;
            color: #dce8ff;
            border-radius: 999px;
            padding: 0.2rem 0.6rem;
            margin-right: 0.35rem;
            margin-bottom: 0.3rem;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .reason-text {
            font-size: 1rem;
            color: #dce7ff;
            line-height: 1.55;
            margin-top: 0.5rem;
        }
        .reason-main {
            margin-top: 0.75rem;
            background: linear-gradient(90deg, #7a1239 0%, #b31e5a 50%, #8e1b7a 100%);
            border: 1px solid #ff89bf;
            color: #ffffff;
            border-radius: 12px;
            padding: 0.8rem 0.9rem;
            font-size: 1.08rem;
            font-weight: 800;
            letter-spacing: 0.2px;
            box-shadow: 0 0 14px rgba(247, 37, 133, 0.45);
        }
        .trace-card {
            background: linear-gradient(135deg, #0d1a39 0%, #1c2f63 100%);
            border: 1px solid #4065bf;
            border-radius: 12px;
            padding: 0.75rem 0.85rem;
            margin-bottom: 0.55rem;
        }
        .trace-head {
            color: #f3f7ff;
            font-weight: 700;
            margin-bottom: 0.45rem;
            font-size: 0.98rem;
        }
        .trace-action {
            background: #111f43;
            border: 1px solid #4f72ca;
            color: #d8e5ff;
            border-radius: 10px;
            padding: 0.45rem 0.6rem;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.9rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
            word-break: break-word;
        }
        .trace-observation {
            color: #dce7ff;
            line-height: 1.5;
            font-size: 0.93rem;
        }
        .arch-btn-wrap {
            margin-top: 0.35rem;
            margin-bottom: 0.6rem;
        }
        .arch-panel {
            background: linear-gradient(135deg, #0d1b3b 0%, #1c2f63 100%);
            border: 1px solid #4767bc;
            border-radius: 14px;
            padding: 0.75rem 0.9rem;
            margin-bottom: 0.6rem;
            color: #dce8ff;
            font-weight: 600;
        }
        @keyframes pulseGlow {
            0% { box-shadow: 0 0 0 rgba(76, 201, 240, 0.1), 0 0 0 rgba(76, 201, 240, 0.1); }
            50% { box-shadow: 0 0 14px rgba(76, 201, 240, 0.45), 0 0 28px rgba(76, 201, 240, 0.2); }
            100% { box-shadow: 0 0 0 rgba(76, 201, 240, 0.1), 0 0 0 rgba(76, 201, 240, 0.1); }
        }
        .st-key-view_arch_btn button {
            background: linear-gradient(135deg, #123a82 0%, #1e5bd6 100%);
            color: #f2f8ff;
            border: 1px solid #6ab8ff;
            font-weight: 700;
            border-radius: 10px;
            animation: pulseGlow 2.2s ease-in-out infinite;
        }
        .st-key-view_arch_btn button:hover {
            border-color: #93d5ff;
            background: linear-gradient(135deg, #16479e 0%, #2b68e2 100%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.dialog("Raw RCA JSON")
def _show_raw_json_dialog(payload: dict[str, Any]) -> None:
    st.caption("Full backend payload")
    st.code(json.dumps(payload, indent=2), language="json")


@st.dialog("Animated Architecture", width="large")
def _show_architecture_dialog() -> None:
    arch_file = Path("docs/animated_architecture.html")
    if not arch_file.exists():
        st.error("Could not find `docs/animated_architecture.html`.")
        return
    html = arch_file.read_text(encoding="utf-8")
    if "arch_modal_view_mode" not in st.session_state:
        st.session_state.arch_modal_view_mode = "Normal"
    st.markdown(
        """
        <div class="arch-panel">
          Interactive architecture flow
        </div>
        """,
        unsafe_allow_html=True,
    )
    mode = st.radio(
        "View Mode",
        options=["Normal", "Expanded"],
        key="arch_modal_view_mode",
        horizontal=True,
    )
    frame_height = 980 if mode == "Expanded" else 760
    st.caption(f"Mode: {mode}")
    components.html(html, height=frame_height, scrolling=True)


def _render_readable_result(payload: dict[str, Any]) -> None:
    component = payload.get("component", "unknown")
    reason = payload.get("reason", "No reason provided")
    uuid = payload.get("uuid", "unknown")
    steps = payload.get("reasoning_trace", [])

    st.markdown(
        f"""
        <div class="result-card">
          <div class="result-title">Root Cause Analysis Result</div>
          <span class="result-chip">Component: {component}</span>
          <span class="result-chip">UUID: {uuid}</span>
          <div class="reason-main">Reason: {reason}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Reasoning Trace")
    if steps:
        for item in steps:
            step = item.get("step", "?")
            action = item.get("action", "N/A")
            observation = item.get("observation", "N/A")
            st.markdown(
                f"""
                <div class="trace-card">
                  <div class="trace-head">Step {step}</div>
                  <div class="trace-action">Action: {action}</div>
                  <div class="trace-observation"><b>Observation:</b> {observation}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No reasoning trace available in response.")


def _render_evaluation_dashboard() -> None:
    eval_dir = Path("outputs/evaluation")
    summary_file = eval_dir / "summary.json"
    acc_plot = eval_dir / "accuracy_curve.png"
    loss_plot = eval_dir / "loss_curve.png"

    st.subheader("Evaluation Dashboard")
    st.caption("Shows the latest artifacts generated by `evaluation.py`.")

    if not eval_dir.exists():
        st.info("No evaluation artifacts found yet. Run `uv run python evaluation.py` first.")
        return

    if summary_file.exists():
        try:
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            metrics = summary.get("metrics", {})
            c1, c2, c3 = st.columns(3)
            train = metrics.get("train", {})
            val = metrics.get("val", {})
            test = metrics.get("test", {})
            with c1:
                st.metric("Train Top-1", f"{float(train.get('top1_accuracy', 0.0)):.3f}")
                st.metric("Train Loss", f"{float(train.get('avg_loss', 0.0)):.3f}")
            with c2:
                st.metric("Val Top-1", f"{float(val.get('top1_accuracy', 0.0)):.3f}")
                st.metric("Val Loss", f"{float(val.get('avg_loss', 0.0)):.3f}")
            with c3:
                st.metric("Test Top-1", f"{float(test.get('top1_accuracy', 0.0)):.3f}")
                st.metric("Test Loss", f"{float(test.get('avg_loss', 0.0)):.3f}")
        except Exception as exc:
            st.warning(f"Could not read summary metrics: {exc}")
    else:
        st.info("`summary.json` not found in `outputs/evaluation`.")

    p1, p2 = st.columns(2)
    with p1:
        if acc_plot.exists():
            st.image(str(acc_plot), caption="Validation/Test Running Accuracy", use_container_width=True)
        else:
            st.info("`accuracy_curve.png` not found.")
    with p2:
        if loss_plot.exists():
            st.image(str(loss_plot), caption="Validation/Test Running Loss", use_container_width=True)
        else:
            st.info("`loss_curve.png` not found.")


def main() -> None:
    st.set_page_config(page_title="CAMARCA Dashboard", page_icon=":satellite:", layout="wide")
    _ui_style()
    if "final_payload" not in st.session_state:
        st.session_state.final_payload = None

    st.markdown(
        """
        <div class="main-card">
          <div class="title-big">Causal-Aware Multi-Agent RCA (CAMARCA) Dashboard</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        prompt = st.text_area("Incident Prompt", value=DEFAULT_PROMPT, height=120)
    with col2:
        api_base = st.text_input("FastAPI URL", value="http://127.0.0.1:8001")
        granularity = st.selectbox("Component granularity", ["gt-aligned", "predicted"], index=0)
        st.markdown('<div class="arch-btn-wrap"></div>', unsafe_allow_html=True)
        if st.button("View Animated Architecture", key="view_arch_btn", use_container_width=True):
            _show_architecture_dialog()

    run_clicked = st.button("Run RCA (SSE Stream)", type="primary", use_container_width=True)

    status_box = st.empty()
    progress_box = st.progress(0, text="Waiting for run...")
    result_box = st.empty()
    raw_box = st.empty()

    if run_clicked:
        if not prompt.strip():
            st.error("Prompt is required.")
            return

        api_base_norm = _normalize_api_base(api_base)
        ok, health_msg = _health_check(api_base_norm)
        if not ok:
            st.error(f"FastAPI health check failed. {health_msg}")
            st.caption("Tip: run `uv run uvicorn camarca.web_api:app --host 0.0.0.0 --port 8001 --reload`")
            return

        stream_ok, stream_msg = _verify_stream_route(api_base_norm)
        if not stream_ok:
            st.error(f"SSE route check failed. {stream_msg}")
            st.caption("Tip: this URL points to another app. Use the CAMARCA API port (recommended 8001).")
            return
        st.caption(stream_msg)

        steps = ["received", "parse", "ingest", "analysis", "fuse", "done"]
        step_to_progress = {name: int((idx + 1) / len(steps) * 100) for idx, name in enumerate(steps)}
        encoded = urllib.parse.urlencode(
            {
                "prompt": prompt.strip(),
                "component_granularity": granularity,
            }
        )
        url = f"{api_base_norm}/api/stream?{encoded}"

        status_box.info(f"Connecting to backend stream: {url}")
        with st.spinner("Running CAMARCA pipeline..."):
            try:
                final_payload: dict[str, Any] | None = None
                for event, data in _read_sse(url):
                    if event == "status":
                        step = data.get("step", "unknown")
                        message = data.get("message", "Processing")
                        status_box.info(f"{step.upper()}: {message}")
                        progress_box.progress(step_to_progress.get(step, 10), text=message)
                    elif event == "result":
                        final_payload = data.get("payload")
                        status_box.success(
                            f"Candidate root cause: {data.get('component', 'n/a')} | {data.get('reason', 'n/a')}"
                        )
                        progress_box.progress(step_to_progress["fuse"], text="Ranking complete")
                    elif event == "error":
                        detail = data.get("detail", "Unknown backend error")
                        status_box.error(detail)
                        break
                    elif event == "done":
                        progress_box.progress(100, text="Completed")

                if final_payload is not None:
                    st.session_state.final_payload = final_payload
            except Exception as exc:
                status_box.error(f"Could not read SSE stream from {url}: {exc}")

    if st.session_state.final_payload is not None:
        payload = st.session_state.final_payload
        result_box.subheader("RCA Result")
        with result_box.container():
            _render_readable_result(payload)
            if st.button("View Raw JSON", key="view_raw_json_btn", use_container_width=False):
                _show_raw_json_dialog(payload)
        raw_box.download_button(
            "Download RCA JSON",
            data=json.dumps(payload, indent=2),
            file_name="rca_result.json",
            mime="application/json",
        )

    st.divider()
    _render_evaluation_dashboard()


if __name__ == "__main__":
    main()
