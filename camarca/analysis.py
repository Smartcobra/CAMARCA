"""
Feature construction, call-graph and causal analysis.

Includes service attribution, log template clustering (lightweight, regex-based),
isolation-forest anomaly scoring on time–service aggregates, and call-graph
propagation of anomaly signals.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

import networkx as nx
import pandas as pd
from sklearn.ensemble import IsolationForest

from camarca.config import (
    BROADCASTER_PENALTY,
    CAUSAL_MAX_TRACE_WEIGHT,
    EDGE_WEIGHT_CALL_FREQ,
    EDGE_WEIGHT_ERROR,
    EDGE_WEIGHT_LATENCY,
    FAN_IN_DAMPEN,
    GATEWAY_OUT_DEGREE_FLOOR,
    HUB_SINK_IN_DEGREE,
    HUB_SINK_LOG_SCALE,
    HYBRID_W_GRAPH_CAUSAL,
    HYBRID_W_LOG_CONF,
    HYBRID_W_METRIC_CONF,
    HYBRID_W_TRACE_CONF,
    IF_CONTAMINATION,
    LEAF_OUT_DEGREE_PENALTY,
    WEIGHT_LOG,
    WEIGHT_METRIC,
    WEIGHT_TRACE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Service level
# ---------------------------------------------------------------------------


def normalize_metric_service_label(service: str) -> str:
    """
    Normalize metric service labels to canonical service names.

    Examples:
    - ``paymentservice-grpc`` -> ``paymentservice``
    - ``frontend-http`` -> ``frontend``
    - ``cartservice-0`` -> ``cartservice`` (pod-style)
    """
    s = str(service).strip().lower()
    # Transport/protocol/runtime suffixes used in service metrics.
    s = re.sub(r"[-_](grpc|http|https|tcp|udp)$", "", s)
    # Pod/index style suffixes (single trailing numeric shard).
    s = re.sub(r"-\d+$", "", s)
    return s


def add_service_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a ``service`` column: prefer existing; else derive from pod-style ids or
    :obj:`k8s_pod` / :obj:`pod` / :obj:`cmdb_id` (e.g. ``cartservice-0`` -> ``cartservice``).
    """
    if df is None or df.empty:
        return df
    if "service" in df.columns:
        return df
    out = df.copy()
    if "k8s_pod" in out.columns:
        out["service"] = out["k8s_pod"].astype(str).str.split("-").str[0]
    elif "pod" in out.columns:
        out["service"] = out["pod"].astype(str).str.split("-").str[0]
    elif "cmdb_id" in out.columns:
        out["service"] = out["cmdb_id"].astype(str).str.split("-").str[0]
    else:
        out["service"] = "unknown"
    return out


def service_level_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (``bucket``, ``service``) with a raw event count. Requires ``service``
    and ``bucket``; see :func:`add_service_column` and time bucketing in ingest.
    """
    if df.empty or "service" not in df.columns or "bucket" not in df.columns:
        return pd.DataFrame(columns=["bucket", "service", "count"])
    return (
        df.groupby(["bucket", "service"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )


# ---------------------------------------------------------------------------
# Traces: parent service resolution + call graph
# ---------------------------------------------------------------------------


def enrich_trace_with_parent_service(trace_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each span, resolve the caller’s service from ``parent_span`` by joining to the
    parent’s ``cmdb_id`` within the same ``trace_id``. Root spans are labeled
    ``__root__`` for the parent.
    """
    d = add_service_column(trace_df)
    if d.empty:
        d["parent_service"] = "unknown"
        return d
    need = {"trace_id", "span_id", "parent_span", "service"}.issubset(d.columns)
    if not need:
        d = d.copy()
        d["parent_service"] = "unknown"
        return d

    smap = d[["trace_id", "span_id", "service"]].drop_duplicates(
        subset=["trace_id", "span_id"]
    )
    pjoin = d.merge(
        smap.rename(columns={"span_id": "parent_span", "service": "parent_service"}),
        on=["trace_id", "parent_span"],
        how="left",
    )
    pspan = pjoin["parent_span"]
    is_root = pspan.isna() | (pspan.astype(str) == "")
    pjoin["parent_service"] = pjoin["parent_service"].where(
        ~is_root, other="__root__"
    )
    pjoin["parent_service"] = pjoin["parent_service"].fillna("unknown")
    return pjoin


def build_trace_graph(trace_df: pd.DataFrame) -> nx.DiGraph:
    """
    Aggregate directed edges (parent service -> child service) with a ``weight`` equal
    to span call count. Uses :func:`enrich_trace_with_parent_service` when needed.
    """
    g = nx.DiGraph()
    if trace_df is None or trace_df.empty:
        return g
    t = (
        trace_df
        if "parent_service" in trace_df.columns
        else enrich_trace_with_parent_service(trace_df)
    )
    if t.empty or "parent_service" not in t.columns or "service" not in t.columns:
        return g
    pair = t[["parent_service", "service"]].dropna()
    pair = pair[pair["service"].astype(str).str.len() > 0]
    if pair.empty:
        return g
    counts = (
        pair.groupby(["parent_service", "service"], as_index=False)
        .size()
        .rename(columns={"size": "weight"})
    )
    for _, r in counts.iterrows():
        p, c, w = r["parent_service"], r["service"], int(r["weight"])
        if c == "unknown":
            continue
        if p in ("", "unknown", "__root__") or pd.isna(p):
            p = "__root__"
        if g.has_edge(p, c):
            g[p][c]["weight"] = g[p][c].get("weight", 0) + w
        else:
            g.add_edge(p, c, weight=w)
    return g


# ---------------------------------------------------------------------------
# Log templates (lightweight, regex; optional Drain3 elsewhere)
# ---------------------------------------------------------------------------


def extract_template(message: str | None) -> str:
    """
    Coarse log template: digits -> ``<NUM>``; long hex/uuid-like tokens -> ``<ID>``.
    Suitable for grouping recurring messages without a full parser.
    """
    text = str(message) if message is not None and not (isinstance(message, float) and pd.isna(message)) else ""
    out = re.sub(r"\d+", "<NUM>", text)
    return re.sub(r"[a-fA-F0-9\-]{6,}", "<ID>", out)


def _log_text_column(log_df: pd.DataFrame) -> str:
    if "message" in log_df.columns:
        return "message"
    if "value" in log_df.columns:
        return "value"
    raise KeyError("Logs need a `message` or `value` text column for templates.")


def apply_log_templates(log_df: pd.DataFrame) -> pd.DataFrame:
    """Group by ``bucket``, ``service``, and normalized ``template``; count rows per group."""
    if log_df is None or log_df.empty:
        return pd.DataFrame(
            columns=["bucket", "service", "template", "count"]
        )
    tcol = _log_text_column(log_df)
    s = add_service_column(log_df) if "service" not in log_df.columns else log_df.copy()
    s["template"] = s[tcol].map(extract_template)
    if "bucket" not in s.columns:
        logger.warning("Log template aggregation skipped: missing `bucket`.")
        return pd.DataFrame(
            columns=["bucket", "service", "template", "count"]
        )
    return (
        s.groupby(["bucket", "service", "template"], dropna=False, sort=False)  # type: ignore[call-overload]
        .size()
        .reset_index(name="count")
    )


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------


def metric_feature_columns(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    numeric = [c for c in df.select_dtypes(include="number").columns if c != "bucket"]
    preferred = [c for c in ("rr", "sr", "mrt", "value", "count") if c in df.columns and c in numeric]
    if preferred:
        return preferred
    return [c for c in numeric if c not in ("timestamp_ns",) and c != "bucket"]


def detect_anomalies(
    df: pd.DataFrame,
    feature_cols: str | Sequence[str] | None = None,
    *,
    contamination: float = IF_CONTAMINATION,
) -> pd.DataFrame:
    """
    Fit an Isolation Forest on ``feature_cols``; add ``anomaly_score`` (sklearn: -1/1) and
    boolean ``is_anomaly``. Fails open (no anomalies) on tiny or wide-empty inputs.
    """
    out = df.copy()
    if out.empty:
        out["anomaly_score"] = 1
        out["is_anomaly"] = False
        return out
    if feature_cols is None:
        cols: list[str] = metric_feature_columns(out)
    elif isinstance(feature_cols, str):
        cols = [feature_cols] if feature_cols in out.columns else []
    else:
        cols = [c for c in feature_cols if c in out.columns]
    if not cols:
        out["anomaly_score"] = 1
        out["is_anomaly"] = False
        logger.info("Anomaly step skipped: no numeric feature columns.")
        return out
    X = out[cols].fillna(0)
    n, d = X.shape[0], X.shape[1]
    if n < 2 or d < 1:
        out["anomaly_score"] = 1
        out["is_anomaly"] = False
        return out
    # Downscale contamination for very small n
    eff = min(float(contamination), 0.5) if n >= 10 else 0.1
    try:
        model = IsolationForest(
            contamination=eff,
            random_state=42,
            n_estimators=100,
            n_jobs=-1,
        )
        out["anomaly_score"] = model.fit_predict(X)
    except (ValueError, TypeError) as e:
        logger.warning("IsolationForest failed, marking all normal: %s", e)
        out["anomaly_score"] = 1
        out["is_anomaly"] = False
        return out
    out["is_anomaly"] = out["anomaly_score"] == -1
    return out


def prepare_metrics_for_anomaly(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (``bucket``, ``service``) with mean of numeric signal columns, or
    a ``row_count`` total when no numeric features exist. Stabilizes Isolation
    Forest for large raw time series.
    """
    if metrics.empty or "bucket" not in metrics.columns:
        return pd.DataFrame()
    m = add_service_column(metrics)
    if "service" in m.columns:
        m = m.copy()
        m["service"] = m["service"].astype(str).map(normalize_metric_service_label)
    gcols: list[str] = [c for c in ("bucket", "service") if c in m.columns]
    if not gcols:
        return pd.DataFrame()
    cols = metric_feature_columns(m)
    if not cols:
        n = m.groupby(gcols, as_index=False, sort=False).size().reset_index(  # type: ignore[call-overload]
            name="row_count"
        )
        return n
    return m.groupby(gcols, as_index=False, sort=False)[cols].mean()


# ---------------------------------------------------------------------------
# Causal graph + root-cause rank
# ---------------------------------------------------------------------------


def build_causal_graph(
    trace_graph: nx.DiGraph,
    metrics_df: pd.DataFrame,
    log_templates_df: pd.DataFrame,
) -> nx.DiGraph:
    """
    Augment the trace call graph with a ``causal_score`` on each edge using
    (trace weight, per-service log template mass, and per-service anomalous bucket count).
    """
    g: nx.DiGraph = trace_graph.copy() if len(trace_graph) else nx.DiGraph()
    if not len(g) or g.number_of_nodes() == 0:
        return g

    if (
        not metrics_df.empty
        and "service" in metrics_df.columns
        and "is_anomaly" in metrics_df.columns
    ):
        anom = metrics_df[metrics_df["is_anomaly"]]
        metric_count_by_service = anom.groupby("service").size()
    else:
        metric_count_by_service = pd.Series(dtype="int64")
    if not log_templates_df.empty and "service" in log_templates_df.columns and "count" in log_templates_df.columns:
        log_count_by_service = log_templates_df.groupby("service")["count"].sum()
    else:
        log_count_by_service = pd.Series(dtype="float64")

    cap = float(CAUSAL_MAX_TRACE_WEIGHT) or 1.0
    for p, c, _data in list(g.edges(data=True)):
        w = float(g[p][c].get("weight", 1) or 1)
        w = min(w, cap) / cap
        mscore = float(
            metric_count_by_service.get(p, 0) + metric_count_by_service.get(c, 0)
        )
        lscore = float(log_count_by_service.get(p, 0) + log_count_by_service.get(c, 0))
        g[p][c]["causal_score"] = (
            WEIGHT_TRACE * w
            + WEIGHT_METRIC * mscore
            + WEIGHT_LOG * lscore
        )
    return g


def rank_root_causes(g: nx.DiGraph) -> list[tuple[str, float]]:
    """
    For each service node, sum ``causal_score`` on *outgoing* edges (influence
    on downstreams). Return nodes sorted by descending total score.
    """
    if g.number_of_nodes() == 0:
        return []
    scores: dict[str, float] = {}
    for node in g.nodes:
        total = 0.0
        for _u, _v, data in g.out_edges(node, data=True):
            total += float(data.get("causal_score", 0.0) or 0.0)
        scores[str(node)] = total
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Merged per-bucket table (log / trace / metrics)
# ---------------------------------------------------------------------------


def extract_log_features(log_df: pd.DataFrame) -> pd.DataFrame:
    if log_df is None or log_df.empty or "bucket" not in log_df.columns:
        return pd.DataFrame(columns=["bucket", "log_error_count"])
    text_col = "message" if "message" in log_df.columns else "value"
    t = log_df["message"] if text_col == "message" else log_df["value"]
    x = log_df.copy()
    x["is_error"] = t.str.contains("error|ERROR", na=False, regex=True)
    return (
        x.groupby("bucket", as_index=False)["is_error"]
        .sum()
        .rename(columns={"is_error": "log_error_count"})
    )


def extract_trace_features(trace_df: pd.DataFrame) -> pd.DataFrame:
    if (
        trace_df is None
        or trace_df.empty
        or "duration" not in trace_df.columns
        or "bucket" not in trace_df.columns
    ):
        return pd.DataFrame(
            columns=["bucket", "trace_avg_duration", "trace_max_duration"]
        )
    agg = trace_df.groupby("bucket")["duration"].agg(["mean", "max"])
    return agg.reset_index().rename(
        columns={"mean": "trace_avg_duration", "max": "trace_max_duration"}
    )


def extract_metric_features(metric_df: pd.DataFrame) -> pd.DataFrame:
    if metric_df is None or metric_df.empty or "bucket" not in metric_df.columns:
        return pd.DataFrame(columns=["bucket"])
    numeric_cols = [
        c
        for c in metric_df.select_dtypes(include="number").columns
        if c not in ("bucket", "timestamp_ns")
    ]
    if not numeric_cols:
        return pd.DataFrame(columns=["bucket"])
    return (
        metric_df.groupby("bucket", as_index=False)[numeric_cols]
        .mean()
    )


def merge_features(
    log_f: pd.DataFrame,
    trace_f: pd.DataFrame,
    metric_f: pd.DataFrame,
) -> pd.DataFrame:
    m = log_f.merge(trace_f, on="bucket", how="outer")
    m = m.merge(metric_f, on="bucket", how="outer")
    return m.fillna(0)


def _normalize_series(s: pd.Series) -> pd.Series:
    if s.empty:
        return s
    vmax = float(s.max())
    if vmax <= 0:
        return s * 0.0
    return s / vmax


def _symmetric_ratio(a: float, b: float, eps: float = 1e-9) -> float:
    return abs(a - b) / (((a + b) / 2.0) + eps)


def _service_baseline_delta(
    traces_enriched: pd.DataFrame,
    metrics_scored: pd.DataFrame,
    baseline_traces_enriched: pd.DataFrame | None = None,
    baseline_metrics_scored: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Estimate per-service fault-vs-reference deviation when explicit normal windows
    are unavailable in the current pipeline call path.

    Reference is built from cross-service medians/quantiles in the same window.
    This is a weaker proxy than true normal windows but still provides a robust
    baseline-delta signal for fusion.
    """
    values: dict[str, float] = {}

    # Metrics: prefer explicit normal baseline windows when available.
    if metrics_scored is not None and not metrics_scored.empty and "service" in metrics_scored.columns:
        m = metrics_scored.copy()
        m["service"] = m["service"].astype(str).map(normalize_metric_service_label)
        metric_cols = [c for c in ("rr", "sr", "mrt", "value", "count", "row_count") if c in m.columns]
        if metric_cols:
            bm = baseline_metrics_scored.copy() if baseline_metrics_scored is not None else pd.DataFrame()
            if not bm.empty and "service" in bm.columns:
                bm = bm.copy()
                bm["service"] = bm["service"].astype(str).map(normalize_metric_service_label)
            for col in metric_cols:
                svc_fault_med = (
                    pd.to_numeric(m[col], errors="coerce")
                    .groupby(m["service"])
                    .median()
                    .dropna()
                )
                if not bm.empty and col in bm.columns:
                    svc_base_med = (
                        pd.to_numeric(bm[col], errors="coerce")
                        .groupby(bm["service"])
                        .median()
                        .dropna()
                    )
                    for svc, fv in svc_fault_med.items():
                        bv = float(svc_base_med.get(svc, 0.0))
                        if bv <= 0 and float(fv) <= 0:
                            continue
                        sr = _symmetric_ratio(float(fv), float(bv))
                        if sr >= 0.05:
                            values[str(svc)] = values.get(str(svc), 0.0) + min(sr, 5.0)
                else:
                    g_med = float(pd.to_numeric(m[col], errors="coerce").median(skipna=True) or 0.0)
                    if g_med <= 0:
                        continue
                    for svc, v in svc_fault_med.items():
                        sr = _symmetric_ratio(float(v), g_med)
                        if sr >= 0.05:
                            values[str(svc)] = values.get(str(svc), 0.0) + min(sr, 5.0)

    # Traces: prefer explicit normal baseline windows when available.
    if (
        traces_enriched is not None
        and not traces_enriched.empty
        and {"service", "duration"}.issubset(traces_enriched.columns)
    ):
        t = traces_enriched.copy()
        t["service"] = t["service"].astype(str)
        svc_fault_med = (
            pd.to_numeric(t["duration"], errors="coerce")
            .groupby(t["service"])
            .median()
            .dropna()
        )
        bt = baseline_traces_enriched.copy() if baseline_traces_enriched is not None else pd.DataFrame()
        if not bt.empty and {"service", "duration"}.issubset(bt.columns):
            bt = bt.copy()
            bt["service"] = bt["service"].astype(str)
            svc_base_med = (
                pd.to_numeric(bt["duration"], errors="coerce")
                .groupby(bt["service"])
                .median()
                .dropna()
            )
            for svc, fv in svc_fault_med.items():
                bv = float(svc_base_med.get(svc, 0.0))
                if bv <= 0 and float(fv) <= 0:
                    continue
                sr = _symmetric_ratio(float(fv), float(bv))
                if sr >= 0.05:
                    values[str(svc)] = values.get(str(svc), 0.0) + min(sr, 5.0)
        else:
            g_med = float(pd.to_numeric(t["duration"], errors="coerce").median(skipna=True) or 0.0)
            if g_med > 0:
                for svc, v in svc_fault_med.items():
                    sr = _symmetric_ratio(float(v), g_med)
                    if sr >= 0.05:
                        values[str(svc)] = values.get(str(svc), 0.0) + min(sr, 5.0)

    if not values:
        return pd.Series(dtype="float64")
    s = pd.Series(values, dtype="float64")
    return _normalize_series(s)


def build_dynamic_service_dependency_graph(
    traces_enriched: pd.DataFrame,
    log_templates: pd.DataFrame,
    metrics_scored: pd.DataFrame,
) -> nx.DiGraph:
    """
    Build a dynamic service dependency graph with edge weights:
    - latency anomaly
    - error propagation
    - call frequency
    """
    g = nx.DiGraph()
    if traces_enriched is None or traces_enriched.empty:
        return g
    t = traces_enriched.copy()
    if "parent_service" not in t.columns:
        t = enrich_trace_with_parent_service(t)
    if "service" not in t.columns or "parent_service" not in t.columns:
        return g

    pair = t[["parent_service", "service"]].copy()
    pair = pair.dropna()
    pair = pair[(pair["service"] != "") & (pair["service"] != "unknown")]
    if pair.empty:
        return g

    # call frequency
    freq = (
        pair.groupby(["parent_service", "service"], as_index=False)
        .size()
        .rename(columns={"size": "call_frequency"})
    )
    freq["call_frequency_n"] = _normalize_series(freq["call_frequency"])

    # latency anomaly by edge (relative duration pressure)
    if "duration" in t.columns:
        d = t[["parent_service", "service", "duration"]].dropna()
        dur = (
            d.groupby(["parent_service", "service"], as_index=False)["duration"]
            .mean()
            .rename(columns={"duration": "edge_avg_duration"})
        )
        global_med = float(d["duration"].median()) if not d.empty else 0.0
        if global_med > 0:
            dur["latency_anomaly"] = (dur["edge_avg_duration"] / global_med).clip(lower=0)
        else:
            dur["latency_anomaly"] = 0.0
        dur["latency_anomaly_n"] = _normalize_series(dur["latency_anomaly"])
    else:
        dur = freq[["parent_service", "service"]].copy()
        dur["latency_anomaly_n"] = 0.0

    # error propagation from logs + metric anomalies
    if (
        log_templates is not None
        and not log_templates.empty
        and "service" in log_templates.columns
        and "count" in log_templates.columns
    ):
        log_err = (
            log_templates.groupby("service")["count"].sum().rename("log_error_mass")
        )
        log_err_n = _normalize_series(log_err)
    else:
        log_err_n = pd.Series(dtype="float64")

    if (
        metrics_scored is not None
        and not metrics_scored.empty
        and "service" in metrics_scored.columns
        and "is_anomaly" in metrics_scored.columns
    ):
        met = metrics_scored[metrics_scored["is_anomaly"]].groupby("service").size()
        met_n = _normalize_series(met.astype("float64"))
    else:
        met_n = pd.Series(dtype="float64")

    # Explicit status-code anomalies from traces (when available) provide
    # high-precision fault evidence and should influence edge propagation.
    if "status_code" in t.columns:
        t_status = t.copy()
        t_status["status_code_num"] = pd.to_numeric(t_status["status_code"], errors="coerce").fillna(0)
        t_status["status_is_anomaly"] = (t_status["status_code_num"] != 0).astype(int)
        tr_status = t_status.groupby("service")["status_is_anomaly"].sum().astype("float64")
        tr_status_n = _normalize_series(tr_status)
    else:
        tr_status_n = pd.Series(dtype="float64")

    e = freq.merge(dur[["parent_service", "service", "latency_anomaly_n"]], on=["parent_service", "service"], how="left")
    e["latency_anomaly_n"] = e["latency_anomaly_n"].fillna(0.0)
    e["error_propagation_n"] = e.apply(
        lambda r: float(log_err_n.get(r["parent_service"], 0.0))
        + float(log_err_n.get(r["service"], 0.0))
        + float(met_n.get(r["parent_service"], 0.0))
        + float(met_n.get(r["service"], 0.0))
        + float(tr_status_n.get(r["parent_service"], 0.0))
        + float(tr_status_n.get(r["service"], 0.0)),
        axis=1,
    )
    e["error_propagation_n"] = _normalize_series(e["error_propagation_n"])

    e["graph_causality_score"] = (
        EDGE_WEIGHT_LATENCY * e["latency_anomaly_n"]
        + EDGE_WEIGHT_ERROR * e["error_propagation_n"]
        + EDGE_WEIGHT_CALL_FREQ * e["call_frequency_n"]
    )

    for _, r in e.iterrows():
        p = r["parent_service"]
        c = r["service"]
        if p in ("", "unknown", "__root__") or pd.isna(p):
            p = "__root__"
        g.add_edge(
            p,
            c,
            call_frequency=float(r["call_frequency"]),
            call_frequency_n=float(r["call_frequency_n"]),
            latency_anomaly=float(r["latency_anomaly_n"]),
            error_propagation=float(r["error_propagation_n"]),
            graph_causality_score=float(r["graph_causality_score"]),
        )
    return g


def _trace_modality_strength(traces_enriched: pd.DataFrame) -> float:
    """0–1 strength from duration and status; calibrated so healthy windows sit mid-range."""
    if traces_enriched is None or traces_enriched.empty:
        return 0.0
    parts: list[float] = []
    if "duration" in traces_enriched.columns:
        d = pd.to_numeric(traces_enriched["duration"], errors="coerce")
        med = float(d.median()) or 1.0
        # 1.5x median: softer than 2x, scaled so typical stress maps above ~0.2 raw
        frac = float((d > (1.5 * med)).mean())
        parts.append(min(1.0, frac * 2.0))
    if "status_code" in traces_enriched.columns:
        st = pd.to_numeric(traces_enriched["status_code"], errors="coerce").fillna(0)
        parts.append(min(1.0, float((st != 0).mean()) * 1.8))
    return float(sum(parts) / max(len(parts), 1)) if parts else 0.0


def _metric_modality_strength(metrics_scored: pd.DataFrame) -> float:
    """
    0–1 strength: IF row anomaly rate is ~5% so we scale it; also use share of
    services with at least one anomalous bucket.
    """
    if metrics_scored is None or metrics_scored.empty or "is_anomaly" not in metrics_scored.columns:
        return 0.0
    s = metrics_scored["is_anomaly"].fillna(False).astype(bool)
    row_r = float(s.mean())
    row_term = min(1.0, row_r * 10.0)
    svc_term = 0.0
    if "service" in metrics_scored.columns:
        nsvc = int(metrics_scored["service"].nunique())
        if nsvc > 0:
            n_anom_svc = int(metrics_scored.loc[s, "service"].nunique())
            svc_term = n_anom_svc / nsvc
    return float(min(1.0, 0.4 * row_term + 0.6 * svc_term))


def _log_modality_strength(log_templates: pd.DataFrame) -> float:
    """0–1 template concentration; avoids huge values from the old q95*len/total rule."""
    if log_templates is None or log_templates.empty or "count" not in log_templates.columns:
        return 0.0
    total = float(log_templates["count"].sum()) or 1.0
    if "template" in log_templates.columns:
        top = log_templates.groupby("template", dropna=False)["count"].sum().sort_values(ascending=False)
        top_share = float(top.head(3).sum()) / total
    else:
        top_share = 1.0
    return float(min(1.0, top_share))


def modality_confidences(
    traces_enriched: pd.DataFrame,
    log_templates: pd.DataFrame,
    metrics_scored: pd.DataFrame,
) -> dict[str, float]:
    """
    Calibrated confidences in ~[0.2, 0.9] for trace/metric with typical fault windows
    near 0.5; log capped so it does not dominate trace/metric.
    """
    t_strength = _trace_modality_strength(traces_enriched)
    m_strength = _metric_modality_strength(metrics_scored)
    l_strength = _log_modality_strength(log_templates)
    # Map 0..1 strength -> confidence centered near 0.5 for mid-range strength
    trace_conf = 0.3 + 0.6 * t_strength
    metric_conf = 0.3 + 0.6 * m_strength
    # Log: cap at 0.5 so trace/metric get fair relative weight
    log_conf = min(0.5, 0.15 + 0.4 * l_strength)
    return {
        "trace_confidence": float(min(0.95, max(0.1, trace_conf))),
        "metric_confidence": float(min(0.95, max(0.1, metric_conf))),
        "log_confidence": float(min(0.5, max(0.1, log_conf))),
    }


def residual_confidence_fusion(
    dynamic_graph: nx.DiGraph,
    traces_enriched: pd.DataFrame,
    log_templates: pd.DataFrame,
    metrics_scored: pd.DataFrame,
    *,
    fusion_weights: dict[str, float] | None = None,
    modality_boosts: dict[str, float] | None = None,
    baseline_traces_enriched: pd.DataFrame | None = None,
    baseline_metrics_scored: pd.DataFrame | None = None,
) -> list[tuple[str, float]]:
    """
    Improved RC-LLM style fusion:
    Final Score =
      w1 * trace_confidence +
      w2 * metric_confidence +
      w3 * log_confidence +
      w4 * graph_causality_score
    """
    conf = modality_confidences(traces_enriched, log_templates, metrics_scored)
    w_trace = float((fusion_weights or {}).get("trace", HYBRID_W_TRACE_CONF))
    w_metric = float((fusion_weights or {}).get("metric", HYBRID_W_METRIC_CONF))
    w_log = float((fusion_weights or {}).get("log", HYBRID_W_LOG_CONF))
    w_graph = float((fusion_weights or {}).get("graph", HYBRID_W_GRAPH_CAUSAL))
    b_trace = float((modality_boosts or {}).get("trace", 1.0))
    b_metric = float((modality_boosts or {}).get("metric", 1.0))
    b_log = float((modality_boosts or {}).get("log", 1.0))
    b_graph = float((modality_boosts or {}).get("graph", 1.0))
    # per-service modality signals
    services = set(dynamic_graph.nodes)
    services.discard("__root__")

    trace_node = pd.Series(dtype="float64")
    if traces_enriched is not None and not traces_enriched.empty and "service" in traces_enriched.columns and "duration" in traces_enriched.columns:
        tr = traces_enriched.groupby("service")["duration"].mean()
        trace_node = _normalize_series(tr.astype("float64"))
    if traces_enriched is not None and not traces_enriched.empty and "service" in traces_enriched.columns and "status_code" in traces_enriched.columns:
        tr_s = traces_enriched.copy()
        tr_s["status_code_num"] = pd.to_numeric(tr_s["status_code"], errors="coerce").fillna(0)
        tr_s["status_is_anomaly"] = (tr_s["status_code_num"] != 0).astype(int)
        trace_status = tr_s.groupby("service")["status_is_anomaly"].sum().astype("float64")
        trace_status = _normalize_series(trace_status)
        # Blend duration-based and status-based trace service signals.
        if trace_node.empty:
            trace_node = trace_status
        else:
            trace_node = 0.7 * trace_node.add(trace_status, fill_value=0.0) + 0.3 * trace_status

    log_node = pd.Series(dtype="float64")
    if log_templates is not None and not log_templates.empty and "service" in log_templates.columns and "count" in log_templates.columns:
        lg = log_templates.groupby("service")["count"].sum()
        log_node = _normalize_series(lg.astype("float64"))

    metric_node = pd.Series(dtype="float64")
    if metrics_scored is not None and not metrics_scored.empty and "service" in metrics_scored.columns and "is_anomaly" in metrics_scored.columns:
        mm = metrics_scored[metrics_scored["is_anomaly"]].groupby("service").size().astype("float64")
        metric_node = _normalize_series(mm)

    # Baseline delta signal (symmetric-ratio style): service deviation from
    # window reference distributions in traces/metrics.
    baseline_delta_node = _service_baseline_delta(
        traces_enriched,
        metrics_scored,
        baseline_traces_enriched=baseline_traces_enriched,
        baseline_metrics_scored=baseline_metrics_scored,
    )

    # Raw sums for edge pressure; fan-in dampening is applied when building graph_node
    # so many parallel callers (catalog) do not always beat a strong single path (cart).
    graph_node: dict[str, float] = {}
    in_sum_map: dict[str, float] = {}
    out_sum_map: dict[str, float] = {}
    in_degree_in: dict[str, int] = {}
    for s in services:
        in_w = [
            float(d.get("graph_causality_score", 0.0)) for _, _, d in dynamic_graph.in_edges(s, data=True)
        ]
        out_w = [
            float(d.get("graph_causality_score", 0.0)) for _, _, d in dynamic_graph.out_edges(s, data=True)
        ]
        in_sum_map[s] = sum(in_w)
        out_sum_map[s] = sum(out_w)
        in_degree_in[s] = len(in_w)
    out_max = max(out_sum_map.values()) if out_sum_map else 1.0

    # Trace-first anchor: high incoming edge pressure on a *mostly* downstream service.
    # Pure 1/(1+out_degree) over-penalizes one hop (e.g. cart -> checkout) vs a catalog sink
    # with out_degree 0; use LEAF_OUT_DEGREE_PENALTY < 1 to soften.
    od_factor = float(LEAF_OUT_DEGREE_PENALTY)
    # Dampen total *incoming* mass when many parents point at the same node (reduces
    # catalog-style fan-in advantage vs mid-tier services with fewer dependencies).
    fan_in_dampen = float(FAN_IN_DAMPEN)
    in_damped: dict[str, float] = {}
    for s in services:
        raw_in = float(in_sum_map.get(s, 0.0))
        kin = max(0, int(in_degree_in.get(s, 0)) - 1)
        in_damped[s] = raw_in / (1.0 + fan_in_dampen * float(kin))
    in_dmax = max(in_damped.values()) if in_damped else 1.0
    for s in services:
        incoming_n = float(in_damped.get(s, 0.0)) / (in_dmax or 1.0)
        outgoing_n = float(out_sum_map.get(s, 0.0)) / (out_max or 1.0)
        od = float(dynamic_graph.out_degree(s))
        leaf_anchor = incoming_n * (1.0 / (1.0 + od_factor * od))
        graph_node[s] = 0.75 * leaf_anchor + 0.25 * outgoing_n

    # Hub + sink: many parents, no callees in this graph -> log count can over-represent
    # duplicated upstream errors (e.g. catalog) vs 1-out-edge services (e.g. cart -> …).
    hub_in = int(HUB_SINK_IN_DEGREE)
    hub_log = float(HUB_SINK_LOG_SCALE)
    log_scale: dict[str, float] = {
        s: (hub_log if in_degree_in.get(s, 0) >= hub_in and dynamic_graph.out_degree(s) == 0 else 1.0)
        for s in services
    }

    br = float(BROADCASTER_PENALTY)
    fused: list[tuple[str, float]] = []
    for s in services:
        score = (
            w_trace * b_trace * conf["trace_confidence"] * float(trace_node.get(s, 0.0))
            + w_metric * b_metric * conf["metric_confidence"] * float(metric_node.get(s, 0.0))
            + w_log
            * b_log
            * conf["log_confidence"]
            * float(log_node.get(s, 0.0))
            * float(log_scale.get(s, 1.0))
            + w_graph * b_graph * float(graph_node.get(s, 0.0))
            + 0.18 * float(baseline_delta_node.get(s, 0.0))
        )
        od = int(dynamic_graph.out_degree(s))
        floor = int(GATEWAY_OUT_DEGREE_FLOOR)
        ex = max(0, od - floor)
        if ex > 0:
            score *= 1.0 / (1.0 + br * float(ex))
        fused.append((str(s), float(score)))
    return sorted(fused, key=lambda x: x[1], reverse=True)
