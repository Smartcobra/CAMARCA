"""Advanced RCA evaluation with tuning, failure-aware scoring, and node track.

Implements:
1) Fusion weight tuning on train split (service-evaluable cases)
2) Failure-type aware modality boosts
3) Separate node-case track (node-level top1 accuracy)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.mplconfig")))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs/.cache")))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix

from camarca import analysis
from camarca.pipeline import _load_window_data, run_hybrid_window

matplotlib.use("Agg")


@dataclass
class EvalCase:
    case_id: str
    event_ts: pd.Timestamp
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    gt_level: str
    gt_component_raw: str
    gt_component_service: str
    gt_failure_type: str


@dataclass
class CaseArtifacts:
    case: EvalCase
    result: Any
    node_top1_pred: str | None


DEFAULT_WEIGHTS = {"trace": 0.25, "metric": 0.25, "log": 0.20, "graph": 0.30}


def _normalize_service_label(label: str) -> str:
    s = str(label).strip().lower()
    if s.startswith("node-"):
        return s
    s = re.sub(r"(\D)\d+$", r"\1", s)
    return s


def _to_service_label(cmdb_id: str, level: str) -> str:
    raw = str(cmdb_id)
    if str(level).lower() == "pod" and "-" in raw:
        raw = raw.split("-", 1)[0]
    return _normalize_service_label(raw)


def load_cases(gt_csv: Path, before_seconds: int, after_seconds: int) -> list[EvalCase]:
    df = pd.read_csv(gt_csv)
    required = {"timestamp", "level", "cmdb_id", "failure_type"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Ground truth file missing required columns: {sorted(missing)}")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["event_ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    out: list[EvalCase] = []
    for i, row in df.iterrows():
        ts: pd.Timestamp = row["event_ts"]
        gt_raw = str(row["cmdb_id"])
        gt_level = str(row["level"])
        out.append(
            EvalCase(
                case_id=f"case_{i:04d}",
                event_ts=ts,
                start_ts=ts - pd.Timedelta(seconds=before_seconds),
                end_ts=ts + pd.Timedelta(seconds=after_seconds),
                gt_level=gt_level,
                gt_component_raw=gt_raw,
                gt_component_service=_to_service_label(gt_raw, gt_level),
                gt_failure_type=str(row["failure_type"]),
            )
        )
    return out


def split_cases(cases: list[EvalCase], train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, list[EvalCase]]:
    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("train/val/test ratios must sum to 1.0")
    n = len(cases)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("Split produced empty subset. Increase cases or adjust ratios.")
    return {
        "train": cases[:n_train],
        "val": cases[n_train : n_train + n_val],
        "test": cases[n_train + n_val :],
    }


def _is_service_evaluable(case: EvalCase) -> bool:
    return not case.gt_component_service.startswith("node-")


def _failure_type_boosts(failure_type: str) -> dict[str, float]:
    f = str(failure_type).lower()
    boosts = {"trace": 1.0, "metric": 1.0, "log": 1.0, "graph": 1.0}
    if "network" in f or "packet" in f:
        boosts.update({"trace": 1.35, "graph": 1.30, "metric": 0.85, "log": 0.95})
    elif "cpu" in f or "memory" in f or "disk" in f or "i/o" in f:
        boosts.update({"metric": 1.35, "graph": 1.10, "trace": 1.05, "log": 0.90})
    elif "write" in f or "read" in f:
        boosts.update({"metric": 1.20, "log": 1.10, "trace": 0.95, "graph": 1.00})
    return boosts


def _predict_node_top1(start: pd.Timestamp, end: pd.Timestamp) -> str | None:
    logs, traces, metrics = _load_window_data(start, end)
    parts: list[pd.Series] = []
    for frame in (logs, traces, metrics):
        if frame is None or frame.empty:
            continue
        if "cmdb_id" not in frame.columns:
            continue
        ids = frame["cmdb_id"].astype(str).str.lower()
        nodes = ids[ids.str.startswith("node-")]
        if not nodes.empty:
            parts.append(nodes)
    if not parts:
        return None
    all_nodes = pd.concat(parts, ignore_index=True)
    if all_nodes.empty:
        return None
    return str(all_nodes.value_counts().index[0])


def _build_artifacts(cases: list[EvalCase]) -> dict[str, CaseArtifacts]:
    out: dict[str, CaseArtifacts] = {}
    total = len(cases)
    for i, case in enumerate(cases, start=1):
        print(f"[{i}/{total}] building artifacts for {case.case_id} ...")
        result = run_hybrid_window(case.start_ts, case.end_ts)
        node_pred = _predict_node_top1(case.start_ts, case.end_ts) if case.gt_component_service.startswith("node-") else None
        out[case.case_id] = CaseArtifacts(case=case, result=result, node_top1_pred=node_pred)
    return out


def _rank_metrics_from_ranking(ranking: list[tuple[str, float]], truth: str) -> dict[str, float | int]:
    n = max(len(ranking), 1)
    true_rank = 0
    for i, (label, _score) in enumerate(ranking, start=1):
        if _normalize_service_label(label) == truth:
            true_rank = i
            break
    rank_loss = float(true_rank / n) if true_rank > 0 else 1.0
    hit1 = 1 if true_rank == 1 else 0
    hit3 = 1 if 0 < true_rank <= 3 else 0
    hit5 = 1 if 0 < true_rank <= 5 else 0
    mrr = 1.0 / true_rank if true_rank > 0 else 0.0
    ndcg = 1.0 / math.log2(true_rank + 1) if true_rank > 0 else 0.0
    rca_score = 0.4 * hit1 + 0.3 * mrr + 0.3 * ndcg
    return {
        "ranking_size": n,
        "true_rank": true_rank,
        "top1_accuracy": hit1,
        "hit_at_3": hit3,
        "hit_at_5": hit5,
        "mrr": float(mrr),
        "ndcg": float(ndcg),
        "rank_loss": float(rank_loss),
        "rca_score": float(rca_score),
    }


def _trace_anchor_scores(art: CaseArtifacts) -> dict[str, float]:
    traces = art.result.traces_enriched if art.result.traces_enriched is not None else pd.DataFrame()
    if traces.empty or "service" not in traces.columns:
        return {}
    t = traces.copy()
    t["service"] = t["service"].astype(str).map(_normalize_service_label)
    score = pd.Series(0.0, index=pd.Index([], dtype="object"))
    if "duration" in t.columns and len(t) > 4:
        med = float(t["duration"].median()) or 1.0
        t["dur_anom"] = (t["duration"] / med).clip(lower=0.0, upper=10.0)
        dur_s = t.groupby("service")["dur_anom"].mean().astype(float)
        score = score.add(dur_s, fill_value=0.0)
    if "status_code" in t.columns:
        sc = pd.to_numeric(t["status_code"], errors="coerce").fillna(0)
        t["status_anom"] = (sc != 0).astype(int)
        st_s = t.groupby("service")["status_anom"].sum().astype(float)
        score = score.add(st_s, fill_value=0.0)
    return {str(k): float(v) for k, v in score.to_dict().items()}


def _metric_symmetric_ratio_scores(art: CaseArtifacts) -> dict[str, float]:
    m = art.result.metrics_scored if art.result.metrics_scored is not None else pd.DataFrame()
    if m.empty or "service" not in m.columns:
        return {}
    d = m.copy()
    d["service"] = d["service"].astype(str).map(_normalize_service_label)
    if "bucket" not in d.columns:
        return {}
    buckets = sorted([int(x) for x in d["bucket"].dropna().unique().tolist()])
    if len(buckets) < 5:
        return {}
    fault_lo, fault_hi = int(min(buckets)), int(max(buckets))
    prev = d[(d["bucket"] >= (fault_lo - 6)) & (d["bucket"] < (fault_lo - 2))]
    post = d[(d["bucket"] > (fault_hi + 2)) & (d["bucket"] <= (fault_hi + 6))]
    normal = pd.concat([prev, post], ignore_index=True) if (not prev.empty or not post.empty) else pd.DataFrame()
    fault = d[(d["bucket"] >= fault_lo) & (d["bucket"] <= fault_hi)]
    if normal.empty or fault.empty:
        return {}
    metric_cols = [c for c in ("rr", "sr", "mrt", "value", "count", "row_count") if c in d.columns]
    if not metric_cols:
        return {}
    eps = 1e-9
    out: dict[str, float] = {}
    for svc in sorted(set(fault["service"].astype(str))):
        fs = fault[fault["service"] == svc]
        ns = normal[normal["service"] == svc]
        if fs.empty or ns.empty:
            continue
        s = 0.0
        for c in metric_cols:
            f_med = float(pd.to_numeric(fs[c], errors="coerce").median(skipna=True) or 0.0)
            n_med = float(pd.to_numeric(ns[c], errors="coerce").median(skipna=True) or 0.0)
            sr = abs(f_med - n_med) / (((f_med + n_med) / 2.0) + eps)
            if sr >= 0.05:
                s += min(sr, 5.0)
        if s > 0:
            out[svc] = s
    return out


def _log_keyword_scores(art: CaseArtifacts) -> dict[str, float]:
    l = art.result.log_templates if art.result.log_templates is not None else pd.DataFrame()
    if l.empty or "service" not in l.columns or "count" not in l.columns:
        return {}
    d = l.copy()
    d["service"] = d["service"].astype(str).map(_normalize_service_label)
    if "template" in d.columns:
        tpl = d["template"].astype(str).str.lower()
        hit = tpl.str.contains("error|exception|timeout|failed|500|404|connection", regex=True, na=False)
        d = d[hit]
    if d.empty:
        return {}
    s = d.groupby("service")["count"].sum().astype(float)
    return {str(k): float(v) for k, v in s.to_dict().items()}


def _normalized_map(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    vmax = max(values.values()) or 1.0
    return {k: float(v) / float(vmax) for k, v in values.items()}


def _reranker_feature_cols_num() -> list[str]:
    return [
        "rank_pos",
        "fused_score",
        "trace_mean_duration",
        "log_error_mass",
        "metric_anomaly_mass",
        "graph_in_causal",
        "graph_out_causal",
        "out_degree",
        "in_degree",
        "support_modalities",
        "modality_agreement",
        "score_gap_to_top",
        "score_margin_next",
    ]


def _candidate_score_map(art: CaseArtifacts, weights: dict[str, float], use_failure_policy: bool) -> dict[str, float]:
    case = art.case
    boosts = _failure_type_boosts(case.gt_failure_type) if use_failure_policy else {"trace": 1.0, "metric": 1.0, "log": 1.0, "graph": 1.0}
    ranking = analysis.residual_confidence_fusion(
        art.result.dynamic_graph,
        art.result.traces_enriched if art.result.traces_enriched is not None else pd.DataFrame(),
        art.result.log_templates if art.result.log_templates is not None else pd.DataFrame(),
        art.result.metrics_scored if art.result.metrics_scored is not None else pd.DataFrame(),
        fusion_weights=weights,
        modality_boosts=boosts,
        baseline_traces_enriched=(
            art.result.baseline_traces_enriched if art.result.baseline_traces_enriched is not None else pd.DataFrame()
        ),
        baseline_metrics_scored=(
            art.result.baseline_metrics_scored if art.result.baseline_metrics_scored is not None else pd.DataFrame()
        ),
    )
    fused = _normalized_map({_normalize_service_label(lbl): float(score) for lbl, score in ranking})
    trace = _normalized_map(_trace_anchor_scores(art))
    metric_sr = _normalized_map(_metric_symmetric_ratio_scores(art))
    logk = _normalized_map(_log_keyword_scores(art))
    # Adaptive routing: if trace weak, rely more on metric+log.
    trace_strength = float(np.mean(list(trace.values()))) if trace else 0.0
    if trace_strength < 0.12:
        alpha = {"fused": 0.20, "trace": 0.15, "metric": 0.45, "log": 0.20}
    else:
        alpha = {"fused": 0.20, "trace": 0.40, "metric": 0.25, "log": 0.15}
    services = set(fused) | set(trace) | set(metric_sr) | set(logk)
    scores: dict[str, float] = {}
    for s in services:
        scores[s] = (
            alpha["fused"] * float(fused.get(s, 0.0))
            + alpha["trace"] * float(trace.get(s, 0.0))
            + alpha["metric"] * float(metric_sr.get(s, 0.0))
            + alpha["log"] * float(logk.get(s, 0.0))
        )
    # Soft evidence gate: avoid over-penalizing true roots that spike in one strong modality.
    for s in list(scores):
        supports = int(s in trace) + int(s in metric_sr) + int(s in logk)
        support_ratio = supports / 3.0
        scores[s] *= 0.78 + (0.22 * support_ratio)
    return scores


def _score_service_case(art: CaseArtifacts, weights: dict[str, float], use_failure_policy: bool) -> dict[str, Any]:
    case = art.case
    score_map = _candidate_score_map(art, weights=weights, use_failure_policy=use_failure_policy)
    dedup = sorted(score_map.items(), key=lambda x: x[1], reverse=True)

    pred_top1 = dedup[0][0] if dedup else "insufficient_evidence"
    m = _rank_metrics_from_ranking(dedup, case.gt_component_service)
    return {
        "pred_top1_service": pred_top1,
        "pred_top1_score": dedup[0][1] if dedup else 0.0,
        **m,
    }


def _service_feature_frame(
    art: CaseArtifacts,
    *,
    weights: dict[str, float],
    use_failure_policy: bool,
) -> pd.DataFrame:
    case = art.case
    score_map = _candidate_score_map(art, weights=weights, use_failure_policy=use_failure_policy)
    if not score_map:
        return pd.DataFrame()

    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    rank_map = {lbl: (i + 1, float(score)) for i, (lbl, score) in enumerate(ranked)}
    top_score = float(ranked[0][1]) if ranked else 0.0
    next_score_by_service: dict[str, float] = {}
    for i, (svc, score) in enumerate(ranked):
        nxt = float(ranked[i + 1][1]) if i + 1 < len(ranked) else 0.0
        next_score_by_service[str(svc)] = nxt
    # Dedup with first occurrence preserved by rank.
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []

    traces = art.result.traces_enriched if art.result.traces_enriched is not None else pd.DataFrame()
    logs = art.result.log_templates if art.result.log_templates is not None else pd.DataFrame()
    metrics = art.result.metrics_scored if art.result.metrics_scored is not None else pd.DataFrame()
    g = art.result.dynamic_graph

    trace_by_service = pd.Series(dtype="float64")
    if not traces.empty and {"service", "duration"}.issubset(traces.columns):
        trace_by_service = traces.groupby("service")["duration"].mean().astype("float64")
    log_by_service = pd.Series(dtype="float64")
    if not logs.empty and {"service", "count"}.issubset(logs.columns):
        log_by_service = logs.groupby("service")["count"].sum().astype("float64")
    metric_by_service = pd.Series(dtype="float64")
    if not metrics.empty and {"service", "is_anomaly"}.issubset(metrics.columns):
        metric_by_service = metrics[metrics["is_anomaly"]].groupby("service").size().astype("float64")

    for raw_lbl, (rank_pos, fused_score) in rank_map.items():
        if raw_lbl in seen:
            continue
        seen.add(raw_lbl)
        in_edges = list(g.in_edges(raw_lbl, data=True)) if g is not None and raw_lbl in g else []
        out_edges = list(g.out_edges(raw_lbl, data=True)) if g is not None and raw_lbl in g else []
        in_causal = float(sum(float(d.get("graph_causality_score", 0.0)) for _, _, d in in_edges))
        out_causal = float(sum(float(d.get("graph_causality_score", 0.0)) for _, _, d in out_edges))
        support_modalities = int(raw_lbl in trace_by_service.index) + int(raw_lbl in metric_by_service.index) + int(
            raw_lbl in log_by_service.index
        )
        next_score = float(next_score_by_service.get(raw_lbl, 0.0))
        rows.append(
            {
                "case_id": case.case_id,
                "service": raw_lbl,
                "failure_type": str(case.gt_failure_type).lower(),
                "rank_pos": int(rank_pos),
                "fused_score": float(fused_score),
                "trace_mean_duration": float(trace_by_service.get(raw_lbl, 0.0)),
                "log_error_mass": float(log_by_service.get(raw_lbl, 0.0)),
                "metric_anomaly_mass": float(metric_by_service.get(raw_lbl, 0.0)),
                "graph_in_causal": in_causal,
                "graph_out_causal": out_causal,
                "out_degree": int(g.out_degree(raw_lbl)) if g is not None and raw_lbl in g else 0,
                "in_degree": int(g.in_degree(raw_lbl)) if g is not None and raw_lbl in g else 0,
                "support_modalities": support_modalities,
                "modality_agreement": float(support_modalities) / 3.0,
                "score_gap_to_top": max(0.0, top_score - float(fused_score)),
                "score_margin_next": max(0.0, float(fused_score) - next_score),
                "is_true": 1 if raw_lbl == case.gt_component_service else 0,
            }
        )
    return pd.DataFrame(rows)


def _build_reranker_dataset(
    artifacts: dict[str, CaseArtifacts],
    case_ids: set[str],
    *,
    weights: dict[str, float],
    use_failure_policy: bool,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for cid in case_ids:
        art = artifacts[cid]
        if not _is_service_evaluable(art.case):
            continue
        f = _service_feature_frame(art, weights=weights, use_failure_policy=use_failure_policy)
        if not f.empty:
            frames.append(f)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _make_reranker_pipeline() -> Pipeline:
    feature_cols_num = _reranker_feature_cols_num()
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), feature_cols_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["failure_type"]),
        ]
    )
    clf = LogisticRegression(max_iter=1200, class_weight="balanced", solver="lbfgs")
    return Pipeline([("pre", pre), ("clf", clf)])


def _train_reranker_ensemble(train_df: pd.DataFrame, n_splits: int = 4) -> list[Pipeline]:
    if train_df.empty or train_df["is_true"].sum() == 0:
        return []
    feature_cols_num = _reranker_feature_cols_num()
    X = train_df[feature_cols_num + ["failure_type", "case_id"]].copy()
    y = train_df["is_true"].astype(int)
    if y.nunique() < 2:
        return []

    groups = X["case_id"].astype(str)
    X_fit = X.drop(columns=["case_id"])
    uniq_groups = groups.nunique()
    split_count = max(2, min(n_splits, int(uniq_groups)))
    models: list[Pipeline] = []

    # Train fold models with group-aware split to avoid leakage between candidates of same case.
    if uniq_groups >= 2 and len(X_fit) >= 20:
        gkf = GroupKFold(n_splits=split_count)
        for train_idx, _valid_idx in gkf.split(X_fit, y, groups=groups):
            Xi = X_fit.iloc[train_idx]
            yi = y.iloc[train_idx]
            if yi.nunique() < 2:
                continue
            m = _make_reranker_pipeline()
            m.fit(Xi, yi)
            models.append(m)

    # Fallback full-data model (and slight regularization diversity for ensembling)
    if not models:
        m = _make_reranker_pipeline()
        m.fit(X_fit, y)
        models.append(m)
    else:
        full_m = _make_reranker_pipeline()
        full_m.fit(X_fit, y)
        models.append(full_m)
    return models


def _apply_reranker(
    base_df: pd.DataFrame,
    artifacts: dict[str, CaseArtifacts],
    *,
    models: list[Pipeline],
    weights: dict[str, float],
    use_failure_policy: bool,
) -> pd.DataFrame:
    if not models:
        return base_df
    out = base_df.copy()
    case_ids = out[out["is_service_evaluable"] == 1]["case_id"].astype(str).unique().tolist()
    for cid in case_ids:
        art = artifacts[cid]
        feat = _service_feature_frame(art, weights=weights, use_failure_policy=use_failure_policy)
        if feat.empty:
            continue
        feature_cols_num = _reranker_feature_cols_num()
        X = feat[feature_cols_num + ["failure_type"]]
        prob_list = [m.predict_proba(X)[:, 1] for m in models]
        p = np.mean(np.vstack(prob_list), axis=0)
        feat = feat.assign(rerank_score=p)
        feat = feat.sort_values("rerank_score", ascending=False).reset_index(drop=True)
        ordered = [(str(r["service"]), float(r["rerank_score"])) for _, r in feat.iterrows()]
        truth = art.case.gt_component_service
        m = _rank_metrics_from_ranking(ordered, truth)
        out.loc[out["case_id"] == cid, "pred_top1_service"] = ordered[0][0] if ordered else "insufficient_evidence"
        out.loc[out["case_id"] == cid, "pred_top1_score"] = ordered[0][1] if ordered else 0.0
        for k, v in m.items():
            out.loc[out["case_id"] == cid, k] = v
    return out


def _score_node_case(art: CaseArtifacts) -> dict[str, Any]:
    truth = art.case.gt_component_service
    pred = (art.node_top1_pred or "unknown").lower()
    # Node-specific refinement: combine node presence in traces/logs/metrics.
    logs, traces, metrics = _load_window_data(art.case.start_ts, art.case.end_ts)
    node_scores: dict[str, float] = {}
    for frame, w in ((logs, 0.35), (traces, 0.35), (metrics, 0.30)):
        if frame is None or frame.empty or "cmdb_id" not in frame.columns:
            continue
        nodes = frame["cmdb_id"].astype(str).str.lower()
        nodes = nodes[nodes.str.startswith("node-")]
        if nodes.empty:
            continue
        vc = nodes.value_counts()
        vmax = float(vc.max()) if len(vc) else 1.0
        for n, c in vc.items():
            node_scores[str(n)] = node_scores.get(str(n), 0.0) + w * (float(c) / (vmax or 1.0))
    if node_scores:
        pred = max(node_scores.items(), key=lambda x: x[1])[0]
    hit1 = 1 if pred == truth else 0
    mrr = float(hit1)
    ndcg = float(hit1)
    rank_loss = 0.0 if hit1 else 1.0
    rca_score = 0.4 * hit1 + 0.3 * mrr + 0.3 * ndcg
    return {
        "pred_top1_service": pred,
        "pred_top1_score": 1.0 if hit1 else 0.0,
        "ranking_size": 1,
        "true_rank": 1 if hit1 else 0,
        "top1_accuracy": hit1,
        "hit_at_3": hit1,
        "hit_at_5": hit1,
        "mrr": mrr,
        "ndcg": ndcg,
        "rank_loss": rank_loss,
        "rca_score": rca_score,
    }


def _evaluate_with_policy(
    artifacts: dict[str, CaseArtifacts],
    split_map: dict[str, str],
    *,
    weights: dict[str, float],
    use_failure_policy: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cid, art in artifacts.items():
        case = art.case
        is_service = _is_service_evaluable(case)
        if is_service:
            scored = _score_service_case(art, weights=weights, use_failure_policy=use_failure_policy)
        else:
            scored = _score_node_case(art)
        rows.append(
            {
                "case_id": case.case_id,
                "event_ts": case.event_ts.isoformat(),
                "start_ts": case.start_ts.isoformat(),
                "end_ts": case.end_ts.isoformat(),
                "gt_level": case.gt_level,
                "gt_component_raw": case.gt_component_raw,
                "gt_component_service": case.gt_component_service,
                "gt_failure_type": case.gt_failure_type,
                "is_service_evaluable": 1 if is_service else 0,
                "split": split_map[case.case_id],
                **scored,
            }
        )
    return pd.DataFrame(rows)


def _aggregate(df: pd.DataFrame, split: str) -> dict[str, Any]:
    sdf = df[df["split"] == split]
    if sdf.empty:
        return {"split": split, "num_cases": 0}
    return {
        "split": split,
        "num_cases": int(len(sdf)),
        "top1_accuracy": float(sdf["top1_accuracy"].mean()),
        "hit_at_3": float(sdf["hit_at_3"].mean()),
        "hit_at_5": float(sdf["hit_at_5"].mean()),
        "mrr": float(sdf["mrr"].mean()),
        "ndcg": float(sdf["ndcg"].mean()),
        "rank_loss": float(sdf["rank_loss"].mean()),
        "rca_score": float(sdf["rca_score"].mean()),
        "avg_loss": float(sdf["rank_loss"].mean()),
    }


def _aggregate_filtered(df: pd.DataFrame, split: str, field: str, value: int) -> dict[str, Any]:
    return _aggregate(df[(df["split"] == split) & (df[field] == value)], split)


def _running_mean(vals: list[float]) -> list[float]:
    out: list[float] = []
    s = 0.0
    for i, v in enumerate(vals, start=1):
        s += float(v)
        out.append(s / i)
    return out


def _plot_curves(df: pd.DataFrame, out_dir: Path) -> None:
    val = df[df["split"] == "val"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    if val.empty or test.empty:
        return

    val_acc = _running_mean(val["top1_accuracy"].tolist())
    test_acc = _running_mean(test["top1_accuracy"].tolist())
    val_loss = _running_mean(val["rank_loss"].tolist())
    test_loss = _running_mean(test["rank_loss"].tolist())

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(val_acc) + 1), val_acc, marker="o", label="Validation Accuracy")
    plt.plot(range(1, len(test_acc) + 1), test_acc, marker="o", label="Test Accuracy")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Cases processed")
    plt.ylabel("Running Top-1 Accuracy")
    plt.title("Validation/Test Accuracy (Running)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(val_loss) + 1), val_loss, marker="o", label="Validation Rank Loss")
    plt.plot(range(1, len(test_loss) + 1), test_loss, marker="o", label="Test Rank Loss")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Cases processed")
    plt.ylabel("Running Rank Loss")
    plt.title("Validation/Test Rank Loss (Running)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=160)
    plt.close()

    metrics = ["mrr", "ndcg", "rca_score"]
    val_scores = [float(val[m].mean()) for m in metrics]
    test_scores = [float(test[m].mean()) for m in metrics]
    x = range(len(metrics))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], val_scores, width=width, label="Validation")
    plt.bar([i + width / 2 for i in x], test_scores, width=width, label="Test")
    plt.xticks(list(x), [m.upper() for m in metrics])
    plt.ylim(0.0, 1.0)
    plt.title("Advanced RCA Metrics")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "advanced_metrics.png", dpi=160)
    plt.close()


def _save_confusion(df: pd.DataFrame, out_dir: Path) -> None:
    sdf = df[df["is_service_evaluable"] == 1]
    if sdf.empty:
        return
    y_true = sdf["gt_component_service"].astype(str)
    y_pred = sdf["pred_top1_service"].astype(str)
    labels = sorted(set(y_true).union(set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir / "confusion_matrix.csv")

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Service Track)")
    plt.colorbar()
    ticks = range(len(labels))
    plt.xticks(ticks, labels, rotation=90, fontsize=8)
    plt.yticks(ticks, labels, fontsize=8)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=160)
    plt.close()


def _generate_weight_grid() -> list[dict[str, float]]:
    vals = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    out: list[dict[str, float]] = []
    for trace in vals:
        for metric in vals:
            for log in vals:
                graph = 1.0 - trace - metric - log
                if graph < 0.10 or graph > 0.55:
                    continue
                out.append({"trace": trace, "metric": metric, "log": log, "graph": round(graph, 4)})
    out.append(DEFAULT_WEIGHTS.copy())
    return out


def _tune_weights(
    artifacts: dict[str, CaseArtifacts],
    train_ids: set[str],
    *,
    use_failure_policy: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    candidates = _generate_weight_grid()
    best = DEFAULT_WEIGHTS.copy()
    best_rank_loss = float("inf")
    best_mrr = -1.0
    for w in candidates:
        losses: list[float] = []
        mrrs: list[float] = []
        for cid in train_ids:
            art = artifacts[cid]
            if not _is_service_evaluable(art.case):
                continue
            scored = _score_service_case(art, weights=w, use_failure_policy=use_failure_policy)
            losses.append(float(scored["rank_loss"]))
            mrrs.append(float(scored["mrr"]))
        if not losses:
            continue
        loss = float(sum(losses) / len(losses))
        mrr = float(sum(mrrs) / len(mrrs))
        if (loss < best_rank_loss) or (math.isclose(loss, best_rank_loss) and mrr > best_mrr):
            best_rank_loss = loss
            best_mrr = mrr
            best = w
    return best, {"train_rank_loss": best_rank_loss, "train_mrr": best_mrr}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Advanced RCA evaluation with tuning and node track.")
    p.add_argument("--ground-truth-csv", default="data/graound_truth/groundtruth-2022-05-09.csv")
    p.add_argument("--window-before-seconds", type=int, default=120)
    p.add_argument("--window-after-seconds", type=int, default=120)
    p.add_argument("--train-ratio", type=float, default=0.6)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--max-cases", type=int, default=0, help="0 means all cases")
    p.add_argument("--output-dir", default="outputs/evaluation")
    p.add_argument("--disable-failure-policy", action="store_true")
    p.add_argument("--disable-weight-tuning", action="store_true")
    p.add_argument("--disable-reranker", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gt_csv = Path(args.ground_truth_csv)
    if not gt_csv.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {gt_csv}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(gt_csv, args.window_before_seconds, args.window_after_seconds)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    splits = split_cases(cases, args.train_ratio, args.val_ratio, args.test_ratio)
    split_map = {c.case_id: s for s, cs in splits.items() for c in cs}

    artifacts = _build_artifacts(cases)

    use_failure_policy = not args.disable_failure_policy
    tuning_stats: dict[str, float] = {}

    baseline_df = _evaluate_with_policy(
        artifacts,
        split_map=split_map,
        weights=DEFAULT_WEIGHTS,
        use_failure_policy=False,
    )
    baseline_val = _aggregate_filtered(baseline_df, "val", "is_service_evaluable", 1)
    baseline_val_loss = float(baseline_val.get("rank_loss", 1.0))

    selected_weights = DEFAULT_WEIGHTS.copy()
    selected_use_failure_policy = use_failure_policy
    selected_model_label = "baseline"
    selected_df = baseline_df

    if not args.disable_weight_tuning:
        tuned_weights, tuning_stats = _tune_weights(
            artifacts,
            train_ids={c.case_id for c in splits["train"]},
            use_failure_policy=use_failure_policy,
        )
        tuned_df = _evaluate_with_policy(
            artifacts,
            split_map=split_map,
            weights=tuned_weights,
            use_failure_policy=use_failure_policy,
        )
        tuned_val = _aggregate_filtered(tuned_df, "val", "is_service_evaluable", 1)
        tuned_val_loss = float(tuned_val.get("rank_loss", 1.0))

        if tuned_val_loss < baseline_val_loss:
            selected_weights = tuned_weights
            selected_use_failure_policy = use_failure_policy
            selected_model_label = "tuned"
            selected_df = tuned_df
        print(f"Selected fusion weights: {selected_weights}")
        print(f"Tuning stats: {tuning_stats}")

    reranker_used = False
    if not args.disable_reranker:
        train_ids = {c.case_id for c in splits["train"]}
        train_ds = _build_reranker_dataset(
            artifacts,
            train_ids,
            weights=selected_weights,
            use_failure_policy=selected_use_failure_policy,
        )
        reranker_models = _train_reranker_ensemble(train_ds)
        reranked_df = _apply_reranker(
            selected_df,
            artifacts,
            models=reranker_models,
            weights=selected_weights,
            use_failure_policy=selected_use_failure_policy,
        )
        base_val = _aggregate_filtered(selected_df, "val", "is_service_evaluable", 1)
        rerank_val = _aggregate_filtered(reranked_df, "val", "is_service_evaluable", 1)
        if float(rerank_val.get("rank_loss", 1.0)) <= float(base_val.get("rank_loss", 1.0)):
            selected_df = reranked_df
            selected_model_label = f"{selected_model_label}+reranker"
            reranker_used = len(reranker_models) > 0

    df = selected_df
    df.to_csv(out_dir / "predictions.csv", index=False)
    _plot_curves(df, out_dir)
    _save_confusion(df, out_dir)

    summary = {
        "config": {
            "ground_truth_csv": str(gt_csv),
            "window_before_seconds": args.window_before_seconds,
            "window_after_seconds": args.window_after_seconds,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "max_cases": args.max_cases,
            "rank_loss_formula": "rank(true_root_cause)/N, else 1.0 if missing",
            "rca_score_formula": "0.4*Hit@1 + 0.3*MRR + 0.3*NDCG",
            "failure_policy_enabled": use_failure_policy,
            "weight_tuning_enabled": not args.disable_weight_tuning,
            "reranker_enabled": not args.disable_reranker,
            "selection_policy": "choose baseline vs tuned by lower val rank_loss (service track)",
            "selected_model": selected_model_label,
        },
        "selected_fusion_weights": selected_weights,
        "selected_failure_policy_enabled": selected_use_failure_policy,
        "baseline_val_service_rank_loss": baseline_val_loss,
        "reranker_used": reranker_used,
        "tuning_stats": tuning_stats,
        "metrics": {
            "train": _aggregate(df, "train"),
            "val": _aggregate(df, "val"),
            "test": _aggregate(df, "test"),
        },
        "metrics_service_track": {
            "train": _aggregate_filtered(df, "train", "is_service_evaluable", 1),
            "val": _aggregate_filtered(df, "val", "is_service_evaluable", 1),
            "test": _aggregate_filtered(df, "test", "is_service_evaluable", 1),
        },
        "metrics_node_track": {
            "train": _aggregate_filtered(df, "train", "is_service_evaluable", 0),
            "val": _aggregate_filtered(df, "val", "is_service_evaluable", 0),
            "test": _aggregate_filtered(df, "test", "is_service_evaluable", 0),
        },
        "artifacts": {
            "predictions_csv": str(out_dir / "predictions.csv"),
            "accuracy_curve_png": str(out_dir / "accuracy_curve.png"),
            "loss_curve_png": str(out_dir / "loss_curve.png"),
            "advanced_metrics_png": str(out_dir / "advanced_metrics.png"),
            "confusion_matrix_csv": str(out_dir / "confusion_matrix.csv"),
            "confusion_matrix_png": str(out_dir / "confusion_matrix.png"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2))
    print(json.dumps(summary["metrics_service_track"], indent=2))
    print(json.dumps(summary["metrics_node_track"], indent=2))


if __name__ == "__main__":
    main()
