"""Default paths, I/O, and model knobs. Override in production or set env-based config.

Speed (after switching to Parquet, largest wins first):
- Run ``python -m camarca.convert_to_parquet`` once; keep zstd (see ``PARQUET_COMPRESSION``).
- Narrow ``INGEST_COLUMNS_LOGS`` / ``INGEST_COLUMNS_TRACES`` to columns the pipeline uses.
- Leave ``INGEST_COLUMNS_METRICS`` as None unless every metric file shares a schema; otherwise
  Pandas can skip ragged usecols and fall back to a full read per file.
- Tighten the time window in ground truth so :func:`camarca.ingest.filter_time_window` works on
  less data (still loads full files unless you partition Parquet by time in object storage).
- For multi-GB runs: log templates and trace parent merge are CPU-heavy—sample logs or use Polars.
"""

from __future__ import annotations

# Prefer ``*.parquet`` over CSV when a sibling or directory contains Parquet.
PREFER_PARQUET = True

# Written by ``camarca.convert_to_parquet``; zstd balances size and decode speed.
PARQUET_COMPRESSION: str = "zstd"

# If set, Parquet/CSV readers try these columns only (faster, less memory). None = all columns.
# Traces: keep fields needed for graph + duration + time.
INGEST_COLUMNS_TRACES: list[str] | None = [
    "timestamp",
    "cmdb_id",
    "span_id",
    "trace_id",
    "duration",
    "parent_span",
    "type",
    "status_code",
    "operation_name",
]
INGEST_COLUMNS_LOGS: list[str] | None = [
    "log_id",
    "timestamp",
    "cmdb_id",
    "log_name",
    "value",
]
# Ragged metric CSVs (JVM vs service) → read full row unless you normalize schemas.
INGEST_COLUMNS_METRICS: list[str] | None = None

# Time discretization for log/trace/metric alignment (nanoseconds are divided by this).
TIME_BUCKET_SECONDS = 30

# Isolation Forest: expected fraction of outliers. Tune per environment.
IF_CONTAMINATION = 0.05

# Fast mode (quick RCA for interactive use)
FAST_MODE_DEFAULT = False
# Fraction of rows kept per modality in fast mode.
FAST_SAMPLE_LOGS_FRAC = 0.10
FAST_SAMPLE_TRACES_FRAC = 0.15
FAST_SAMPLE_METRICS_FRAC = 0.20
# Deterministic sampling.
FAST_SAMPLE_RANDOM_STATE = 42

# Causal graph: blend trace hop mass, anomalous metric (bucket,service) count, and log template mass.
WEIGHT_TRACE = 0.5
WEIGHT_METRIC = 0.3
WEIGHT_LOG = 0.2
# Cap raw call counts before weighting (avoids million-scale dominance from hot edges).
CAUSAL_MAX_TRACE_WEIGHT = 1e4

# Hybrid graph edge composition
EDGE_WEIGHT_LATENCY = 0.4
EDGE_WEIGHT_ERROR = 0.4
EDGE_WEIGHT_CALL_FREQ = 0.2

# Residual confidence fusion weights
HYBRID_W_TRACE_CONF = 0.25
HYBRID_W_METRIC_CONF = 0.25
HYBRID_W_LOG_CONF = 0.20
HYBRID_W_GRAPH_CAUSAL = 0.30

# Graph leaf term uses 1 / (1 + LEAF_OUT_DEGREE_PENALTY * out_degree). Using 1.0 would halve
# the score for every outgoing edge vs a pure sink; real faults often sit on mid-tier services
# with one downstream hop; tune with FAN_IN_DAMPEN.
LEAF_OUT_DEGREE_PENALTY = 0.085

# Divide raw incoming edge mass by (1 + FAN_IN_DAMPEN * max(0, in_edge_count-1)) before
# normalizing, so many parallel callers (e.g. catalog) do not dwarf mid-tier services.
FAN_IN_DAMPEN = 0.52

# Score *= 1 / (1 + BROADCASTER_PENALTY * max(0, out_degree - GATEWAY_OUT_DEGREE_FLOOR)) so
# only many-child services (e.g. frontend) are demoted, not 1-hop services (e.g. cart -> x).
BROADCASTER_PENALTY = 0.22
GATEWAY_OUT_DEGREE_FLOOR = 2

# Slightly down-weight the log term for "hub sinks" (>= this many in-edges, no out-edges in
# the call graph) because log volume is inflated by every upstream caller.
HUB_SINK_IN_DEGREE = 3
# Multiplier on the (normalized) per-service log score for such nodes; < 1.0
HUB_SINK_LOG_SCALE = 0.84
