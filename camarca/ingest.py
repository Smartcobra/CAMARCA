"""
Load ground truth, logs, traces, and metrics; normalize timestamps to integer nanoseconds.

**Parquet (recommended):** Place ``*.parquet`` next to (or instead of) CSVs. Loaders
prefer Parquet when present—faster I/O, smaller disk, columnar reads.

**Speed tips (see also ``camarca.config``):**
- One-time: convert CSVs with ``python -m camarca.convert_to_parquet``.
- Prefer zstd compression; keep one format per directory (avoid mixing duplicate bases).
- Set ``INGEST_COLUMNS_*`` in config to read only columns the pipeline uses (big win for logs).
- After Parquet, the next bottlenecks are usually log-template regex and trace parent merge;
  consider sampling logs or processing in Polars/Dask for very large sets.
"""

from __future__ import annotations

import logging
from glob import glob
from pathlib import Path

import pandas as pd

from camarca.config import (
    INGEST_COLUMNS_LOGS,
    INGEST_COLUMNS_METRICS,
    INGEST_COLUMNS_TRACES,
    PARQUET_COMPRESSION,
    PREFER_PARQUET,
    TIME_BUCKET_SECONDS,
)

logger = logging.getLogger(__name__)

# PyArrow multithreaded decode (default True in recent pandas)
_READ_PARQUET_KWARGS = {"engine": "pyarrow", "use_threads": True}


def to_ns(ts) -> int:
    t = pd.to_datetime(ts)
    return int(t.value)


def _timestamp_series_to_datetime(series: pd.Series) -> pd.Series:
    s = series
    if pd.api.types.is_numeric_dtype(s) and s.notna().any():
        m = float(s.max())
        if m < 1e12:
            return pd.to_datetime(s, unit="s", utc=True)
        if m < 1e15:
            return pd.to_datetime(s, unit="ms", utc=True)
        if m < 1e18:
            return pd.to_datetime(s, unit="us", utc=True)
    return pd.to_datetime(s, utc=True)


def normalize_time(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Parse ``time_col`` to UTC, then set ``timestamp`` and ``timestamp_ns`` (int64, ns)."""
    df = df.copy()
    df["timestamp"] = _timestamp_series_to_datetime(df[time_col])
    ts = df["timestamp"].dt.as_unit("ns")
    df["timestamp_ns"] = ts.astype("int64")
    return df


def _read_tabular(path: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read CSV or Parquet; try ``columns`` for pruning, fall back to all columns on mismatch."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        if columns:
            try:
                return pd.read_parquet(path, columns=columns, **_READ_PARQUET_KWARGS)
            except (OSError, ValueError, TypeError) as e:
                logger.debug("Parquet column subset read failed %s: %s", path, e)
        return pd.read_parquet(path, **_READ_PARQUET_KWARGS)
    if columns:
        try:
            return pd.read_csv(path, usecols=columns)
        except (OSError, ValueError, TypeError) as e:
            logger.debug("CSV usecols read failed %s: %s", path, e)
    return pd.read_csv(path)


def _resolve_parquet_sibling(path: str) -> str:
    """
    If ``PREFER_PARQUET`` and ``path`` is ``.csv`` with a same-stem ``.parquet`` neighbor, use it.
    """
    if not PREFER_PARQUET:
        return path
    p = Path(path)
    if p.suffix.lower() == ".parquet" and p.is_file():
        return str(p)
    alt = p.with_suffix(".parquet")
    if p.suffix.lower() == ".csv" and alt.is_file():
        logger.info("Using Parquet (faster I/O): %s", alt.name)
        return str(alt)
    return path


def _log_files_in_dir(log_path: str) -> list[str]:
    """List log shards: all ``*.parquet`` if any exist, else all ``*.csv``."""
    d = Path(log_path)
    if not d.is_dir():
        return []
    pq = sorted(d.glob("*.parquet")) if PREFER_PARQUET else []
    if pq and PREFER_PARQUET:
        return [str(x) for x in pq]
    return sorted(glob(f"{log_path}/*.csv"))


def _metric_files(metrics_path: str) -> list[str]:
    """All metric Parquet under tree, or all CSV, not mixed to avoid double-counting."""
    pq = glob(f"{metrics_path}/**/*.parquet", recursive=True) if PREFER_PARQUET else []
    if pq and PREFER_PARQUET:
        return sorted(pq)
    return sorted(glob(f"{metrics_path}/**/*.csv", recursive=True))


def load_ground_truth(path: str) -> pd.DataFrame:
    path = _resolve_parquet_sibling(path)
    df = _read_tabular(path)
    if "start_time" in df.columns and "end_time" in df.columns:
        out = df.copy()
        out["start_ts"] = pd.to_datetime(out["start_time"])
        out["end_ts"] = pd.to_datetime(out["end_time"])
        return out
    if "timestamp" in df.columns:
        t = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        return pd.DataFrame(
            {
                "start_ts": [t.min()],
                "end_ts": [t.max()],
            }
        )
    raise ValueError("Ground truth must have (start_time, end_time) or `timestamp`.")


def load_logs(log_path: str) -> pd.DataFrame:
    files = _log_files_in_dir(log_path)
    cols = INGEST_COLUMNS_LOGS
    dfs: list[pd.DataFrame] = []
    for f in files:
        try:
            if f.endswith(".parquet"):
                df = _read_tabular(f, columns=cols)
            else:
                # CSV: optional subset via usecols if we know schema
                df = _read_tabular(f, columns=cols) if cols else _read_tabular(f)
        except OSError as e:
            logger.warning("Skip log file %s: %s", f, e)
            continue
        if "timestamp" not in df.columns:
            continue
        df = normalize_time(df, "timestamp")
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def _finalize_traces_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "startTime" in df.columns:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["startTime"], unit="us", utc=True)
    elif "timestamp" in df.columns:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        raise ValueError("Traces need `startTime` (Jaeger) or `timestamp` (ms).")
    ts = df["timestamp"].dt.as_unit("ns")
    df = df.copy()
    df["timestamp_ns"] = ts.astype("int64")
    return df


def load_traces(trace_path: str) -> pd.DataFrame:
    path = _resolve_parquet_sibling(trace_path)
    cols = INGEST_COLUMNS_TRACES
    if Path(path).suffix.lower() == ".parquet":
        df = _read_tabular(path, columns=cols)
    else:
        df = _read_tabular(path, columns=cols if cols else None)
    return _finalize_traces_frame(df)


def load_metrics(metrics_path: str) -> pd.DataFrame:
    all_files = _metric_files(metrics_path)
    cols = INGEST_COLUMNS_METRICS
    dfs: list[pd.DataFrame] = []
    for f in all_files:
        try:
            if f.endswith(".parquet"):
                df = _read_tabular(f, columns=cols)
            else:
                df = _read_tabular(f, columns=cols if cols else None)
            if "timestamp" in df.columns:
                df = normalize_time(df, "timestamp")
            elif "time" in df.columns:
                df = normalize_time(df, "time")
            else:
                continue
            dfs.append(df)
        except Exception as e:  # noqa: BLE001
            logger.debug("Skip metric file %s: %s", f, e)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def filter_time_window(df: pd.DataFrame, start_ns: int, end_ns: int) -> pd.DataFrame:
    if df.empty or "timestamp_ns" not in df.columns:
        return df
    return df[(df["timestamp_ns"] >= start_ns) & (df["timestamp_ns"] <= end_ns)]


def apply_time_bucket(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "timestamp_ns" not in df.columns:
        return df
    out = df.copy()
    out["bucket"] = (out["timestamp_ns"] // (TIME_BUCKET_SECONDS * 1e9)).astype(int)
    return out


def write_parquet(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> None:
    """Write a single table with zstd (configurable) for one-off conversion scripts."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(
        path,
        index=index,
        engine="pyarrow",
        compression=PARQUET_COMPRESSION,
    )
