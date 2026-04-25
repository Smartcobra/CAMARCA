"""Build hour-partitioned Parquet layout for faster window reads."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from camarca.ingest import load_logs, load_metrics, load_traces, write_parquet

logger = logging.getLogger(__name__)


def _write_hour_partitions(
    df: pd.DataFrame,
    *,
    out_root: Path,
    ts_col: str = "timestamp",
    basename: str,
    dry_run: bool = False,
) -> int:
    if df.empty or ts_col not in df.columns:
        return 0
    x = df.copy()
    x["_dt"] = x[ts_col].dt.strftime("%Y-%m-%d")
    x["_hour"] = x[ts_col].dt.strftime("%H")
    n = 0
    for (dt, hh), g in x.groupby(["_dt", "_hour"], sort=True):
        out = out_root / f"dt={dt}" / f"hour={hh}" / f"{basename}.parquet"
        if out.is_file():
            continue
        if dry_run:
            logger.info("Would write %s rows -> %s", len(g), out)
            n += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        write_parquet(g.drop(columns=["_dt", "_hour"]), out, index=False)
        n += 1
    return n


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    p = argparse.ArgumentParser(
        description="Create hour-partitioned Parquet for logs/traces/metrics."
    )
    p.add_argument("--out", default="data/partitioned", help="Output root for partitioned parquet.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    logs = load_logs("data/logs/2022-05-09")
    traces = load_traces("data/traces/2022-05-09/trace_jaeger-span.csv")
    metrics = load_metrics("data/metrics/2022-05-09")

    n_logs = _write_hour_partitions(
        logs, out_root=out / "logs", basename="logs", dry_run=args.dry_run
    )
    n_traces = _write_hour_partitions(
        traces, out_root=out / "traces", basename="traces", dry_run=args.dry_run
    )
    n_metrics = _write_hour_partitions(
        metrics, out_root=out / "metrics", basename="metrics", dry_run=args.dry_run
    )
    logger.info("Done. partitions logs=%s traces=%s metrics=%s", n_logs, n_traces, n_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
