"""
One-time (or batch) conversion of the demo CSVs to Parquet for faster pipeline loads.

Run from the repository root::

    uv run python -m camarca.convert_to_parquet

This writes ``*.parquet`` next to each ``*.csv`` (same directory, same base name).
The pipeline then auto-selects Parquet when :data:`camarca.config.PREFER_PARQUET` is True.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from camarca.ingest import write_parquet

logger = logging.getLogger(__name__)


def _convert_one_csv(csv_path: Path, *, dry_run: bool = False) -> bool:
    pq = csv_path.with_suffix(".parquet")
    if pq.is_file():
        logger.info("Skip (exists): %s", pq)
        return False
    if dry_run:
        logger.info("Would write: %s", pq)
        return True
    logger.info("Converting: %s -> %s", csv_path.name, pq.name)
    df = pd.read_csv(csv_path)
    write_parquet(df, pq, index=False)
    return True


def convert_default_dataset(
    root: Path,
    *,
    dry_run: bool = False,
) -> int:
    """Convert ground truth, logs, single trace file, and all metrics CSVs under ``root``."""
    n = 0
    gt = root / "ground truth" / "groundtruth-2022-05-09.csv"
    if gt.is_file() and _convert_one_csv(gt, dry_run=dry_run):
        n += 1
    log_dir = root / "logs" / "2022-05-09"
    if log_dir.is_dir():
        for f in sorted(log_dir.glob("*.csv")):
            if _convert_one_csv(f, dry_run=dry_run):
                n += 1
    tr = root / "traces" / "2022-05-09" / "trace_jaeger-span.csv"
    if tr.is_file() and _convert_one_csv(tr, dry_run=dry_run):
        n += 1
    mroot = root / "metrics" / "2022-05-09"
    if mroot.is_dir():
        for f in sorted(mroot.rglob("*.csv")):
            if _convert_one_csv(f, dry_run=dry_run):
                n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        stream=sys.stderr,
    )
    p = argparse.ArgumentParser(
        description="Materialize Parquet next to each CSV for faster Camarca ingest.",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root containing ground truth, logs, traces, metrics (default: ./data).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without writing.",
    )
    args = p.parse_args(argv)
    if not args.data_root.is_dir():
        logger.error("Data root is not a directory: %s", args.data_root)
        return 1
    n = convert_default_dataset(args.data_root, dry_run=args.dry_run)
    logger.info("Done. %s new Parquet file(s) %s.", n, "planned" if args.dry_run else "written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
