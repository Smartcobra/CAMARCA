"""Small Parquet cache helpers for repeated fault-window queries."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def _window_key(start_ns: int, end_ns: int, tag: str) -> str:
    raw = f"{tag}:{start_ns}:{end_ns}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def cache_path(cache_dir: str | Path, *, start_ns: int, end_ns: int, tag: str) -> Path:
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    return cdir / f"{tag}_{_window_key(start_ns, end_ns, tag)}.parquet"


def load_df(cache_dir: str | Path, *, start_ns: int, end_ns: int, tag: str) -> pd.DataFrame | None:
    p = cache_path(cache_dir, start_ns=start_ns, end_ns=end_ns, tag=tag)
    if p.is_file():
        return pd.read_parquet(p, engine="pyarrow")
    return None


def save_df(
    df: pd.DataFrame,
    cache_dir: str | Path,
    *,
    start_ns: int,
    end_ns: int,
    tag: str,
) -> Path:
    p = cache_path(cache_dir, start_ns=start_ns, end_ns=end_ns, tag=tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, engine="pyarrow", index=False)
    return p
