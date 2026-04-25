"""Shared logging setup helpers."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO, fmt: str | None = None) -> None:
    """Idempotent logging config for ``uv run`` and ``python -m``."""
    if fmt is None:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    if not any(
        isinstance(h, logging.StreamHandler)
        and getattr(h, "stream", None) in (sys.stdout, sys.stderr)
        for h in logging.root.handlers
    ):
        logging.basicConfig(
            level=level,
            format=fmt,
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
