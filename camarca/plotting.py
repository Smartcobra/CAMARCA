"""Plot helpers for RCA outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_top_root_causes(
    ranking: list[tuple[str, float]],
    output_path: str = "outputs/top5_rca.png",
    *,
    top_n: int = 5,
    title: str = "Top Root Cause Services",
) -> str:
    """Save a horizontal bar chart for the top-N RCA services."""
    items = ranking[:top_n]
    if not items:
        raise ValueError("Ranking is empty; cannot plot root causes.")

    labels = [svc for svc, _ in items][::-1]
    scores = [score for _, score in items][::-1]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, scores, color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel("RCA Score")
    ax.set_ylabel("Service")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    # Annotate bars for quick visual reading.
    for bar, score in zip(bars, scores, strict=False):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {score:.2f}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(Path(output_path))
