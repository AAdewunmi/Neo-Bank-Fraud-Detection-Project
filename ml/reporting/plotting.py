"""
Headless-safe plotting helpers.

P1 #7
- Sets a non-interactive backend before importing pyplot.
- Avoids CI failures on headless runners.

This file is optional for Day 2 training, but it makes Day 3 to Day 4 reporting smoother.
"""

from __future__ import annotations

from pathlib import Path

import json


def _use_headless_backend() -> None:
    """
    Set matplotlib backend to Agg before importing pyplot.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)


def save_pr_curve_from_threshold_table(thresholds_csv: str, output_path: str, title: str) -> None:
    """
    Plot precision and recall versus threshold and save to disk.

    thresholds_csv is expected to have columns threshold, precision, recall.
    """
    _use_headless_backend()
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(thresholds_csv)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df["threshold"], df["precision"], label="precision")
    ax.plot(df["threshold"], df["recall"], label="recall")
    ax.set_title(title)
    ax.set_xlabel("threshold")
    ax.set_ylabel("value")
    ax.legend()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_ap_badge(metrics_json: str, output_path: str) -> None:
    """
    Save a small text-only report file for quick inspection in CI artefacts.
    """
    data = json.loads(Path(metrics_json).read_text())
    ap = data.get("average_precision")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(f"average_precision={ap}\n")
