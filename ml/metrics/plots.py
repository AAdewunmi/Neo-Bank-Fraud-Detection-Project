"""
Plotting utilities for PR curve and threshold trade offs.

Headless safety
- Forces a non interactive backend before importing pyplot.
- Avoids CI failures related to display availability.

Inputs
- Metrics JSON written by train_fraud_supervised.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import json

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:  # pragma: no cover - handled in unit tests via fallbacks
    plt = None
    _HAS_MPL = False


def _write_placeholder_png(out_png: str) -> None:
    """
    Write a minimal valid PNG when matplotlib is unavailable.
    """
    # 1x1 transparent PNG.
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00"
        b"\x00\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    Path(out_png).write_bytes(png_bytes)


def _load_metrics(metrics_path: str) -> dict:
    """
    Load metrics JSON from disk.
    """
    return json.loads(Path(metrics_path).read_text(encoding="utf-8"))


def _series(m: dict, key: str) -> List[float]:
    """
    Read a numeric list from metrics safely.
    """
    v = m.get(key, [])
    if not isinstance(v, list):
        return []
    return [float(x) for x in v]


def pr_curve_from_metrics(metrics_path: str, out_png: str) -> None:
    """
    Render a PR curve PNG.

    Uses precision_curve and recall_curve when available.
    Falls back to precision and recall for older metrics files.
    """
    m = _load_metrics(metrics_path)
    recall = _series(m, "recall_curve") or _series(m, "recall")
    precision = _series(m, "precision_curve") or _series(m, "precision")

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)

    if not _HAS_MPL:
        _write_placeholder_png(out_png)
        return

    plt.figure()
    if recall and precision and len(recall) == len(precision):
        plt.step(recall, precision, where="post")
    else:
        plt.text(0.1, 0.5, "PR data unavailable", fontsize=12)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")
    plt.tight_layout()
    plt.savefig(out_png, dpi=144)
    plt.close()


def threshold_precision_recall(metrics_path: str, out_png: str) -> None:
    """
    Render precision and recall vs threshold PNG.

    Expects thresholds, precision, recall arrays aligned to the same length.
    """
    m = _load_metrics(metrics_path)
    thr = _series(m, "thresholds")
    prec = _series(m, "precision")
    rec = _series(m, "recall")

    k = min(len(thr), len(prec), len(rec))
    thr = thr[:k]
    prec = prec[:k]
    rec = rec[:k]

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)

    if not _HAS_MPL:
        _write_placeholder_png(out_png)
        return

    plt.figure()
    if k > 0:
        plt.plot(thr, prec, label="Precision")
        plt.plot(thr, rec, label="Recall")
        plt.legend()
    else:
        plt.text(0.1, 0.5, "Threshold data unavailable", fontsize=12)
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision and Recall vs Threshold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=144)
    plt.close()
