from __future__ import annotations

from pathlib import Path
import json
import os

import pytest

from ml.metrics.plots import pr_curve_from_metrics, threshold_precision_recall


@pytest.mark.ml
def test_plot_functions_write_png(tmp_path: Path) -> None:
    os.environ["MPLBACKEND"] = "Agg"

    m = {
        "precision_curve": [1.0, 0.8, 0.7],
        "recall_curve": [0.0, 0.5, 0.9],
        "thresholds": [0.2, 0.6],
        "precision": [0.8, 0.7],
        "recall": [0.5, 0.9],
    }
    metrics_path = tmp_path / "m.json"
    metrics_path.write_text(json.dumps(m), encoding="utf-8")

    out1 = tmp_path / "pr.png"
    out2 = tmp_path / "thr.png"

    pr_curve_from_metrics(str(metrics_path), str(out1))
    threshold_precision_recall(str(metrics_path), str(out2))

    assert out1.exists()
    assert out2.exists()
    assert out1.stat().st_size > 0
    assert out2.stat().st_size > 0
