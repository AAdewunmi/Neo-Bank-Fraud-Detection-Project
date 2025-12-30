from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


pytestmark = pytest.mark.ml


def test_trainer_writes_metrics_and_thresholds(tmp_path: Path) -> None:
    pytest.importorskip("xgboost")

    df = pd.DataFrame(
        {
            "amount": np.linspace(1, 500, 400),
            "is_international": np.random.default_rng(42).integers(0, 2, size=400),
            "hour": np.random.default_rng(42).integers(0, 24, size=400),
            "is_weekend": np.random.default_rng(42).integers(0, 2, size=400),
            "amount_bucket": np.random.default_rng(42).integers(0, 4, size=400),
            "velocity_24h": np.random.default_rng(42).integers(0, 8, size=400),
            "is_fraud": (np.linspace(1, 500, 400) > 450).astype(int),
        }
    )

    input_csv = tmp_path / "mini.csv"
    df.to_csv(input_csv, index=False)

    artefacts_dir = tmp_path / "artefacts"
    reports_dir = tmp_path / "reports"

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "ml.training.train_fraud_supervised",
            "--input",
            str(input_csv),
            "--label_col",
            "is_fraud",
            "--synthetic",
            "no",
            "--registry",
            "model_registry.json",
            "--artefacts_dir",
            str(artefacts_dir),
            "--reports_dir",
            str(reports_dir),
        ]
    )

    metrics_files = list(reports_dir.glob("fraud_xgb_*_metrics.json"))
    assert metrics_files, "metrics json not written"

    metrics = json.loads(metrics_files[-1].read_text())
    assert "average_precision" in metrics
    assert "threshold_table_csv" in metrics

    thresholds_csv = Path(metrics["threshold_table_csv"])
    assert thresholds_csv.exists(), "threshold table csv not written"

    tdf = pd.read_csv(thresholds_csv)
    assert {"threshold", "precision", "recall", "f1"}.issubset(set(tdf.columns))
    assert len(tdf) > 5
