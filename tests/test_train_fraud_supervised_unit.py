from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _install_fake_xgboost(monkeypatch) -> list[dict[str, object]]:
    created: list[dict[str, object]] = []
    module = types.ModuleType("xgboost")

    class FakeXGBClassifier:
        def __init__(self, **kwargs) -> None:
            created.append(kwargs)

        def fit(self, X, y):
            self._n = len(X)
            return self

        def predict_proba(self, X):
            base = np.linspace(0.1, 0.9, len(X))
            return np.vstack([1 - base, base]).T

    module.XGBClassifier = FakeXGBClassifier
    monkeypatch.setitem(sys.modules, "xgboost", module)
    return created


def test_train_fraud_supervised_writes_registry_and_reports(
    tmp_path, monkeypatch
) -> None:
    created = _install_fake_xgboost(monkeypatch)

    import ml.training.train_fraud_supervised as train_module

    df = pd.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0, 4.0],
            "is_international": [0, 1, 0, 1],
            "hour": [1, 2, 3, 4],
            "is_weekend": [0, 0, 1, 1],
            "amount_bucket": [0, 1, 1, 2],
            "velocity_24h": [1, 2, 3, 4],
            "is_fraud": [0, 1, 0, 1],
        }
    )
    input_csv = tmp_path / "input.csv"
    df.to_csv(input_csv, index=False)

    registry_path = tmp_path / "model_registry.json"
    artefacts_dir = tmp_path / "artefacts"
    reports_dir = tmp_path / "reports"

    args = argparse.Namespace(
        input=str(input_csv),
        label_col="is_fraud",
        amount_col="amount",
        synthetic="no",
        synthetic_quantile=0.95,
        features=None,
        registry=str(registry_path),
        artefacts_dir=str(artefacts_dir),
        reports_dir=str(reports_dir),
        seed=42,
    )

    monkeypatch.setattr(train_module, "dump", lambda *args, **kwargs: None)
    train_module.main(args)

    assert created
    registry = json.loads(registry_path.read_text())
    assert registry["fraud"]["latest"]
    assert registry["fraud"][registry["fraud"]["latest"]]["type"] == "supervised_xgb"

    metrics_files = list(reports_dir.glob("fraud_xgb_*_metrics.json"))
    assert metrics_files
    thresholds_files = list(reports_dir.glob("fraud_xgb_*_thresholds.csv"))
    assert thresholds_files


def test_train_fraud_supervised_auto_synthetic_fallback(tmp_path, monkeypatch) -> None:
    _install_fake_xgboost(monkeypatch)

    import ml.training.train_fraud_supervised as train_module

    def split_with_fallback(X, y, test_size, random_state, stratify=None):
        if stratify is not None:
            raise ValueError("force fallback")
        split = int(len(X) * (1 - test_size))
        return X[:split], X[split:], y[:split], y[split:]

    monkeypatch.setattr(train_module, "train_test_split", split_with_fallback)

    df = pd.DataFrame(
        {
            "amount": [10.0, 20.0, 30.0, 40.0],
            "is_international": [0, 0, 0, 0],
            "hour": [1, 2, 3, 4],
            "is_weekend": [0, 0, 0, 0],
            "amount_bucket": [0, 0, 0, 0],
            "velocity_24h": [1, 1, 1, 1],
        }
    )
    input_csv = tmp_path / "input.csv"
    df.to_csv(input_csv, index=False)

    registry_path = tmp_path / "model_registry.json"

    args = argparse.Namespace(
        input=str(input_csv),
        label_col="is_fraud",
        amount_col="amount",
        synthetic="auto",
        synthetic_quantile=0.9,
        features=None,
        registry=str(registry_path),
        artefacts_dir=str(tmp_path / "artefacts"),
        reports_dir=str(tmp_path / "reports"),
        seed=123,
    )

    monkeypatch.setattr(train_module, "dump", lambda *args, **kwargs: None)
    train_module.main(args)

    registry = json.loads(registry_path.read_text())
    latest = registry["fraud"]["latest"]
    label_source = registry["fraud"][latest]["metrics"]["label_source"]
    assert "synthetic_top_quantile_0.9" in label_source


def test_require_columns_raises() -> None:
    import ml.training.train_fraud_supervised as train_module

    df = pd.DataFrame({"amount": [1.0]})
    with pytest.raises(ValueError):
        train_module._require_columns(df, ["missing_col"])
