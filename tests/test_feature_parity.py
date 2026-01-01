# tests/test_fraud_feature_parity.py
"""Tests for Day 4 train-inference feature parity.

These tests protect two contracts
1) Feature builders return a strict, ordered schema.
2) The inference scorer can compute engineered fraud features when the registry requests them.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ml.fraud_features import FEATURE_ORDER, compute_infer_features, compute_train_features
from ml.inference import scorer as scorer_module
from ml.inference.scorer import Scorer


def test_feature_builders_return_strict_ordered_schema() -> None:
    """Feature builders must emit the same ordered columns for train and inference."""
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime(2025, 12, 31, 10, 0, tzinfo=timezone.utc).isoformat(),
                datetime(2025, 12, 31, 11, 0, tzinfo=timezone.utc).isoformat(),
            ],
            "customer_id": ["c1", "c1"],
            "amount": [10.0, 20.0],
        }
    )

    train_feat = compute_train_features(df)
    infer_feat = compute_infer_features(df)

    assert list(train_feat.columns) == FEATURE_ORDER
    assert list(infer_feat.columns) == FEATURE_ORDER


def test_scorer_computes_engineered_fraud_features_when_registry_requests_them(monkeypatch) -> None:
    """Scorer should compute engineered features before calling the fraud model."""

    class FakeCatModel:
        def predict(self, X):
            return ["Other"] * len(X)

        def predict_proba(self, X):
            return np.tile([0.7, 0.3], (len(X), 1))

    class FakeFraudModel:
        def __init__(self):
            self.last_shape = None

        def predict_proba(self, X):
            self.last_shape = X.shape

            # Risk increases with amount as a deterministic proxy.
            # X[:, 0] is "amount" by FEATURE_ORDER.
            amount = np.asarray(X[:, 0], dtype=float)
            denom = float(amount.max()) if float(amount.max()) else 1.0
            risk = np.clip(amount / denom, 0.0, 1.0)

            p0 = 1.0 - risk
            p1 = risk
            return np.stack([p0, p1], axis=1)

    fraud_model = FakeFraudModel()

    registry = {
        "categorisation": {
            "latest": "cat_v1",
            "cat_v1": {"artefact": "cat.joblib", "text_cols": ["merchant", "description"]},
        },
        "fraud": {
            "latest": "fraud_v1",
            "fraud_v1": {"artefact": "fraud.joblib", "features": FEATURE_ORDER},
        },
    }

    def fake_load_registry(_path: str):
        return registry

    def fake_load(path: str):
        if path == "cat.joblib":
            return FakeCatModel()
        if path == "fraud.joblib":
            return fraud_model
        raise AssertionError(f"Unexpected load path: {path}")

    monkeypatch.setattr(scorer_module, "load_registry", fake_load_registry)
    monkeypatch.setattr(scorer_module, "load", fake_load)

    df = pd.DataFrame(
        {
            "merchant": ["m1", "m2", "m3"],
            "description": ["d1", "d2", "d3"],
            "timestamp": [
                datetime(2025, 12, 31, 10, 0, tzinfo=timezone.utc).isoformat(),
                datetime(2025, 12, 31, 10, 5, tzinfo=timezone.utc).isoformat(),
                datetime(2025, 12, 31, 10, 10, tzinfo=timezone.utc).isoformat(),
            ],
            "customer_id": ["c1", "c1", "c2"],
            "amount": [10.0, 12.0, 500.0],
        }
    )

    scorer = Scorer()
    scored, diags = scorer.score(df, threshold=0.5)

    assert fraud_model.last_shape == (len(df), len(FEATURE_ORDER))
    assert "fraud_risk" in scored.columns
    assert "fraud_flag" in scored.columns
    assert diags["n"] == len(df)
