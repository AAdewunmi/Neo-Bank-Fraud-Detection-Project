# tests/test_scorer_branches.py
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

import ml.inference.scorer as scorer_module
from ml.inference.scorer import Scorer


def _install_fake_sentence_transformers(monkeypatch):
    module = types.ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, name: str, device: str = "cpu") -> None:
            self.name = name
            self.device = device

        def encode(
            self,
            texts,
            convert_to_numpy: bool = True,
            normalize_embeddings: bool = True,
        ):
            return np.zeros((len(texts), 3))

    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def _registry_with_models():
    return {
        "categorisation": {
            "latest": "v1",
            "v1": {
                "text_cols": ["merchant", "description"],
                "artefact": "cat.joblib",
            },
        },
        "fraud": {
            "latest": "f1",
            "f1": {
                "features": ["amount"],
                "artefact": "fraud.joblib",
            },
        },
    }


def test_scorer_caches_artefacts_and_encoders(monkeypatch):
    monkeypatch.setattr(scorer_module, "load_registry", lambda _: _registry_with_models())
    scorer = Scorer()

    calls = []

    def fake_load(path):
        calls.append(path)
        return {"path": path}

    monkeypatch.setattr(scorer_module, "load", fake_load)

    obj_a = scorer._load_artefact("cat.joblib")
    obj_b = scorer._load_artefact("cat.joblib")
    assert obj_a is obj_b
    assert calls == ["cat.joblib"]

    _install_fake_sentence_transformers(monkeypatch)
    enc_a = scorer._get_encoder("fake-mini")
    enc_b = scorer._get_encoder("fake-mini")
    assert enc_a is enc_b


def test_scorer_embeddings_path_with_decision_function(monkeypatch):
    monkeypatch.setattr(scorer_module, "load_registry", lambda _: _registry_with_models())
    _install_fake_sentence_transformers(monkeypatch)

    class FakeCatModel:
        def predict(self, X):
            return ["Food"] * len(X)

        def predict_proba(self, X):
            return np.tile([0.2, 0.8], (len(X), 1))

    class FakeFraudModel:
        def decision_function(self, X):
            return -np.linspace(0.1, 1.0, len(X))

    def fake_load(path):
        if path == "cat.joblib":
            return {"model": FakeCatModel(), "encoder_name": "fake-mini"}
        return FakeFraudModel()

    monkeypatch.setattr(scorer_module, "load", fake_load)

    scorer = Scorer()
    df = pd.DataFrame(
        {
            "merchant": ["m1", "m2"],
            "description": ["d1", "d2"],
            "amount": [10.0, 20.0],
        }
    )

    scored, diags = scorer.score(df, threshold=0.5)

    assert {"category", "category_confidence", "fraud_risk", "flagged"}.issubset(
        scored.columns
    )
    assert diags["pct_flagged"] >= 0.0


def test_scorer_plain_model_with_score_samples(monkeypatch):
    monkeypatch.setattr(scorer_module, "load_registry", lambda _: _registry_with_models())

    class FakeCatModel:
        def predict(self, X):
            return ["Other"] * len(X)

    class FakeFraudModel:
        def score_samples(self, X):
            return np.array([-0.1, -0.2, -0.3])

    def fake_load(path):
        if path == "cat.joblib":
            return FakeCatModel()
        return FakeFraudModel()

    monkeypatch.setattr(scorer_module, "load", fake_load)

    scorer = Scorer()
    df = pd.DataFrame(
        {
            "merchant": ["m1", "m2", "m3"],
            "description": ["d1", "d2", "d3"],
            "amount": [5.0, 6.0, 7.0],
        }
    )

    scored, _ = scorer.score(df, threshold=0.4)
    assert np.allclose(scored["category_confidence"].to_numpy(), 1.0)


def test_scorer_raises_for_missing_fraud_methods(monkeypatch):
    monkeypatch.setattr(scorer_module, "load_registry", lambda _: _registry_with_models())

    class FakeCatModel:
        def predict(self, X):
            return ["X"] * len(X)

        def predict_proba(self, X):
            return np.tile([0.4, 0.6], (len(X), 1))

    class FakeFraudModel:
        pass

    def fake_load(path):
        if path == "cat.joblib":
            return FakeCatModel()
        return FakeFraudModel()

    monkeypatch.setattr(scorer_module, "load", fake_load)

    scorer = Scorer()
    df = pd.DataFrame(
        {
            "merchant": ["m1"],
            "description": ["d1"],
            "amount": [1.0],
        }
    )

    with pytest.raises(AttributeError):
        scorer.score(df, threshold=0.5)
