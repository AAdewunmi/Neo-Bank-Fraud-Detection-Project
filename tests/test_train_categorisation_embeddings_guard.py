# tests/test_train_categorisation_embeddings_guard.py
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ml.training import train_categorisation_embeddings as train_module


def test_train_embeddings_relaxes_lightgbm_params_for_small_dataset(tmp_path, monkeypatch):
    class FakeEncoder:
        def __init__(self, model_name: str, device: str = "cpu") -> None:
            self.model_name = model_name
            self.device = device

        def encode(self, texts):
            return np.zeros((len(texts), 4))

    created = []

    class FakeLGBMClassifier:
        def __init__(self, **kwargs) -> None:
            self.params = kwargs
            created.append(self)

        def fit(self, X, y):
            self._label = str(list(y)[0])
            return self

        def predict(self, X):
            return np.array([self._label] * len(X))

        def predict_proba(self, X):
            return np.tile([0.3, 0.7], (len(X), 1))

    monkeypatch.setattr(train_module, "MiniLMEncoder", FakeEncoder)
    monkeypatch.setattr(train_module.lgb, "LGBMClassifier", FakeLGBMClassifier)
    monkeypatch.setattr(train_module, "dump", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module, "classification_report", lambda *args, **kwargs: "ok")
    monkeypatch.setattr(train_module, "f1_score", lambda *args, **kwargs: 0.0)
    monkeypatch.chdir(tmp_path)

    df = pd.DataFrame(
        {
            "merchant": [f"m{i}" for i in range(10)],
            "description": [f"d{i}" for i in range(10)],
            "category": ["A"] * 5 + ["B"] * 5,
        }
    )
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)

    args = argparse.Namespace(
        input=str(csv_path),
        target_col="category",
        text_cols=["merchant", "description"],
        registry="model_registry.json",
        encoder_name="fake-encoder",
    )

    train_module.main(args)

    assert created
    assert created[0].params["min_data_in_leaf"] == 1
    assert created[0].params["min_data_in_bin"] == 1
