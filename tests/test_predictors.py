from __future__ import annotations

import sys
import types

import numpy as np

import ml.inference.predictors as predictors


def test_as_text_categoriser_prefers_existing_predictor() -> None:
    class AlreadyPredicts:
        def predict_with_confidence(self, texts):  # pragma: no cover - simple stub
            return np.array(texts), np.ones(len(texts))

    predictor = AlreadyPredicts()
    assert predictors.as_text_categoriser(predictor) is predictor


def test_sklearn_pipeline_confidence_path() -> None:
    class FakePipeline:
        def predict(self, texts):
            return ["A"] * len(texts)

        def predict_proba(self, texts):
            return np.tile([0.1, 0.9], (len(texts), 1))

    adapter = predictors.SklearnTextPipelineCategoriser(pipeline=FakePipeline())
    labels, conf = adapter.predict_with_confidence(["t1", "t2"])

    assert labels.tolist() == ["A", "A"]
    assert np.allclose(conf, 0.9)


def test_embeddings_categoriser_uses_cached_encoder(monkeypatch) -> None:
    predictors._ENCODER_CACHE = {}

    module = types.ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, name: str) -> None:
            self.name = name

        def encode(
            self,
            texts,
            convert_to_numpy: bool = True,
            normalize_embeddings: bool = True,
        ):
            return np.zeros((len(texts), 3))

    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)

    class FakeModel:
        def predict(self, X):
            return ["Food"] * len(X)

        def predict_proba(self, X):
            return np.tile([0.2, 0.8], (len(X), 1))

    encoder_a = predictors._get_sentence_transformer("fake-mini")
    encoder_b = predictors._get_sentence_transformer("fake-mini")
    assert encoder_a is encoder_b

    adapter = predictors.EmbeddingsLightGBMCategoriser(
        model=FakeModel(),
        encoder_name="fake-mini",
    )
    labels, conf = adapter.predict_with_confidence(["m1", "m2"])
    assert labels.tolist() == ["Food", "Food"]
    assert np.allclose(conf, 0.8)


def test_sigmoid_clamps() -> None:
    vals = predictors.sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert np.all(vals >= 0.0)
    assert np.all(vals <= 1.0)
