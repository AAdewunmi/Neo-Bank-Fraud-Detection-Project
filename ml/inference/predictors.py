"""
Inference-time predictor adapters.

Goals
- Cache transformer encoders so they load once per process.
- Unify categorisation artefacts behind one predictor contract.
- Keep imports lightweight so unit tests can run without ML dependencies installed.

Design
- Training can persist either a plain sklearn pipeline, a predictor wrapper,
or a legacy dict artefact.
- Scorer calls as_text_categoriser(...) and then predict_with_confidence(...) without branching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Sequence, Tuple
import os

import numpy as np


class TextCategoriser(Protocol):
    """
    Unified predictor contract for transaction categorisation.
    """

    def predict_with_confidence(self, texts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict labels and return per-row confidence values in [0, 1].
        """
        raise NotImplementedError


_ENCODER_CACHE: Dict[str, Any] = {}


def _get_sentence_transformer(model_name: str) -> Any:
    """
    Load a SentenceTransformer once per process and cache it.

    This function imports sentence_transformers lazily so unit tests can run
    without ML dependencies installed.
    """
    if model_name in _ENCODER_CACHE:
        return _ENCODER_CACHE[model_name]

    if "TOKENIZERS_PARALLELISM" not in os.environ:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from sentence_transformers import SentenceTransformer  # lazy import

    encoder = SentenceTransformer(model_name)
    _ENCODER_CACHE[model_name] = encoder
    return encoder


@dataclass
class SklearnTextPipelineCategoriser(TextCategoriser):
    """
    Adapter for sklearn text pipelines.

    Expects the wrapped object to expose predict(...) and optionally predict_proba(...).
    """

    pipeline: Any

    def predict_with_confidence(self, texts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        labels = np.asarray(self.pipeline.predict(list(texts)))
        if hasattr(self.pipeline, "predict_proba"):
            proba = np.asarray(self.pipeline.predict_proba(list(texts)))
            conf = np.max(proba, axis=1)
            return labels, conf
        return labels, np.ones(len(labels), dtype=float)


@dataclass
class EmbeddingsLightGBMCategoriser(TextCategoriser):
    """
    Predictor for embeddings plus LightGBM categorisation.

    Stores only encoder_name and model.
    Encoder loads via a process-level cache.
    """

    model: Any
    encoder_name: str

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        encoder = _get_sentence_transformer(self.encoder_name)
        vec = encoder.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vec)

    def predict_with_confidence(self, texts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        X = self._embed(texts)
        labels = np.asarray(self.model.predict(X))
        if hasattr(self.model, "predict_proba"):
            proba = np.asarray(self.model.predict_proba(X))
            conf = np.max(proba, axis=1)
            return labels, conf
        return labels, np.ones(len(labels), dtype=float)


def as_text_categoriser(artefact: Any) -> TextCategoriser:
    """
    Convert a loaded artefact into a TextCategoriser without changing scorer logic.

    Supported inputs
    - Objects already implementing predict_with_confidence(...)
    - sklearn pipelines with predict(...) and optional predict_proba(...)
    - Legacy dict artefacts from Day 1 embeddings trainer
      dict keys expected model and encoder_name
    """
    if hasattr(artefact, "predict_with_confidence"):
        return artefact

    if isinstance(artefact, dict) and "model" in artefact and "encoder_name" in artefact:
        return EmbeddingsLightGBMCategoriser(
            model=artefact["model"],
            encoder_name=str(artefact["encoder_name"]),
        )

    return SklearnTextPipelineCategoriser(pipeline=artefact)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid for mapping margins to [0, 1].
    """
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))
