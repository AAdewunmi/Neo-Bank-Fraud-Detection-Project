# ml/inference/scorer.py
"""
Inference scorer for categorisation and fraud.

Assumes model_registry.json provides:
- categorisation.latest and categorisation[version]
- fraud.latest and fraud[version] (optional but recommended)

Supported categorisation artefacts:
- sklearn text pipeline (expects predict(texts) and optional predict_proba(texts))
- legacy embeddings artefact saved as a dict {"model": ..., "encoder_name": ...}
- unified predictor objects implementing predict_with_confidence(texts)

Supported fraud artefacts:
- supervised models with predict_proba(X) (uses class-1 probability)
- anomaly/supervised models with decision_function(X) or score_samples(X)
- models with predict(X) that already return values in [0, 1]

Performance and UX notes:
- SentenceTransformer encoder loads are cached per Python process so the first call pays the
  download/init cost, and later calls reuse the same encoder.
- Joblib artefact loads are cached per Scorer instance (test-safe).

Environment controls:
- HF_HOME
    If not set, this module defaults it to .cache/huggingface under the current working directory.
    This keeps downloads local to the project and avoids repeated network fetches after
    the first run.
- LEDGERGUARD_EMBED_BATCH_SIZE
    Integer batch size passed to SentenceTransformer.encode. Default 64.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence, Tuple
from pathlib import Path
import os

import numpy as np
import pandas as pd
from joblib import load

from ml.training.utils import load_registry


_ENCODER_CACHE: dict[str, Any] = {}


def _sigmoid_from_raw_scores(raw_scores: np.ndarray) -> np.ndarray:
    """
    Map raw scores to [0, 1] using a numerically stable sigmoid(-x).

    This matches the original LedgerGuard mapping:
      fraud_risk = 1 / (1 + exp(raw_scores))
    """
    x = np.asarray(raw_scores, dtype=float)
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(x))


class TextCategoriser(Protocol):
    """Unified predictor contract for transaction categorisation."""

    def predict_with_confidence(self, texts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict labels and per-row confidence values in [0, 1].
        """
        raise NotImplementedError


class _SklearnTextPipelineCategoriser:
    """
    Adapter for sklearn-style text pipelines.

    Expects:
    - predict(texts)
    - optional predict_proba(texts)
    """

    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline

    def predict_with_confidence(self, texts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        labels = np.asarray(self._pipeline.predict(list(texts)))
        if hasattr(self._pipeline, "predict_proba"):
            proba = np.asarray(self._pipeline.predict_proba(list(texts)))
            conf = np.max(proba, axis=1)
            return labels, conf
        return labels, np.ones(len(labels), dtype=float)


class _EmbeddingsLightGBMCategoriser:
    """
    Predictor for embeddings plus LightGBM categorisation.

    Takes a LightGBM-like model and an encoder_name and uses a cached encoder loader.
    """

    def __init__(self, model: Any, encoder_name: str, get_encoder: Any) -> None:
        self._model = model
        self._encoder_name = encoder_name
        self._get_encoder = get_encoder

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        encoder = self._get_encoder(self._encoder_name)

        try:
            batch_size = int(os.environ.get("LEDGERGUARD_EMBED_BATCH_SIZE", "64"))
        except ValueError:
            batch_size = 64
        batch_size = max(1, min(batch_size, 1024))

        vec = encoder.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return np.asarray(vec)

    def predict_with_confidence(self, texts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        X = self._encode_texts(texts)
        labels = np.asarray(self._model.predict(X))
        if hasattr(self._model, "predict_proba"):
            proba = np.asarray(self._model.predict_proba(X))
            conf = np.max(proba, axis=1)
            return labels, conf
        return labels, np.ones(len(labels), dtype=float)


def _as_text_categoriser(artefact: Any, get_encoder: Any) -> TextCategoriser:
    """
    Convert a loaded artefact into a TextCategoriser.

    Supported inputs:
    - objects already implementing predict_with_confidence(...)
    - legacy dict embeddings artefact: {"model": ..., "encoder_name": ...}
    - objects that expose attributes model and encoder_name
    - sklearn pipelines with predict(...) and optional predict_proba(...)
    """
    if hasattr(artefact, "predict_with_confidence"):
        return artefact  # type: ignore[return-value]

    if isinstance(artefact, dict) and "model" in artefact and "encoder_name" in artefact:
        return _EmbeddingsLightGBMCategoriser(
            model=artefact["model"],
            encoder_name=str(artefact["encoder_name"]),
            get_encoder=get_encoder,
        )

    if hasattr(artefact, "model") and hasattr(artefact, "encoder_name"):
        return _EmbeddingsLightGBMCategoriser(
            model=getattr(artefact, "model"),
            encoder_name=str(getattr(artefact, "encoder_name")),
            get_encoder=get_encoder,
        )

    return _SklearnTextPipelineCategoriser(pipeline=artefact)


class Scorer:
    """
    Loads registered models and produces per-transaction scores.

    The scorer caches loaded joblib artefacts per instance (test-safe) and caches embedding encoders
    per process (fast repeated scoring in the web app).
    """

    def __init__(self, registry_path: str = "model_registry.json") -> None:
        self._registry_path = registry_path
        self._reg = load_registry(registry_path)

        # Backwards-compatible alias used in tests/older code.
        self._registry = self._reg

        self._artefact_cache: dict[str, Any] = {}

    def _load_artefact(self, artefact_path: str) -> Any:
        """Load a joblib artefact with per-instance caching."""
        if artefact_path in self._artefact_cache:
            return self._artefact_cache[artefact_path]

        obj = load(artefact_path)
        self._artefact_cache[artefact_path] = obj
        return obj

    def _get_encoder(self, encoder_name: str) -> Any:
        """
        Cache sentence-transformers encoders by name to avoid repeated heavyweight loads.

        Import is delayed so the fast unit-test job can run without ML dependencies installed.
        """
        if encoder_name in _ENCODER_CACHE:
            return _ENCODER_CACHE[encoder_name]

        # Keep HF cache local to the project unless the caller configured it already.
        if "HF_HOME" not in os.environ:
            cache_dir = Path(".cache") / "huggingface"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(cache_dir)

        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        from sentence_transformers import SentenceTransformer

        enc = SentenceTransformer(encoder_name, device="cpu")
        _ENCODER_CACHE[encoder_name] = enc
        return enc

    def _load_model(
        self, section: str, version: Optional[str] = None
    ) -> tuple[dict[str, Any], Any]:
        """Load a model entry and its artefact."""
        if section not in self._reg:
            raise KeyError(f"Registry missing section: {section}")

        selected = version or self._reg[section].get("latest")
        if not selected:
            raise KeyError(f"No latest model set for section: {section}")

        entry = self._reg[section][selected]
        artefact = self._load_artefact(entry["artefact"])
        return entry, artefact

    def _get_entry(self, section: str, version: Optional[str] = None) -> dict[str, Any]:
        """Fetch a registry entry for a section and version."""
        if section not in self._reg:
            raise KeyError(f"Registry missing section: {section}")

        selected = version or self._reg[section].get("latest")
        if not selected:
            raise KeyError(f"No latest model set for section: {section}")

        return self._reg[section][selected]

    def score(
        self,
        df: pd.DataFrame,
        threshold: float = 0.7,
        categorisation_version: Optional[str] = None,
        fraud_version: Optional[str] = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Score a dataframe of transactions.

        Returns a tuple of (scored dataframe, diagnostics dict).
        """
        out = df.copy()

        # -----------------------------
        # Categorisation
        # -----------------------------
        cat_entry = self._get_entry("categorisation", categorisation_version)
        _, cat_artefact = self._load_model("categorisation", categorisation_version)

        cat_cols = cat_entry.get("text_cols", ["merchant", "description"])
        text_series = out[list(cat_cols)].astype(str).agg(" ".join, axis=1).tolist()

        categoriser = _as_text_categoriser(cat_artefact, get_encoder=self._get_encoder)
        categories, cat_conf = categoriser.predict_with_confidence(text_series)

        out["category"] = categories
        out["category_pred"] = categories
        out["category_confidence"] = np.asarray(cat_conf, dtype=float)

        # -----------------------------
        # Fraud scoring
        # -----------------------------
        flagged = np.zeros(len(out), dtype=bool)
        fraud_risk = np.zeros(len(out), dtype=float)

        fraud_section = self._reg.get("fraud", {})
        has_fraud = bool(fraud_section) and bool(fraud_section.get("latest"))

        if has_fraud:
            fraud_entry = self._get_entry("fraud", fraud_version)
            _, fraud_model = self._load_model("fraud", fraud_version)

            fraud_features = fraud_entry.get("features", ["amount"])
            fraud_X = (
                out[list(fraud_features)]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
                .astype(float)
                .values
            )

            if hasattr(fraud_model, "predict_proba"):
                fraud_risk = np.asarray(fraud_model.predict_proba(fraud_X)[:, 1], dtype=float)
            elif hasattr(fraud_model, "decision_function"):
                raw_scores = np.asarray(fraud_model.decision_function(fraud_X), dtype=float)
                fraud_risk = _sigmoid_from_raw_scores(raw_scores)
            elif hasattr(fraud_model, "score_samples"):
                raw_scores = np.asarray(fraud_model.score_samples(fraud_X), dtype=float)
                fraud_risk = _sigmoid_from_raw_scores(raw_scores)
            elif hasattr(fraud_model, "predict"):
                pred = np.asarray(fraud_model.predict(fraud_X), dtype=float)
                fraud_risk = np.clip(pred, 0.0, 1.0)
            else:
                raise AttributeError(
                    "Fraud model missing predict_proba/decision_function/score_samples/predict"
                )

            fraud_risk = np.clip(fraud_risk, 0.0, 1.0)
            flagged = fraud_risk >= float(threshold)

        out["fraud_risk"] = fraud_risk
        out["fraud_flag"] = flagged
        out["flagged"] = flagged

        diags = {
            "n": int(len(out)),
            "threshold": float(threshold),
            "pct_flagged": float(np.mean(flagged)) if len(out) else 0.0,
            "categorisation_version": categorisation_version
            or self._reg.get("categorisation", {}).get("latest"),
            "fraud_version": fraud_version or self._reg.get("fraud", {}).get("latest"),
        }

        return out, diags
