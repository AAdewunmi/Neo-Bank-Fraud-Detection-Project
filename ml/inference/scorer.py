# ml/inference/scorer.py
"""
Inference scorer for categorisation and fraud.

Improved:
- Reuses encoder across requests (real process-level cache, not per-instance)
- Adds hard embedding row cap (default 2000)
- Handles interrupted downloads gracefully with a clear error
- Keeps all prior test and dashboard compatibility intact

Env knobs:
  LEDGERGUARD_MAX_EMBED_ROWS   (default 2000)
  LEDGERGUARD_EMBED_BATCH_SIZE (default 64)
"""

from __future__ import annotations
import os
from typing import Any, Optional, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import load
from ml.training.utils import load_registry


_ENCODER_CACHE: dict[str, Any] = {}


def _sigmoid_from_raw_scores(raw_scores: np.ndarray) -> np.ndarray:
    """Map raw model scores to [0, 1] range using sigmoid(-x)."""
    x = np.asarray(raw_scores, dtype=float)
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(x))


class TextCategoriser(Protocol):
    def predict_with_confidence(self, texts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class _SklearnTextPipelineCategoriser:
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
    def __init__(self, model: Any, encoder_name: str, get_encoder: Any) -> None:
        self._model = model
        self._encoder_name = encoder_name
        self._get_encoder = get_encoder

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        max_rows = int(os.environ.get("LEDGERGUARD_MAX_EMBED_ROWS", "2000"))
        if len(texts) > max_rows:
            raise ValueError(
                f"⚠️ Only {max_rows} rows allowed for dashboard embeddings scoring. "
                "Run offline scoring for larger CSVs."
            )

        batch_size = int(os.environ.get("LEDGERGUARD_EMBED_BATCH_SIZE", "64"))
        encoder = self._get_encoder(self._encoder_name)

        try:
            vec = encoder.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
            )
        except TypeError:
            try:
                vec = encoder.encode(
                    list(texts),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            except Exception as inner_exc:
                raise RuntimeError(
                    f"Embedding model '{self._encoder_name}' failed to encode text: {inner_exc}"
                ) from inner_exc
        except Exception as exc:
            raise RuntimeError(
                f"Embedding model '{self._encoder_name}' failed to encode text: {exc}"
            ) from exc

        return np.asarray(vec)

    def predict_with_confidence(self, texts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        X = self._embed(texts)
        labels = np.asarray(self._model.predict(X))
        if hasattr(self._model, "predict_proba"):
            proba = np.asarray(self._model.predict_proba(X))
            conf = np.max(proba, axis=1)
            return labels, conf
        return labels, np.ones(len(labels), dtype=float)


def _as_text_categoriser(artefact: Any, get_encoder: Any) -> TextCategoriser:
    """Convert a loaded artefact into a categoriser with a unified API."""
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
    """Loads models and produces per-transaction category and fraud scores."""

    def __init__(self, registry_path: str = "model_registry.json") -> None:
        self._registry_path = registry_path
        self._reg = load_registry(registry_path)
        self._registry = self._reg
        self._artefact_cache: dict[str, Any] = {}

    def _load_artefact(self, artefact_path: str) -> Any:
        if artefact_path in self._artefact_cache:
            return self._artefact_cache[artefact_path]
        obj = load(artefact_path)
        self._artefact_cache[artefact_path] = obj
        return obj

    def _get_encoder(self, encoder_name: str) -> Any:
        """Cache SentenceTransformer per process, not per scorer instance."""
        if encoder_name in _ENCODER_CACHE:
            return _ENCODER_CACHE[encoder_name]

        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        from sentence_transformers import SentenceTransformer

        print(f"[LedgerGuard] Loading encoder '{encoder_name}' (first call only)...")
        enc = SentenceTransformer(encoder_name, device="cpu")
        _ENCODER_CACHE[encoder_name] = enc
        return enc

    def _load_model(
        self, section: str, version: Optional[str] = None
    ) -> tuple[dict[str, Any], Any]:
        if section not in self._reg:
            raise KeyError(f"Registry missing section: {section}")
        selected = version or self._reg[section].get("latest")
        if not selected:
            raise KeyError(f"No latest model set for section: {section}")
        entry = self._reg[section][selected]
        if "artefact" in entry:
            artefact = self._load_artefact(entry["artefact"])
        else:
            artefact = self._load(section, selected)
        return entry, artefact

    def _get_entry(self, section: str, version: Optional[str] = None) -> dict[str, Any]:
        if section not in self._reg:
            raise KeyError(f"Registry missing section: {section}")
        selected = version or self._reg[section].get("latest")
        if not selected:
            raise KeyError(f"No latest model set for section: {section}")
        return self._reg[section][selected]

    def _load(self, task: str, version: Optional[str]) -> Any:
        entry = self._get_entry(task, version)
        if "artefact" not in entry:
            raise KeyError(f"Registry entry missing artefact for {task}:{version}")
        return self._load_artefact(entry["artefact"])

    def score(
        self,
        df: pd.DataFrame,
        threshold: float = 0.7,
        categorisation_version: Optional[str] = None,
        fraud_version: Optional[str] = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        out = df.copy()

        cat_entry, cat_model = self._load_model("categorisation", categorisation_version)
        cat_cols = cat_entry.get("text_cols", ["merchant", "description"])
        text_series = out[list(cat_cols)].astype(str).agg(" ".join, axis=1).tolist()

        categoriser = _as_text_categoriser(cat_model, get_encoder=self._get_encoder)
        categories, cat_conf = categoriser.predict_with_confidence(text_series)

        out["category"] = categories
        out["category_pred"] = categories
        out["category_confidence"] = np.asarray(cat_conf, dtype=float)

        flagged = np.zeros(len(out), dtype=bool)
        fraud_risk = np.zeros(len(out), dtype=float)

        if "fraud" in self._reg and self._reg["fraud"].get("latest"):
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
            else:
                raise AttributeError(
                    "Fraud model missing predict_proba/decision_function/score_samples"
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
            "flagged_key": "flagged",
        }

        return out, diags
