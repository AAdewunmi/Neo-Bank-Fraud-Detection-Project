# ml/inference/scorer.py
"""
Inference scorer for categorisation and fraud.

Assumes model_registry.json provides:
- categorisation.latest and categorisation[version]
- fraud.latest and fraud[version] (if fraud scoring is enabled)

This file adds:
- support for embeddings_lightgbm categorisation artefacts
- in-process cache for sentence-transformers encoders
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from joblib import load

from ml.training.utils import load_registry


class Scorer:
    """
    Loads registered models and produces per-transaction scores.

    The scorer caches loaded artefacts and any embedding encoders so repeated calls stay fast.
    """

    def __init__(self, registry_path: str = "model_registry.json") -> None:
        self._registry_path = registry_path
        self._reg = load_registry(registry_path)
        # Backwards-compatible alias for tests/older code.
        self._registry = self._reg
        self._artefact_cache: dict[str, Any] = {}
        self._encoder_cache: dict[str, Any] = {}

    def _load_artefact(self, artefact_path: str) -> Any:
        """
        Load a joblib artefact with caching.
        """
        if artefact_path in self._artefact_cache:
            return self._artefact_cache[artefact_path]

        obj = load(artefact_path)
        self._artefact_cache[artefact_path] = obj
        return obj

    def _get_encoder(self, encoder_name: str):
        """
        Cache sentence-transformers encoders by name to avoid repeated heavyweight loads.
        """
        if encoder_name in self._encoder_cache:
            return self._encoder_cache[encoder_name]

        from sentence_transformers import SentenceTransformer

        enc = SentenceTransformer(encoder_name, device="cpu")
        self._encoder_cache[encoder_name] = enc
        return enc

    def _load(self, section: str, version: Optional[str] = None) -> Any:
        """
        Backwards-compatible loader used in older tests.
        """
        _, artefact = self._load_model(section, version)
        return artefact

    def _load_model(
        self,
        section: str,
        version: Optional[str] = None,
    ) -> tuple[dict[str, Any], Any]:
        """
        Load a model entry and its artefact.
        """
        if section not in self._reg:
            raise KeyError(f"Registry missing section: {section}")

        selected = version or self._reg[section].get("latest")
        if not selected:
            raise KeyError(f"No latest model set for section: {section}")

        entry = self._reg[section][selected]
        artefact = self._load_artefact(entry["artefact"])
        return entry, artefact

    def _get_entry(
        self,
        section: str,
        version: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Fetch a registry entry for a section + version.
        """
        if section not in self._reg:
            raise KeyError(f"Registry missing section: {section}")

        selected = version or self._reg[section].get("latest")
        if not selected:
            raise KeyError(f"No latest model set for section: {section}")

        return self._reg[section][selected]

    def score(
        self,
        df: pd.DataFrame,
        threshold: float,
        categorisation_version: Optional[str] = None,
        fraud_version: Optional[str] = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Score a dataframe of transactions.

        Returns a tuple of (scored dataframe, diagnostics dict).
        """
        out = df.copy()

        cat_entry = self._get_entry("categorisation", categorisation_version)
        cat_model = self._load("categorisation", categorisation_version)
        cat_cols = cat_entry["text_cols"]
        text_series = out[cat_cols].astype(str).agg(" ".join, axis=1)

        # Two modes:
        # - sklearn pipeline: model.predict(text_series) and predict_proba(text_series)
        # - embeddings_lightgbm artefact: {"model": LGBM, "encoder_name": "..."}
        if (
            isinstance(cat_model, dict)
            and "model" in cat_model
            and "encoder_name" in cat_model
        ):
            encoder = self._get_encoder(cat_model["encoder_name"])
            X = np.asarray(
                encoder.encode(
                    list(text_series),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            )
            categories = cat_model["model"].predict(X)
            if hasattr(cat_model["model"], "predict_proba"):
                cat_conf = cat_model["model"].predict_proba(X).max(axis=1)
            else:
                cat_conf = np.ones(len(out))
        else:
            categories = cat_model.predict(text_series)
            if hasattr(cat_model, "predict_proba"):
                cat_conf = cat_model.predict_proba(text_series).max(axis=1)
            else:
                cat_conf = np.ones(len(out))

        out["category"] = categories
        out["category_pred"] = categories
        out["category_confidence"] = cat_conf

        fraud_entry = self._get_entry("fraud", fraud_version)
        fraud_model = self._load("fraud", fraud_version)
        fraud_features = fraud_entry.get("features", ["amount"])
        fraud_X = (
            out[fraud_features]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )

        if hasattr(fraud_model, "decision_function"):
            raw_scores = fraud_model.decision_function(fraud_X)
        elif hasattr(fraud_model, "score_samples"):
            raw_scores = fraud_model.score_samples(fraud_X)
        else:
            raise AttributeError("Fraud model missing decision_function/score_samples")

        raw_scores = np.asarray(raw_scores, dtype=float)
        fraud_risk = 1.0 / (1.0 + np.exp(raw_scores))
        fraud_risk = np.clip(fraud_risk, 0.0, 1.0)
        flagged = fraud_risk >= threshold

        out["fraud_risk"] = fraud_risk
        out["flagged"] = flagged

        diags = {
            "n": int(len(out)),
            "threshold": float(threshold),
            "pct_flagged": float(np.mean(flagged)) if len(out) else 0.0,
        }

        return out, diags
