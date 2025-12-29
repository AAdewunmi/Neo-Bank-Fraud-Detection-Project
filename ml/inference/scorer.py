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

    def score(self, df: pd.DataFrame, categorisation_version: Optional[str] = None) -> pd.DataFrame:
        """
        Score a dataframe of transactions.

        Returns a new dataframe with category and category_confidence fields added.
        """
        out = df.copy()

        cat_entry, cat_model = self._load_model("categorisation", categorisation_version)
        cat_cols = cat_entry["text_cols"]
        text_series = out[cat_cols].astype(str).agg(" ".join, axis=1)

        # Two modes:
        # - sklearn pipeline: model.predict(text_series) and predict_proba(text_series)
        # - embeddings_lightgbm artefact: {"model": LGBM, "encoder_name": "..."}
        if isinstance(cat_model, dict) and "model" in cat_model and "encoder_name" in cat_model:
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

        out["category_pred"] = categories
        out["category_confidence"] = cat_conf
        return out
