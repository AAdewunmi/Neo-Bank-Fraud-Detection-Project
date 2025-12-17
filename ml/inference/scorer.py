"""
Unified scoring for categorisation + fraud baselines.

Week 1 guarantee:
- stable output contract for the dashboard layer
- threshold monotonicity for fraud flagging
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from joblib import load


class Scorer:
    """
    Load baseline models from a registry and score transactions.

    The scorer caches loaded models in-memory to avoid repeated disk reads.
    """

    def __init__(self, registry_path: str = "model_registry.json") -> None:
        """
        Initialise a scorer with a registry path.

        Args:
            registry_path: Path to model registry JSON.
        """
        self.registry_path = registry_path
        self._registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
        self._cache: Dict[str, Any] = {}

    def _load(self, task: str, version: str | None) -> Any:
        """
        Load a model artefact for a task/version from registry.

        Args:
            task: "categorisation" or "fraud"
            version: specific version or None for latest

        Returns:
            Loaded model object.
        """
        resolved = version or self._registry[task]["latest"]
        cache_key = f"{task}:{resolved}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        artefact_path = self._registry[task][resolved]["artefact"]
        model = load(artefact_path)
        self._cache[cache_key] = model
        return model

    # def score(self, df: pd.DataFrame, threshold: float = 0.65) -> Tuple[pd.DataFrame,
    #                                                                     Dict[str, Any]]:
    #     """
    #     Score a dataframe and return outputs + diagnostics.

    #     Required input columns:
    #       - timestamp, amount, customer_id, merchant, description

    #     Args:
    #         df: Input transactions dataframe.
    #         threshold: Fraud flag threshold in [0, 1] (higher -> fewer flags).

    #     Returns:
    #         (scored_df, diagnostics) where scored_df includes:
    #           category, category_confidence, fraud_risk, flagged
    #     """
    #     cat_version = self._registry["categorisation"]["latest"]
    #     frd_version = self._registry["fraud"]["latest"]

    #     cat_model = self._load("categorisation", cat_version)
    #     frd_model = self._load("fraud", frd_version)

    #     text_cols = self._registry["categorisation"][cat_version]["text_cols"]
    #     text = df[text_cols].astype(str).agg(" ".join, axis=1)

    #     categories = cat_model.predict(text)

    #     if hasattr(cat_model, "predict_proba"):
    #         confidence = cat_model.predict_proba(text).max(axis=1)
    #     else:
    #         # Baseline fallback: if no probabilities, treat as confident.
    #         confidence = np.ones(len(df), dtype=float)

    #     # IsolationForest: decision_function higher = more normal.
    #     # Convert to risk where higher = more anomalous.
    #     amt = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0).to_frame()
    #     raw = frd_model.decision_function(amt)

    #     raw_max = float(np.max(raw))
    #     raw_min = float(np.min(raw))
    #     denom = (raw_max - raw_min) + 1e-9
    #     risk = (raw_max - raw) / denom

    #     flagged = risk >= threshold

    #     out = df.copy()
    #     out["category"] = categories
    #     out["category_confidence"] = confidence
    #     out["fraud_risk"] = np.round(risk, 4)
    #     out["flagged"] = flagged

    #     diagnostics = {
    #         "n": int(len(df)),
    #         "threshold": float(threshold),
    #         "pct_flagged": float(np.mean(flagged)) if len(df) else 0.0,
    #     }
    #     return out, diagnostics
