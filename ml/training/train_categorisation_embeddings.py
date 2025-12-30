# ml/training/train_categorisation_embeddings.py
"""
Train a categorisation model using MiniLM embeddings plus LightGBM.

Merged behaviour
- Production-first artefact interface: persists a predictor object that
implements predict_with_confidence.
- schema_hash derived from real inputs: text_cols plus target_col.
- Robust splitting: uses stratify when possible, falls back cleanly when data is small.
- Small-dataset guardrails for LightGBM: relaxes min_data_in_leaf/min_data_in_bin when needed.
- Deterministic text concatenation: uses text_cols order provided on CLI.

Usage
  PYTHONPATH=. python -m ml.training.train_categorisation_embeddings \
    --input data/sample_transactions.csv \
    --target_col category \
    --text_cols merchant description \
    --encoder_name sentence-transformers/all-MiniLM-L6-v2 \
    --registry model_registry.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

import lightgbm as lgb

from ml.training.embeddings import MiniLMEncoder
from ml.training.utils import load_registry, save_registry, schema_hash


class EmbeddingsLightGBMCategoriser:
    """
    Unified categorisation predictor artefact.

    This object is intended to be joblib-persisted and loaded at inference.
    It exposes a stable method predict_with_confidence so scorer does not branch.

    Notes
    - The encoder is loaded lazily and cached per process.
    - sentence-transformers is imported lazily so non-ML unit tests can run without it installed.
    """

    _ENCODER_CACHE: dict[str, object] = {}

    def __init__(self, model: object, encoder_name: str) -> None:
        self.model = model
        self.encoder_name = encoder_name

    def _get_encoder(self) -> object:
        if self.encoder_name in self._ENCODER_CACHE:
            return self._ENCODER_CACHE[self.encoder_name]

        import os

        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        from sentence_transformers import SentenceTransformer

        enc = SentenceTransformer(self.encoder_name, device="cpu")
        self._ENCODER_CACHE[self.encoder_name] = enc
        return enc

    def predict_with_confidence(self, texts: list[str]) -> Tuple[np.ndarray, np.ndarray]:
        encoder = self._get_encoder()
        X = np.asarray(
            encoder.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        )

        labels = np.asarray(self.model.predict(X))
        if hasattr(self.model, "predict_proba"):
            proba = np.asarray(self.model.predict_proba(X))
            conf = np.max(proba, axis=1)
            return labels, conf
        return labels, np.ones(len(labels), dtype=float)


def _safe_train_test_split(
    text: pd.Series,
    y: pd.Series,
    test_size: float,
    seed: int,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Attempt stratified split, fall back to non-stratified when classes are too small.

    This keeps the trainer runnable on tiny demo datasets without crashing.
    """
    test_count = max(1, int(round(len(y) * test_size)))
    class_counts = y.value_counts()
    can_stratify = (
        len(class_counts) <= test_count
        and (class_counts.min() if len(class_counts) else 0) >= 2
    )

    if not can_stratify:
        print(
            "[train_categorisation_embeddings] Disabling stratify: "
            "not enough samples per class for the chosen test_size."
        )

    return train_test_split(
        text,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y if can_stratify else None,
    )


def main(args: argparse.Namespace) -> None:
    """
    Train embeddings plus LightGBM categoriser and persist artefacts plus registry entry.
    """
    df = pd.read_csv(args.input).dropna(subset=[args.target_col])

    text_series = df[list(args.text_cols)].astype(str).agg(" ".join, axis=1)
    y = df[args.target_col].astype(str)

    X_train_text, X_test_text, y_train, y_test = _safe_train_test_split(
        text=text_series,
        y=y,
        test_size=0.2,
        seed=42,
    )

    encoder = MiniLMEncoder(model_name=args.encoder_name, device="cpu")
    X_train = encoder.encode(X_train_text)
    X_test = encoder.encode(X_test_text)

    lgbm_params = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
    }

    if len(X_train) < 50:
        lgbm_params.update(
            {
                "min_data_in_leaf": 1,
                "min_data_in_bin": 1,
            }
        )
        print(
            "[train_categorisation_embeddings] Small dataset detected; "
            "relaxing LightGBM min_data_in_leaf/min_data_in_bin."
        )

    clf = lgb.LGBMClassifier(**lgbm_params)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    artefacts = Path("artefacts")
    artefacts.mkdir(exist_ok=True)

    version = f"cat_minilm_lgbm_{pd.Timestamp.utcnow():%Y%m%d%H%M%S}"
    model_path = artefacts / f"{version}.joblib"

    predictor = EmbeddingsLightGBMCategoriser(
        model=clf,
        encoder_name=args.encoder_name,
    )
    dump(predictor, model_path)

    reg = load_registry(args.registry)
    reg.setdefault("categorisation", {})
    reg["categorisation"][version] = {
        "artefact": str(model_path),
        "schema_hash": schema_hash(list(args.text_cols) + [args.target_col]),
        "text_cols": list(args.text_cols),
        "target_col": args.target_col,
        "metrics": {"macro_f1": float(macro_f1)},
        "type": "embeddings_lightgbm",
    }
    reg["categorisation"]["latest"] = version
    save_registry(reg, args.registry)

    print(f"Saved: {model_path}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--target_col", required=True)
    parser.add_argument("--text_cols", nargs="+", required=True)
    parser.add_argument(
        "--encoder_name",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--registry", default="model_registry.json")
    main(parser.parse_args())
