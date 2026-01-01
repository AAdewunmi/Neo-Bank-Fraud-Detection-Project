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
    --use_embeddings yes \
    --safe_threads yes \
    --fallback_tfidf yes \
    --registry model_registry.json
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

import lightgbm as lgb

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


class TfidfLinearCategoriser:
    """
    Simple TF-IDF + linear model predictor artefact.
    """

    def __init__(self, model: object, vectorizer: TfidfVectorizer) -> None:
        self.model = model
        self.vectorizer = vectorizer

    def predict_with_confidence(self, texts: list[str]) -> Tuple[np.ndarray, np.ndarray]:
        X = self.vectorizer.transform(list(texts))
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
    if str(getattr(args, "safe_threads", "yes")).strip().lower() == "yes":
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

    df = pd.read_csv(args.input).dropna(subset=[args.target_col])

    text_series = df[list(args.text_cols)].astype(str).agg(" ".join, axis=1)
    y = df[args.target_col].astype(str)

    X_train_text, X_test_text, y_train, y_test = _safe_train_test_split(
        text=text_series,
        y=y,
        test_size=0.2,
        seed=42,
    )

    use_embeddings = str(getattr(args, "use_embeddings", "yes")).strip().lower() == "yes"
    if use_embeddings:
        from ml.training.embeddings import MiniLMEncoder

        encoder = MiniLMEncoder(model_name=args.encoder_name, device="cpu")
        try:
            X_train = encoder.encode(X_train_text)
            X_test = encoder.encode(X_test_text)
        except Exception as exc:
            if str(getattr(args, "fallback_tfidf", "yes")).strip().lower() == "yes":
                print(
                    "[train_categorisation_embeddings] Encoder failed; "
                    "falling back to TF-IDF + logistic regression."
                )
                use_embeddings = False
            else:
                raise RuntimeError(
                    "Embedding encoding failed. If this is a native crash, try running with "
                    "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false."
                ) from exc

    embeddings_status = "enabled"
    if use_embeddings:
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
        predictor: object = EmbeddingsLightGBMCategoriser(
            model=clf,
            encoder_name=args.encoder_name,
        )
        model_type = "embeddings_lightgbm"
    else:
        embeddings_status = "fallback_tfidf" if getattr(args, "use_embeddings", "yes") == "yes" else "disabled"
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        predictor = TfidfLinearCategoriser(model=clf, vectorizer=vectorizer)
        model_type = "tfidf_logreg"

    macro_f1 = f1_score(y_test, y_pred, average="macro")

    artefacts = Path("artefacts")
    artefacts.mkdir(exist_ok=True)

    version = f"cat_minilm_lgbm_{pd.Timestamp.utcnow():%Y%m%d%H%M%S}"
    model_path = artefacts / f"{version}.joblib"

    dump(predictor, model_path)

    reg = load_registry(args.registry)
    reg.setdefault("categorisation", {})
    reg["categorisation"][version] = {
        "artefact": str(model_path),
        "schema_hash": schema_hash(list(args.text_cols) + [args.target_col]),
        "text_cols": list(args.text_cols),
        "target_col": args.target_col,
        "metrics": {"macro_f1": float(macro_f1), "embeddings_status": embeddings_status},
        "type": model_type,
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
    parser.add_argument("--use_embeddings", default="yes", choices=["yes", "no"])
    parser.add_argument("--safe_threads", default="yes", choices=["yes", "no"])
    parser.add_argument("--fallback_tfidf", default="yes", choices=["yes", "no"])
    parser.add_argument("--registry", default="model_registry.json")
    main(parser.parse_args())
