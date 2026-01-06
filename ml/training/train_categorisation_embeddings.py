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
    --synthetic no \
    --use_embeddings no \
    --safe_threads yes \
    --fallback_tfidf yes \
    --split_mode time \
    --timestamp_col timestamp \
    --group_col customer_id \
    --registry model_registry.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

try:
    import lightgbm as lgb
except Exception:
    lgb = None

from ml.training.utils import load_registry, save_registry, schema_hash
from ml.training.splits import split_train_test
from ml.training.model_card import write_model_card

sys.modules.setdefault("ml.training.train_categorisation_embeddings", sys.modules[__name__])

try:
    from ml.training.embeddings import MiniLMEncoder as _MiniLMEncoder
except Exception:
    _MiniLMEncoder = None

MiniLMEncoder = _MiniLMEncoder


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


EmbeddingsLightGBMCategoriser.__module__ = "ml.training.train_categorisation_embeddings"
TfidfLinearCategoriser.__module__ = "ml.training.train_categorisation_embeddings"


def _write_model_card(
    artefacts_dir: Path,
    version: str,
    entry: dict[str, object],
    label_meta: dict[str, object],
    split_meta: dict[str, object],
) -> Path:
    """
    Write a minimal model card markdown file for audit and portfolio review.
    """
    path = artefacts_dir / f"{version}_model_card.md"
    lines = []
    lines.append(f"# Categorisation Model Card {version}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Model type: {entry.get('type')}")
    macro_f1 = entry.get("metrics", {}).get("macro_f1", 0.0)
    lines.append(f"- Metric: Macro F1 {macro_f1:.4f}")
    lines.append(f"- Text cols: {', '.join(entry.get('text_cols', []))}")
    lines.append(f"- Target col: {entry.get('target_col')}")
    lines.append("")
    lines.append("## Labels")
    label_mode = label_meta.get("label_mode", "unknown")
    lines.append(f"- Label mode: {label_mode}")
    if bool(entry.get("synthetic", False)):
        lines.append("- Synthetic labels are used in this run.")
        lines.append(
            "- This model is for offline demonstration only and "
            "must not be used for real categorisation decisions."
        )
    lines.append("")
    lines.append("## Split")
    lines.append(f"- Split type: {split_meta.get('split_type')}")
    if split_meta.get("split_type") == "time":
        lines.append(f"- Timestamp col: {split_meta.get('timestamp_col')}")
        lines.append(f"- Leakage guard: {split_meta.get('leakage_guard')}")
        lines.append(f"- Overlap groups removed: {split_meta.get('overlap_groups')}")
    if split_meta.get("split_type") == "group":
        lines.append(f"- Group col: {split_meta.get('group_col')}")
        lines.append(f"- Groups: {split_meta.get('n_groups')}")
    lines.append(f"- n_train: {split_meta.get('n_train')}")
    lines.append(f"- n_test: {split_meta.get('n_test')}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(args: argparse.Namespace) -> None:
    """
    Train embeddings plus LightGBM categoriser and persist artefacts plus registry entry.
    """
    if str(getattr(args, "safe_threads", "yes")).strip().lower() == "yes":
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

    df = pd.read_csv(args.input).dropna(subset=[args.target_col])
    synthetic_flag = str(getattr(args, "synthetic", "no")).strip().lower() == "yes"
    label_meta = {
        "label_mode": "synthetic" if synthetic_flag else "real",
        "label_col": args.target_col,
    }

    text_series = df[list(args.text_cols)].astype(str).agg(" ".join, axis=1)
    y = df[args.target_col].astype(str)

    split = split_train_test(
        df=df,
        y=y.values,
        test_size=float(getattr(args, "test_size", 0.2)),
        seed=int(getattr(args, "seed", 42)),
        split_mode=str(getattr(args, "split_mode", "random")),
        timestamp_col=str(getattr(args, "timestamp_col", "timestamp")),
        group_col=str(getattr(args, "group_col", "customer_id")),
        leakage_guard=str(getattr(args, "leakage_guard", "yes")).strip().lower() == "yes",
    )
    if split.meta.get("split_type") == "random_no_stratify":
        print(
            "[train_categorisation_embeddings] Disabling stratify: "
            "not enough samples per class for the chosen test_size."
        )

    X_train_text = text_series.iloc[split.train_idx]
    X_test_text = text_series.iloc[split.test_idx]
    y_train = y.iloc[split.train_idx]
    y_test = y.iloc[split.test_idx]

    use_embeddings = str(getattr(args, "use_embeddings", "yes")).strip().lower() == "yes"
    if use_embeddings:
        if MiniLMEncoder is None:
            raise RuntimeError(
                "MiniLMEncoder is unavailable. Install embeddings dependencies or "
                "run with --use_embeddings no."
            )
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
        global lgb
        if lgb is None:
            try:
                import importlib

                lgb = importlib.import_module("lightgbm")
            except ImportError as exc:
                raise ImportError(
                    "lightgbm is required for embedding-based categorisation. "
                    "Install it or run with --use_embeddings no."
                ) from exc

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
        embeddings_status = (
            "fallback_tfidf"
            if getattr(args, "use_embeddings", "yes") == "yes"
            else "disabled"
        )
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

    version_prefix = (
        "cat_minilm_lgbm"
        if model_type == "embeddings_lightgbm"
        else "cat_tfidf_logreg"
    )
    if synthetic_flag:
        version_prefix = f"{version_prefix}_synth"
    version = f"{version_prefix}_{pd.Timestamp.utcnow():%Y%m%d%H%M%S}"
    model_path = artefacts / f"{version}.joblib"

    dump(predictor, model_path)

    card_path = write_model_card(str(model_path), args.input, {"macro_f1": float(macro_f1)})
    print("Model card:", card_path)

    reg = load_registry(args.registry)
    section = "categorisation_synthetic" if synthetic_flag else "categorisation"
    reg.setdefault(section, {})
    entry = {
        "artefact": str(model_path),
        "schema_hash": schema_hash(list(args.text_cols) + [args.target_col]),
        "text_cols": list(args.text_cols),
        "target_col": args.target_col,
        "metrics": {
            "macro_f1": float(macro_f1),
            "embeddings_status": embeddings_status,
            "label_meta": label_meta,
            "split_meta": split.meta,
        },
        "type": model_type,
        "synthetic": bool(synthetic_flag),
        "synthetic_note": "synthetic_labels" if synthetic_flag else "real_labels",
        "label_mode": label_meta.get("label_mode"),
        "dataset_path": str(args.input),
        "card_json": str(card_path),
    }
    model_card_path = _write_model_card(
        artefacts_dir=artefacts,
        version=version,
        entry=entry,
        label_meta=label_meta,
        split_meta=split.meta,
    )
    entry["model_card"] = str(model_card_path)
    reg[section][version] = entry
    reg[section]["latest"] = version
    save_registry(reg, args.registry)

    print(f"Saved: {model_path}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Registry section: {section}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--target_col", required=True)
    parser.add_argument("--text_cols", nargs="+", required=True)
    parser.add_argument(
        "--encoder_name",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--synthetic", default="no", choices=["yes", "no"])
    parser.add_argument("--use_embeddings", default="yes", choices=["yes", "no"])
    parser.add_argument("--safe_threads", default="yes", choices=["yes", "no"])
    parser.add_argument("--fallback_tfidf", default="yes", choices=["yes", "no"])
    parser.add_argument("--split_mode", default="random", choices=["time", "group", "random"])
    parser.add_argument("--timestamp_col", default="timestamp")
    parser.add_argument("--group_col", default="customer_id")
    parser.add_argument("--leakage_guard", default="yes", choices=["yes", "no"])
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--registry", default="model_registry.json")
    main(parser.parse_args())
