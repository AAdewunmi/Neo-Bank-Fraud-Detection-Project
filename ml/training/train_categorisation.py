"""
Train a deterministic categorisation baseline (TF-IDF + Logistic Regression).

Usage:
  source .venv/bin/activate
  python ml/training/train_categorisation.py \
    --input data/sample_transactions.csv \
    --target_col category \
    --text_cols merchant description \
    --registry model_registry.json

Output:
- artefacts/cat_lr_<timestamp>.joblib
- model_registry.json updated (categorisation.latest -> version)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.training.utils import schema_hash, load_registry, save_registry


def build_pipeline() -> Pipeline:
    """
    Build a deterministic baseline pipeline.

    Returns:
        Sklearn Pipeline with TF-IDF vectorizer and Logistic Regression classifier.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=1.0,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,  # ensures solver behavior stays stable
                    solver="liblinear",
                ),
            ),
        ]
    )


def main(args: argparse.Namespace) -> None:
    """
    Train, evaluate, persist, and register a categorisation baseline.

    Args:
        args: Parsed CLI arguments.
    """
    df = pd.read_csv(args.input)

    # Hard requirement: target present for supervised training.
    if args.target_col not in df.columns:
        raise ValueError(f"Missing target column: {args.target_col}")

    df = df.dropna(subset=[args.target_col]).copy()

    # Deterministic text assembly: fixed column order and whitespace join.
    text = df[args.text_cols].astype(str).agg(" ".join, axis=1)
    y = df[args.target_col].astype(str)

    # Stratify keeps label proportions stable even on small samples.
    X_train, X_test, y_train, y_test = train_test_split(
        text,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro") if len(y_test) else 0.0

    artefacts_dir = Path("artefacts")
    artefacts_dir.mkdir(exist_ok=True)

    version = f"cat_lr_{pd.Timestamp.utcnow():%Y%m%d%H%M%S}"
    artefact_path = artefacts_dir / f"{version}.joblib"
    dump(pipeline, artefact_path)

    registry = load_registry(args.registry)
    registry["categorisation"][version] = {
        "artefact": str(artefact_path),
        "schema_hash": schema_hash(list(args.text_cols) + [args.target_col]),
        "text_cols": list(args.text_cols),
        "target_col": args.target_col,
        "metrics": {"macro_f1": float(macro_f1)},
    }
    registry["categorisation"]["latest"] = version
    save_registry(registry, args.registry)

    print(f"Saved artefact: {artefact_path}")
    print(f"Macro F1: {macro_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--target_col", required=True)
    parser.add_argument("--text_cols", nargs="+", required=True)
    parser.add_argument("--registry", default="model_registry.json")
    main(parser.parse_args())
