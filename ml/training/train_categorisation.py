"""
Train a deterministic categorisation baseline (TF-IDF + Logistic Regression).

This trainer is designed to be boringly reliable in Week 1:
- deterministic model config
- schema hash + registry updates
- robust split logic for tiny datasets (avoids common stratify failures)

Usage:
 PYTHONPATH=. python -m ml.training.train_categorisation \
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
import math
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
                    random_state=42,
                    solver="liblinear",
                ),
            ),
        ]
    )


def _can_stratify(y: pd.Series) -> bool:
    """
    Determine whether stratified splitting is valid.

    Stratified splitting requires:
    - at least 2 samples per class (so train and test can each get one)
    - at least 2 classes

    Args:
        y: Target labels.

    Returns:
        True if stratification is safe, else False.
    """
    if y.nunique() <= 1:
        return False
    counts = y.value_counts()
    return int(counts.min()) >= 2


def _choose_test_fraction_for_stratify(
    n_samples: int,
    n_classes: int,
    requested_test_size: float,
) -> float:
    """
    Choose a test fraction that satisfies stratified splitting constraints.

    scikit-learn requires:
    - number of test samples >= number of classes
    - number of train samples >= number of classes

    For small datasets, the default 0.25 can yield too few test rows.

    Args:
        n_samples: Total row count.
        n_classes: Number of unique classes.
        requested_test_size: Requested test fraction.

    Returns:
        A float test_size fraction that satisfies constraints where possible.
    """
    # Start with the requested fraction.
    test_frac = float(requested_test_size)

    # Compute the minimum test count to cover all classes at least once.
    # ceil is conservative and matches how sklearn converts fractions to counts.
    min_test_count = n_classes
    min_train_count = n_classes

    # If the dataset is too small overall, caller should disable stratification.
    # Example: 2 classes needs at least 4 rows to place >=1 per class in both splits.
    if n_samples < (min_test_count + min_train_count):
        return test_frac

    # Ensure test split has at least one sample per class.
    n_test = max(int(math.ceil(test_frac * n_samples)), min_test_count)

    # Ensure train split also has at least one sample per class.
    n_train = n_samples - n_test
    if n_train < min_train_count:
        n_test = n_samples - min_train_count
        n_train = n_samples - n_test

    # Convert back to fraction.
    # This may increase test_size above the requested value for small datasets,
    # which is preferred to crashing during Week 1 training.
    return n_test / float(n_samples)


def main(args: argparse.Namespace) -> None:
    """
    Train, evaluate, persist, and register a categorisation baseline.

    Args:
        args: Parsed CLI arguments.
    """
    df = pd.read_csv(args.input)

    if args.target_col not in df.columns:
        raise ValueError(f"Missing target column: {args.target_col}")

    for col in args.text_cols:
        if col not in df.columns:
            raise ValueError(f"Missing text column: {col}")

    # Drop rows missing the target and normalise labels to avoid accidental extra classes
    # (e.g., 'Transport' vs 'Transport ').
    df = df.dropna(subset=[args.target_col]).copy()
    y = df[args.target_col].astype(str).str.strip()

    # Deterministic text assembly: fixed column order and whitespace join.
    # Also strip to keep text stable.
    text = df[list(args.text_cols)].astype(str).apply(lambda s: s.str.strip()).agg(" ".join, axis=1)

    pipeline = build_pipeline()

    # Decide split strategy.
    can_stratify = _can_stratify(y)

    # If stratifying, choose a test fraction that can hold at least one sample per class.
    test_size = float(args.test_size)
    if can_stratify:
        n_samples = int(len(y))
        n_classes = int(y.nunique())
        adjusted = _choose_test_fraction_for_stratify(n_samples, n_classes, test_size)

        # Verify that the adjusted split will still leave enough rows per class.
        # If not, fall back to unstratified split rather than crashing.
        n_test = int(math.ceil(adjusted * n_samples))
        n_train = n_samples - n_test
        if n_test < n_classes or n_train < n_classes:
            can_stratify = False
        else:
            test_size = adjusted

    X_train, X_test, y_train, y_test = train_test_split(
        text,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y if can_stratify else None,
    )

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
        "split": {
            "test_size": float(test_size),
            "stratified": bool(can_stratify),
            "n_samples": int(len(y)),
            "n_classes": int(y.nunique()),
        },
    }
    registry["categorisation"]["latest"] = version
    save_registry(registry, args.registry)

    print(f"Saved artefact: {artefact_path}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Split: test_size={test_size:.4f} stratified={can_stratify}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--target_col", required=True)
    parser.add_argument("--text_cols", nargs="+", required=True)
    parser.add_argument("--registry", default="model_registry.json")
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.25,
        help="Test split fraction (may be increased automatically for stratified tiny datasets).",
    )
    main(parser.parse_args())
