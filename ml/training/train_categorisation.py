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
    --synthetic no \
    --registry model_registry.json

Output:
- artefacts/cat_lr_<timestamp>.joblib
- model_registry.json updated (categorisation.latest -> version)
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict


import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.training.model_card import write_model_card
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


def _write_model_card(
    artefacts_dir: Path,
    version: str,
    entry: Dict[str, Any],
    label_meta: Dict[str, Any],
    split_meta: Dict[str, Any],
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
    lines.append(f"- Test size: {split_meta.get('test_size')}")
    lines.append(f"- Stratified: {split_meta.get('stratified')}")
    lines.append(f"- n_samples: {split_meta.get('n_samples')}")
    lines.append(f"- n_classes: {split_meta.get('n_classes')}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(args: argparse.Namespace) -> None:
    """
    Train, evaluate, persist, and register a categorisation baseline.

    Args:
        args: Parsed CLI arguments.
    """
    df = pd.read_csv(args.input)
    synthetic_flag = str(getattr(args, "synthetic", "no")).strip().lower() == "yes"
    label_meta = {
        "label_mode": "synthetic" if synthetic_flag else "real",
        "label_col": args.target_col,
    }

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

    version_prefix = "cat_lr_synth" if synthetic_flag else "cat_lr"
    version = f"{version_prefix}_{pd.Timestamp.utcnow():%Y%m%d%H%M%S}"
    artefact_path = artefacts_dir / f"{version}.joblib"
    dump(pipeline, artefact_path)
    card_json_path = write_model_card(
        str(artefact_path),
        args.input,
        {"macro_f1": float(macro_f1)},
    )

    registry = load_registry(args.registry)
    section = "categorisation_synthetic" if synthetic_flag else "categorisation"
    registry.setdefault(section, {})
    entry = {
        "artefact": str(artefact_path),
        "schema_hash": schema_hash(list(args.text_cols) + [args.target_col]),
        "text_cols": list(args.text_cols),
        "target_col": args.target_col,
        "metrics": {"macro_f1": float(macro_f1), "label_meta": label_meta},
        "split": {
            "test_size": float(test_size),
            "stratified": bool(can_stratify),
            "n_samples": int(len(y)),
            "n_classes": int(y.nunique()),
        },
        "type": "tfidf_logreg",
        "synthetic": bool(synthetic_flag),
        "synthetic_note": "synthetic_labels" if synthetic_flag else "real_labels",
        "label_mode": label_meta.get("label_mode"),
        "dataset_path": str(args.input),
        "card_json": str(card_json_path),
    }
    model_card_path = _write_model_card(
        artefacts_dir=artefacts_dir,
        version=version,
        entry=entry,
        label_meta=label_meta,
        split_meta=entry["split"],
    )
    entry["model_card"] = str(model_card_path)
    registry[section][version] = entry
    registry[section]["latest"] = version
    save_registry(registry, args.registry)

    print(f"Saved artefact: {artefact_path}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Split: test_size={test_size:.4f} stratified={can_stratify}")
    print(f"Registry section: {section}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--target_col", required=True)
    parser.add_argument("--text_cols", nargs="+", required=True)
    parser.add_argument("--registry", default="model_registry.json")
    parser.add_argument("--synthetic", default="no", choices=["yes", "no"])
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.25,
        help="Test split fraction (may be increased automatically for stratified tiny datasets).",
    )
    main(parser.parse_args())
