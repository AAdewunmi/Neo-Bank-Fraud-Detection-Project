# ml/training/train_categorisation_embeddings.py
"""
Train a categorisation model using MiniLM embeddings + LightGBM.

Usage:
  python ml/training/train_categorisation_embeddings.py \
    --input data/sample_transactions.csv \
    --target_col category \
    --text_cols merchant description \
    --registry model_registry.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

import lightgbm as lgb

from ml.training.embeddings import MiniLMEncoder
from ml.training.utils import load_registry, save_registry, schema_hash


def main(args: argparse.Namespace) -> None:
    """
    Train embeddings + LightGBM categoriser and persist artefacts plus registry entry.
    """
    df = pd.read_csv(args.input).dropna(subset=[args.target_col])

    # Combine text columns deterministically in the order provided.
    text_series = df[list(args.text_cols)].astype(str).agg(" ".join, axis=1)
    y = df[args.target_col].astype(str)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        text_series,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    encoder = MiniLMEncoder(model_name=args.encoder_name, device="cpu")
    X_train = encoder.encode(X_train_text)
    X_test = encoder.encode(X_test_text)

    clf = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    artefacts = Path("artefacts")
    artefacts.mkdir(exist_ok=True)

    version = f"cat_minilm_lgbm_{pd.Timestamp.utcnow():%Y%m%d%H%M%S}"
    model_path = artefacts / f"{version}.joblib"

    dump(
        {
            "model": clf,
            "encoder_name": args.encoder_name,
        },
        model_path,
    )

    reg = load_registry(args.registry)
    reg.setdefault("categorisation", {})
    reg["categorisation"][version] = {
        "artefact": str(model_path),
        "schema_hash": schema_hash(
            list(args.text_cols) + [args.target_col],
        ),
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
