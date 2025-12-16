"""
Train an unsupervised fraud baseline (Isolation Forest) on numeric features.

Usage:
  python -m ml.training.train_fraud_baseline \
    --input data/sample_transactions.csv \
    --amount_col amount \
    --registry model_registry.json

Output:
- artefacts/fraud_if_<timestamp>.joblib
- model_registry.json updated (fraud.latest -> version)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.ensemble import IsolationForest

from ml.training.utils import load_registry, save_registry, schema_hash


def main(args: argparse.Namespace) -> None:
    """
    Train, persist, and register an Isolation Forest fraud baseline.

    Args:
        args: Parsed CLI arguments.
    """
    df = pd.read_csv(args.input)
    if args.amount_col not in df.columns:
        raise ValueError(f"Missing amount column: {args.amount_col}")

    # Defensive numeric coercion; Week 1 is about not crashing on messy inputs.
    X = pd.to_numeric(df[args.amount_col], errors="coerce").fillna(0.0).to_frame()

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
    )
    model.fit(X)

    artefacts_dir = Path("artefacts")
    artefacts_dir.mkdir(exist_ok=True)

    version = f"fraud_if_{pd.Timestamp.utcnow():%Y%m%d%H%M%S}"
    artefact_path = artefacts_dir / f"{version}.joblib"
    dump(model, artefact_path)

    registry = load_registry(args.registry)
    registry["fraud"][version] = {
        "artefact": str(artefact_path),
        "schema_hash": schema_hash([args.amount_col]),
        "features": [args.amount_col],
        "metrics": {"note": "unsupervised baseline; validate via threshold tests"},
    }
    registry["fraud"]["latest"] = version
    save_registry(registry, args.registry)

    print(f"Saved artefact: {artefact_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--amount_col", default="amount")
    parser.add_argument("--registry", default="model_registry.json")
    main(parser.parse_args())
