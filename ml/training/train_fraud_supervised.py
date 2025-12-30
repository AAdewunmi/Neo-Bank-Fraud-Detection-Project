"""
Train a supervised fraud classifier using XGBoost.

Behaviour
- If labels exist, trains supervised and reports PR-AUC as Average Precision.
- If labels are missing or synthetic mode is enabled, runs a clearly labelled
synthetic-label experiment.

Outputs
- Versioned model artefact in artefacts_dir
- Metrics JSON in reports_dir
- Threshold table CSV in reports_dir

Usage examples

Supervised
  PYTHONPATH=. python -m ml.training.train_fraud_supervised \
    --input data/synthetic_transactions_large.csv \
    --label_col is_fraud \
    --synthetic no \
    --registry model_registry.json

Synthetic demo
  PYTHONPATH=. python -m ml.training.train_fraud_supervised \
    --input data/sample_transactions.csv \
    --synthetic yes \
    --registry model_registry.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split

from ml.training.utils import load_registry, save_registry, schema_hash


DEFAULT_FEATURES: List[str] = [
    "amount",
    "is_international",
    "hour",
    "is_weekend",
    "amount_bucket",
    "velocity_24h",
]


def make_synthetic_labels(
    df: pd.DataFrame,
    amount_col: str,
    quantile: float,
    seed: int,
) -> np.ndarray:
    """
    Create a synthetic fraud label for demonstration.

    Definition
    - Top quantile amounts are positives.
    - Adds a small amount of seed-controlled noise so PR curves are not
      perfectly stepped.
    """
    rng = np.random.default_rng(seed)
    thresh = float(df[amount_col].quantile(quantile))
    base = (df[amount_col].astype(float) >= thresh).astype(int).to_numpy()
    flip = rng.random(len(base)) < 0.005
    base[flip] = 1 - base[flip]
    return base


def _require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def main(args: argparse.Namespace) -> None:
    try:
        import xgboost as xgb
    except Exception as exc:
        raise RuntimeError(
            "xgboost is required for this lab. Install it in requirements/ml.txt."
        ) from exc

    df = pd.read_csv(args.input)

    features = args.features or DEFAULT_FEATURES
    _require_columns(df, features)
    _require_columns(df, [args.amount_col])

    synthetic_mode = args.synthetic.lower()

    label_source: str
    if synthetic_mode == "yes":
        y = make_synthetic_labels(
            df,
            args.amount_col,
            args.synthetic_quantile,
            seed=args.seed,
        )
        label_source = f"synthetic_top_quantile_{args.synthetic_quantile}"
    elif synthetic_mode == "no":
        _require_columns(df, [args.label_col])
        y = df[args.label_col].astype(int).to_numpy()
        label_source = f"supervised_{args.label_col}"
    else:
        if args.label_col in df.columns and df[args.label_col].notna().any():
            y = df[args.label_col].astype(int).to_numpy()
            label_source = f"supervised_{args.label_col}"
        else:
            y = make_synthetic_labels(
                df,
                args.amount_col,
                args.synthetic_quantile,
                seed=args.seed,
            )
            label_source = f"synthetic_top_quantile_{args.synthetic_quantile}"

    X = df[features].astype(float).to_numpy()

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=args.seed,
            stratify=y,
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=args.seed,
            stratify=None,
        )

    pos = max(1, int((y_tr == 1).sum()))
    neg = max(1, int((y_tr == 0).sum()))
    scale_pos_weight = float(neg / pos)

    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        random_state=args.seed,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
    )
    clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_te)[:, 1]
    ap = float(average_precision_score(y_te, proba))

    precision, recall, thresholds = precision_recall_curve(y_te, proba)

    pr_table = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision[1:],
            "recall": recall[1:],
        }
    )
    pr_table["f1"] = (2.0 * pr_table["precision"] * pr_table["recall"]) / (
        pr_table["precision"] + pr_table["recall"] + 1e-12
    )

    artefacts_dir = Path(args.artefacts_dir)
    reports_dir = Path(args.reports_dir)
    artefacts_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    version = f"fraud_xgb_{pd.Timestamp.utcnow():%Y%m%d%H%M%S}"
    model_path = artefacts_dir / f"{version}.joblib"
    dump(clf, model_path)

    metrics = {
        "average_precision": ap,
        "label_source": label_source,
        "features": features,
        "threshold_table_csv": str(reports_dir / f"{version}_thresholds.csv"),
        "metrics_version": version,
    }
    metrics_path = reports_dir / f"{version}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    thresholds_csv = reports_dir / f"{version}_thresholds.csv"
    pr_table.to_csv(thresholds_csv, index=False)

    reg = load_registry(args.registry)
    reg["fraud"][version] = {
        "artefact": str(model_path),
        "schema_hash": schema_hash(features),
        "features": features,
        "metrics": {"average_precision": ap, "label_source": label_source},
        "type": "supervised_xgb",
    }
    reg["fraud"]["latest"] = version
    save_registry(reg, args.registry)

    print("Saved:", model_path)
    print("AP (PR-AUC):", round(ap, 4))
    print("Threshold table:", thresholds_csv)
    print("Metrics JSON:", metrics_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--label_col", default="is_fraud")
    p.add_argument("--amount_col", default="amount")
    p.add_argument("--synthetic", default="auto", choices=["auto", "yes", "no"])
    p.add_argument("--synthetic_quantile", type=float, default=0.95)
    p.add_argument("--features", nargs="+", default=None)
    p.add_argument("--registry", default="model_registry.json")
    p.add_argument("--artefacts_dir", default="artefacts")
    p.add_argument("--reports_dir", default="reports")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
