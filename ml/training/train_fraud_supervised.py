"""
Train a supervised fraud classifier using XGBoost.

This trainer supports two label modes
- real label mode using label_col
- synthetic label mode for lab demonstration only

Trust and reproducibility improvements
- Stronger splits via time split and group split to reduce leakage
- Synthetic label safety rails via explicit naming, metadata, and model card output
- PR AUC reporting via Average Precision plus a threshold table for Ops controls

Usage examples

Synthetic lab run using a time split
PYTHONPATH=. python -m ml.training.train_fraud_supervised \
  --input data/synthetic_transactions_large.csv \
  --label_col is_fraud \
  --synthetic yes \
  --amount_col amount \
  --split_mode time \
  --timestamp_col timestamp \
  --group_col customer_id \
  --registry model_registry.json

Real label run using a group split
PYTHONPATH=. python -m ml.training.train_fraud_supervised \
  --input data/real_transactions.csv \
  --label_col is_fraud \
  --synthetic no \
  --features amount velocity_24h is_international is_weekend hour \
  --split_mode group \
  --group_col customer_id \
  --registry model_registry.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import average_precision_score, precision_recall_curve

import xgboost as xgb

from ml.training.splits import split_train_test
from ml.training.utils import load_registry, save_registry, schema_hash


def make_synthetic_labels(
    df: pd.DataFrame,
    amount_col: str,
    positive_quantile: float = 0.95,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Create a synthetic fraud label.

    Rule
    - Transactions at or above the chosen amount quantile are labelled as fraud.
    - This is for demonstration only and must be treated as synthetic.

    Returns y and metadata describing the rule.
    """
    amounts = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0).astype(float)
    threshold_amount = float(amounts.quantile(positive_quantile))
    y = (amounts >= threshold_amount).astype(int).values

    meta = {
        "label_mode": "synthetic_top_amount_quantile",
        "positive_quantile": float(positive_quantile),
        "threshold_amount": float(threshold_amount),
        "positive_rate": float(np.mean(y)) if len(y) else 0.0,
    }
    return y, meta


def _resolve_features(df: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
    """
    Build a numeric feature matrix from the requested feature names.
    """
    X = (
        df[feature_names]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(float)
        .values
    )
    return X


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

    label_mode = label_meta.get("label_mode", "unknown")
    synthetic = bool(entry.get("synthetic", False))

    lines: List[str] = []
    lines.append(f"# Fraud Model Card {version}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Model type: {entry.get('type')}")
    ap_metric = entry.get('metrics', {}).get('average_precision', 0.0)
    lines.append(f"- Metric: Average Precision (PR AUC) {ap_metric:.4f}")
    lines.append(f"- Features: {', '.join(entry.get('features', []))}")
    lines.append(f"- Split: {split_meta.get('split_type')}")
    lines.append("")
    lines.append("## Labels")
    lines.append(f"- Label mode: {label_mode}")
    if synthetic:
        lines.append("- Synthetic labels are used in this run.")
        lines.append(
            "- This model is for offline demonstration only and "
            "must not be used for real fraud decisions."
        )
        quantile = label_meta.get('positive_quantile')
        lines.append(f"- Synthetic rule: top quantile by amount {quantile}")
        lines.append(f"- Amount threshold: {label_meta.get('threshold_amount')}")
    lines.append("")
    lines.append("## Leakage controls")
    if split_meta.get("split_type") == "time":
        lines.append(f"- Time split using {split_meta.get('timestamp_col')}")
        lines.append(f"- Leakage guard {split_meta.get('leakage_guard')}")
        lines.append(f"- Overlap groups removed {split_meta.get('overlap_groups')}")
    if split_meta.get("split_type") == "group":
        lines.append(f"- Group split using {split_meta.get('group_col')}")
        lines.append(f"- Groups {split_meta.get('n_groups')}")
    lines.append("")
    lines.append("## Risk mapping note")
    lines.append("- Risk uses model predict_proba outputs in [0, 1].")
    lines.append("- Calibration is not applied yet, thresholding uses the "
                 "saved precision recall trade off table.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input)

    if args.synthetic.strip().lower() == "yes":
        y, label_meta = make_synthetic_labels(
            df=df,
            amount_col=args.amount_col,
            positive_quantile=args.synthetic_quantile,
        )
        synthetic_flag = True
    else:
        if args.label_col not in df.columns:
            raise ValueError(f"Missing label column {args.label_col}")
        y = pd.to_numeric(df[args.label_col], errors="coerce").fillna(0).astype(int).values
        label_meta = {
            "label_mode": "real",
            "label_col": args.label_col,
            "positive_rate": float(np.mean(y)) if len(y) else 0.0,
        }
        synthetic_flag = False

    if len(np.unique(y)) < 2:
        raise ValueError("Need both positive and negative labels to train and evaluate.")

    feature_names = list(args.features) if args.features else [args.amount_col]
    for name in feature_names:
        if name not in df.columns:
            raise ValueError(f"Missing feature column {name}")

    split = split_train_test(
        df=df,
        y=y,
        test_size=args.test_size,
        seed=args.seed,
        split_mode=args.split_mode,
        timestamp_col=args.timestamp_col,
        group_col=args.group_col,
        leakage_guard=True,
    )

    X = _resolve_features(df, feature_names)

    X_tr = X[split.train_idx]
    y_tr = y[split.train_idx]
    X_te = X[split.test_idx]
    y_te = y[split.test_idx]

    pos_rate_train = float(np.mean(y_tr)) if len(y_tr) else 0.0
    pos_rate_test = float(np.mean(y_te)) if len(y_te) else 0.0

    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        random_state=args.seed,
        scale_pos_weight=(float((y_tr == 0).sum()) / max(1.0, float((y_tr == 1).sum()))),
    )
    clf.fit(X_tr, y_tr)

    proba = np.asarray(clf.predict_proba(X_te)[:, 1], dtype=float)
    ap = float(average_precision_score(y_te, proba))

    precision_curve, recall_curve, thresholds = precision_recall_curve(y_te, proba)

    precision_at_thr = precision_curve[1:].tolist()
    recall_at_thr = recall_curve[1:].tolist()
    thresholds_list = thresholds.tolist()

    artefacts = Path("artefacts")
    artefacts.mkdir(exist_ok=True)

    version_prefix = "fraud_xgb_synth" if synthetic_flag else "fraud_xgb"
    version = f"{version_prefix}_{pd.Timestamp.utcnow():%Y%m%d%H%M%S}"
    model_path = artefacts / f"{version}.joblib"
    metrics_path = artefacts / f"{version}_metrics.json"

    dump(clf, model_path)

    metrics: Dict[str, Any] = {
        "average_precision": ap,
        "label_mode": label_meta.get("label_mode"),
        "synthetic": bool(synthetic_flag),
        "label_meta": label_meta,
        "split_meta": split.meta,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "positive_rate_train": pos_rate_train,
        "positive_rate_test": pos_rate_test,
        "features": feature_names,
        "thresholds": thresholds_list,
        "precision": precision_at_thr,
        "recall": recall_at_thr,
        "precision_curve": precision_curve.tolist(),
        "recall_curve": recall_curve.tolist(),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    reg = load_registry(args.registry)
    section = "fraud_synthetic" if synthetic_flag else "fraud"
    reg.setdefault(section, {})

    entry = {
        "artefact": str(model_path),
        "metrics_path": str(metrics_path),
        "schema_hash": schema_hash(feature_names),
        "features": feature_names,
        "metrics": {"average_precision": ap},
        "type": "supervised_xgb",
        "synthetic": bool(synthetic_flag),
        "label_mode": label_meta.get("label_mode"),
        "split_type": split.meta.get("split_type"),
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
    print(f"AP (PR AUC): {ap:.4f}")
    print(f"Metrics: {metrics_path}")
    print(f"Model card: {model_card_path}")
    print(f"Registry section: {section}")
    print(f"Split: {split.meta.get('split_type')} train {len(X_tr)} test {len(X_te)}")
    print(f"Positive rate train {pos_rate_train:.4f} test {pos_rate_test:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--label_col", default="is_fraud")
    p.add_argument("--amount_col", default="amount")
    p.add_argument("--synthetic", default="yes")
    p.add_argument("--synthetic_quantile", type=float, default=0.95)
    p.add_argument("--features", nargs="*", default=None)
    p.add_argument("--split_mode", default="time", choices=["time", "group", "random"])
    p.add_argument("--timestamp_col", default="timestamp")
    p.add_argument("--group_col", default="customer_id")
    p.add_argument("--test_size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--registry", default="model_registry.json")
    main(p.parse_args())
