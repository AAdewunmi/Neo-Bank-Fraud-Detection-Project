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
from sklearn.model_selection import train_test_split

import xgboost as xgb

from ml.fraud_features import FEATURE_ORDER, build_customer_state, compute_train_features
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


def _require_columns(df: pd.DataFrame, columns: List[str]) -> None:
    """
    Ensure all required columns are present.
    """
    missing = [name for name in columns if name not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}")


def _safe_random_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Attempt stratified split, fall back to non-stratified when classes are too small.
    """
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )
        meta = {"split_type": "random_stratified"}
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )
        meta = {"split_type": "random_no_stratify"}
    return X_tr, X_te, y_tr, y_te, meta


def _build_threshold_table(
    y_true: np.ndarray,
    proba: np.ndarray,
    base_thresholds: List[float],
) -> pd.DataFrame:
    """
    Build a threshold table with precision/recall/f1 for reporting.
    """
    grid = np.linspace(0.0, 1.0, 21)
    thresholds = np.unique(np.concatenate([np.asarray(base_thresholds), grid]))
    rows = []
    positives = float((y_true == 1).sum())
    for thr in thresholds:
        preds = proba >= thr
        tp = float(((preds == 1) & (y_true == 1)).sum())
        fp = float(((preds == 1) & (y_true == 0)).sum())
        precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
        recall = 0.0 if positives == 0 else tp / positives
        denom = precision + recall
        f1 = 0.0 if denom == 0 else 2 * (precision * recall) / denom
        rows.append(
            {
                "threshold": float(thr),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    return pd.DataFrame(rows)


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

    synthetic_mode = args.synthetic.strip().lower()
    test_size = float(getattr(args, "test_size", 0.25))
    split_mode = getattr(args, "split_mode", "random").strip().lower()
    timestamp_col = getattr(args, "timestamp_col", "timestamp")
    group_col = getattr(args, "group_col", "customer_id")

    label_source = ""
    if synthetic_mode == "yes":
        y, label_meta = make_synthetic_labels(
            df=df,
            amount_col=args.amount_col,
            positive_quantile=args.synthetic_quantile,
        )
        synthetic_flag = True
        label_source = f"synthetic_top_quantile_{args.synthetic_quantile}"
    elif synthetic_mode == "auto":
        if args.label_col in df.columns:
            y = (
                pd.to_numeric(df[args.label_col], errors="coerce")
                .fillna(0)
                .astype(int)
                .values
            )
            if len(np.unique(y)) >= 2:
                label_meta = {
                    "label_mode": "real",
                    "label_col": args.label_col,
                    "positive_rate": float(np.mean(y)) if len(y) else 0.0,
                }
                synthetic_flag = False
                label_source = f"real_label_col_{args.label_col}"
            else:
                y, label_meta = make_synthetic_labels(
                    df=df,
                    amount_col=args.amount_col,
                    positive_quantile=args.synthetic_quantile,
                )
                synthetic_flag = True
                label_source = f"synthetic_top_quantile_{args.synthetic_quantile}"
        else:
            y, label_meta = make_synthetic_labels(
                df=df,
                amount_col=args.amount_col,
                positive_quantile=args.synthetic_quantile,
            )
            synthetic_flag = True
            label_source = f"synthetic_top_quantile_{args.synthetic_quantile}"
    else:
        _require_columns(df, [args.label_col])
        y = (
            pd.to_numeric(df[args.label_col], errors="coerce")
            .fillna(0)
            .astype(int)
            .values
        )
        label_meta = {
            "label_mode": "real",
            "label_col": args.label_col,
            "positive_rate": float(np.mean(y)) if len(y) else 0.0,
        }
        synthetic_flag = False
        label_source = f"real_label_col_{args.label_col}"

    if len(np.unique(y)) < 2:
        raise ValueError("Need both positive and negative labels to train and evaluate.")

    _require_columns(df, [args.amount_col])
    if timestamp_col not in df.columns:
        df[timestamp_col] = pd.NaT
    if group_col not in df.columns:
        df[group_col] = "unknown"

    df_feat = df.rename(columns={args.amount_col: "amount"})
    feat = compute_train_features(df_feat)
    X = feat.values
    feature_names = FEATURE_ORDER[:]

    if split_mode == "time":
        ts_valid = pd.to_datetime(df[timestamp_col], errors="coerce").notna().any()
        if not ts_valid:
            split_mode = "random"

    if split_mode in {"time", "group"}:
        split = split_train_test(
            df=df,
            y=y,
            test_size=test_size,
            seed=args.seed,
            split_mode=split_mode,
            timestamp_col=timestamp_col,
            group_col=group_col,
            leakage_guard=True,
        )
        X_tr = X[split.train_idx]
        y_tr = y[split.train_idx]
        X_te = X[split.test_idx]
        y_te = y[split.test_idx]
        split_meta = split.meta
    else:
        X_tr, X_te, y_tr, y_te, split_meta = _safe_random_split(
            X=X,
            y=y,
            test_size=test_size,
            seed=args.seed,
        )
        split_meta.update(
            {
                "n_total": int(len(df)),
                "n_train": int(len(X_tr)),
                "n_test": int(len(X_te)),
            }
        )

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

    artefacts = Path(getattr(args, "artefacts_dir", "artefacts"))
    reports = Path(getattr(args, "reports_dir", "reports"))
    artefacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    version_prefix = "fraud_xgb_synth" if synthetic_flag else "fraud_xgb"
    version = f"{version_prefix}_{pd.Timestamp.utcnow():%Y%m%d%H%M%S}"
    model_path = artefacts / f"{version}.joblib"
    state_path = artefacts / f"{version}_customer_state.csv"
    metrics_path = reports / f"{version}_metrics.json"
    thresholds_path = reports / f"{version}_thresholds.csv"

    dump(clf, model_path)

    state_df = build_customer_state(df_feat)
    state_df.to_csv(state_path, index=False)

    thresholds_df = _build_threshold_table(
        y_true=y_te,
        proba=proba,
        base_thresholds=thresholds_list,
    )
    thresholds_df.to_csv(thresholds_path, index=False)

    metrics: Dict[str, Any] = {
        "average_precision": ap,
        "label_source": label_source,
        "label_mode": label_meta.get("label_mode"),
        "synthetic": bool(synthetic_flag),
        "label_meta": label_meta,
        "split_meta": split_meta,
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
        "threshold_table_csv": str(thresholds_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    reg = load_registry(args.registry)
    section = "fraud_synthetic" if synthetic_mode == "yes" else "fraud"
    reg.setdefault(section, {})

    entry = {
        "artefact": str(model_path),
        "metrics_path": str(metrics_path),
        "schema_hash": schema_hash(feature_names),
        "features": feature_names,
        "metrics": {"average_precision": ap, "label_source": label_source},
        "type": "supervised_xgb",
        "risk_mode": "predict_proba",
        "feature_state_path": str(state_path),
        "feature_state_type": "customer_amount_stats_v1",
        "synthetic": bool(synthetic_flag),
        "label_mode": label_meta.get("label_mode"),
        "split_type": split_meta.get("split_type"),
    }

    model_card_path = _write_model_card(
        artefacts_dir=artefacts,
        version=version,
        entry=entry,
        label_meta=label_meta,
        split_meta=split_meta,
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
    print(f"Split: {split_meta.get('split_type')} train {len(X_tr)} test {len(X_te)}")
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
    p.add_argument("--artefacts_dir", default="artefacts")
    p.add_argument("--reports_dir", default="reports")
    main(p.parse_args())
