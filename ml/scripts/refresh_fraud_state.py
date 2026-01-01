# ml/scripts/refresh_fraud_state.py
"""
Refresh per-customer fraud feature state for production drift.

Usage:
  PYTHONPATH=. python -m ml.scripts.refresh_fraud_state \
    --input data/recent_transactions.csv \
    --registry model_registry.json \
    --amount_col amount \
    --customer_col customer_id \
    --update_registry yes
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ml.fraud_features import build_customer_state
from ml.training.utils import load_registry, save_registry


def _resolve_state_path(
    registry_path: str,
    explicit_path: str | None,
    section: str | None,
) -> tuple[str, str]:
    """
    Resolve the state path, returning (path, version).
    """
    if explicit_path:
        return explicit_path, ""

    registry = load_registry(registry_path)
    sections = [section] if section else ["fraud", "fraud_synthetic"]
    for sec in sections:
        if sec not in registry or not registry[sec].get("latest"):
            continue
        version = registry[sec]["latest"]
        entry = registry[sec][version]
        state_path = entry.get("feature_state_path")
        if state_path:
            return str(state_path), version
        if entry.get("type") == "supervised_xgb":
            default_path = str(Path("artefacts") / f"{version}_customer_state.csv")
            return default_path, version
    raise KeyError(
        "Registry entry missing feature_state_path for fraud model. "
        "Provide --state_path to create a new state store."
    )


def main(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input)
    if args.amount_col not in df.columns:
        raise ValueError(f"Missing amount column: {args.amount_col}")
    if args.customer_col not in df.columns:
        raise ValueError(f"Missing customer column: {args.customer_col}")

    state_path, version = _resolve_state_path(
        args.registry,
        args.state_path or None,
        args.section or None,
    )

    df = df.rename(
        columns={args.amount_col: "amount", args.customer_col: "customer_id"}
    )
    state_df = build_customer_state(df)

    out_path = Path(state_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state_df.to_csv(out_path, index=False)

    if str(args.update_registry).strip().lower() == "yes" and version:
        registry = load_registry(args.registry)
        target_section = args.section if args.section else "fraud"
        if target_section not in registry:
            raise KeyError(f"Registry missing section: {target_section}")
        if version not in registry[target_section]:
            raise KeyError(f"Registry missing version in {target_section}: {version}")
        registry[target_section][version]["feature_state_path"] = str(out_path)
        save_registry(registry, args.registry)

    print(f"Saved state store: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--registry", default="model_registry.json")
    parser.add_argument("--state_path", default="")
    parser.add_argument("--amount_col", default="amount")
    parser.add_argument("--customer_col", default="customer_id")
    parser.add_argument("--update_registry", default="no", choices=["yes", "no"])
    parser.add_argument("--section", default="")
    main(parser.parse_args())
