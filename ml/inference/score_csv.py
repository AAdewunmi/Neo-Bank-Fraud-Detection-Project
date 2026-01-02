"""
CLI for offline scoring of CSVs with the LedgerGuard scorer.

USAGE:

PYTHONPATH=. python -m ml.inference.score_csv \
  --input data/paysim.csv \
  --output reports/paysim_scored.csv \
  --dataset paysim \
  --threshold 0.7 \
  --registry model_registry.json

"""
from __future__ import annotations

import argparse

import pandas as pd

from ml.inference.scorer import Scorer


def _adapt_paysim(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" not in out.columns and "step" in out.columns:
        out["timestamp"] = pd.to_datetime(
            pd.to_numeric(out["step"], errors="coerce"),
            unit="h",
            origin="unix",
            errors="coerce",
        )
    if "customer_id" not in out.columns and "nameOrig" in out.columns:
        out["customer_id"] = out["nameOrig"]
    if "merchant" not in out.columns:
        if "nameDest" in out.columns:
            out["merchant"] = out["nameDest"]
        elif "type" in out.columns:
            out["merchant"] = out["type"]
        else:
            out["merchant"] = ""
    if "description" not in out.columns:
        if "type" in out.columns:
            out["description"] = out["type"]
        else:
            out["description"] = ""
    return out


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--registry", default="model_registry.json")
    parser.add_argument("--dataset", choices=["ledgerguard", "paysim"], default="ledgerguard")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.dataset == "paysim":
        df = _adapt_paysim(df)

    _require_columns(df, ["timestamp", "amount", "customer_id", "merchant", "description"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    scorer = Scorer(registry_path=args.registry)
    scored, diags = scorer.score(df, threshold=args.threshold)
    scored.to_csv(args.output, index=False)

    print(f"Wrote scored CSV: {args.output}")
    print(f"Diagnostics: {diags}")


if __name__ == "__main__":
    main()
