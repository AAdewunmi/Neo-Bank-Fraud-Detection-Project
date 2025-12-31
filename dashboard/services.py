"""
Dashboard services for Week 1.

Week 1 intent:
- enforce the data contract at ingestion time
- return dataframes that downstream scoring can rely on
- keep view logic thin and predictable
"""
from __future__ import annotations

import io
from typing import Any, Dict, List, Tuple

import pandas as pd

from ml.inference.scorer import Scorer

REQUIRED_COLUMNS: List[str] = ["timestamp", "amount", "customer_id", "merchant", "description"]


def read_csv(file_obj) -> pd.DataFrame:
    """
    Read an uploaded CSV and enforce the Week 1 data contract.

    Args:
        file_obj: File-like object (e.g., Django UploadedFile or BytesIO).

    Returns:
        Cleaned dataframe containing at least REQUIRED_COLUMNS.

    Raises:
        ValueError: If required columns are missing or customer_id is empty.
    """
    raw = file_obj.read()
    df = pd.read_csv(io.BytesIO(raw))

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    df["merchant"] = df["merchant"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)

    df["customer_id"] = df["customer_id"].fillna("").astype(str)
    if (df["customer_id"].str.len() == 0).any():
        raise ValueError("customer_id must be non-empty for all rows")

    return df


def score_df(df: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Score a DataFrame using the shared ML scorer.

    Args:
        df: Validated input DataFrame.
        threshold: Fraud threshold in [0.0, 1.0].

    Returns:
        Tuple of (scored dataframe, diagnostics dict).
    """
    scorer = Scorer()
    scored_df, diags = scorer.score(df, threshold=threshold)

    # Normalise baseline diagnostics keys so the view relies on a stable contract.
    diags["n"] = int(diags.get("n", len(scored_df)))
    diags["threshold"] = float(diags.get("threshold", threshold))
    diags["pct_flagged"] = float(diags.get("pct_flagged", 0.0))

    # Week 1 definition: "auto categorised" means category is present and non-empty.
    # If categorisation isn't wired yet, this will cleanly report 0.0.
    pct_auto_categorised = 0.0
    if "category" in scored_df.columns:
        cat = scored_df["category"].fillna("").astype(str)
        pct_auto_categorised = float((cat.str.len() > 0).mean())

    diags["pct_auto_categorised"] = float(diags.get("pct_auto_categorised", pct_auto_categorised))

    # Guaranteed console visibility.
    print(f"[LedgerGuard] score_df diagnostics: {diags}", flush=True)

    return scored_df, diags


def read_csv_file(file_obj) -> pd.DataFrame:
    """
    Backwards-compatible alias for read_csv used by the dashboard view.
    """
    return read_csv(file_obj)


def score_transactions(df: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Backwards-compatible alias for score_df used by the dashboard view.
    """
    return score_df(df, threshold)
