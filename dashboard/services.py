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
    return scorer.score(df, threshold=threshold)
