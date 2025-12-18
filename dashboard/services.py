"""
Dashboard services for Week 1.

Week 1 intent:
- enforce the data contract at ingestion time
- return dataframes that downstream scoring can rely on
"""
from __future__ import annotations

import io
from typing import List

import pandas as pd


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

    # Coerce amount defensively to avoid runtime failures downstream.
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    # merchant/description may be null; default to empty strings.
    df["merchant"] = df["merchant"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)

    # customer_id is required and must not be empty after coercion.
    df["customer_id"] = df["customer_id"].fillna("").astype(str)
    if (df["customer_id"].str.len() == 0).any():
        raise ValueError("customer_id must be non-empty for all rows")

    return df
