"""
Dashboard services for Week 1.

Week 1 intent:
- enforce the data contract at ingestion time
- return dataframes that downstream scoring can rely on
- keep view logic thin and predictable
"""
from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Tuple

import pandas as pd

from ml.inference.scorer import Scorer

REQUIRED_COLUMNS: List[str] = ["timestamp", "amount", "customer_id", "merchant", "description"]
_SCORER: Scorer | None = None


def _get_scorer() -> Scorer:
    global _SCORER
    if _SCORER is None:
        _SCORER = Scorer()
    return _SCORER


def _engineer_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    ts = pd.to_datetime(df.get("timestamp"), errors="coerce", utc=True)

    if "hour" not in df.columns:
        df["hour"] = ts.dt.hour.fillna(0).astype(int)

    if "is_weekend" not in df.columns:
        df["is_weekend"] = (ts.dt.weekday >= 5).fillna(False).astype(int)

    if "amount_bucket" not in df.columns:
        amounts = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0).astype(float)
        df["amount_bucket"] = pd.cut(
            amounts,
            bins=[-1.0, 9.99, 99.99, 999.99, float("inf")],
            labels=[0, 1, 2, 3],
        ).astype(int)

    if "velocity_24h" not in df.columns:
        if "customer_id" in df.columns and ts.notna().any():
            df["_ts"] = ts
            df["_orig_idx"] = range(len(df))
            df.sort_values(["customer_id", "_ts"], inplace=True)

            def _rolling_count(group: pd.DataFrame) -> pd.Series:
                s = pd.Series(1, index=group["_ts"])
                counts = s.rolling("24h").sum().fillna(0).astype(int)
                return pd.Series(counts.values, index=group.index)

            df["velocity_24h"] = (
                df.groupby("customer_id", sort=False, group_keys=False)[["_ts"]]
                .apply(_rolling_count)
                .astype(int)
            )
            df.sort_values("_orig_idx", inplace=True)
            df.drop(columns=["_ts", "_orig_idx"], inplace=True)
        else:
            df["velocity_24h"] = 0

    if "is_international" not in df.columns:
        home_country = os.environ.get("LEDGERGUARD_HOME_COUNTRY")
        home_currency = os.environ.get("LEDGERGUARD_HOME_CURRENCY")
        if home_country and "country" in df.columns:
            df["is_international"] = (
                df["country"].astype(str).str.upper() != home_country.upper()
            ).astype(int)
        elif home_currency and "currency" in df.columns:
            df["is_international"] = (
                df["currency"].astype(str).str.upper() != home_currency.upper()
            ).astype(int)
        else:
            df["is_international"] = 0

    return df


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

    return _engineer_fraud_features(df)


def score_df(df: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Score a DataFrame using the shared ML scorer.

    Args:
        df: Validated input DataFrame.
        threshold: Fraud threshold in [0.0, 1.0].

    Returns:
        Tuple of (scored dataframe, diagnostics dict).
    """
    scorer = _get_scorer()
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
