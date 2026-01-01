"""
Tabular fraud features with a strict, name-ordered schema.
Train-time uses full data; inference-time uses batch or state store stats.
"""
from __future__ import annotations
import pandas as pd

FEATURE_ORDER = ["amount", "amount_z", "amount_mean_cust", "amount_std_cust", "hour"]
STATE_COLUMNS = ["customer_id", "amount_mean_cust", "amount_std_cust"]


def build_customer_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-customer aggregates for inference parity.
    """
    out = df.copy()
    grp = out.groupby("customer_id")["amount"]
    state = grp.agg(amount_mean_cust="mean", amount_std_cust="std").reset_index()
    state["amount_std_cust"] = state["amount_std_cust"].fillna(0.0)
    return state[STATE_COLUMNS]


def compute_train_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features with per-customer stats using the full training frame.
    """
    out = df.copy()
    out["hour"] = pd.to_datetime(out["timestamp"], errors="coerce").dt.hour.fillna(0).astype(int)
    grp = out.groupby("customer_id")["amount"]
    out["amount_mean_cust"] = grp.transform("mean")
    out["amount_std_cust"] = grp.transform("std").fillna(0.0)
    out["amount_z"] = (out["amount"] - out["amount_mean_cust"]) / (out["amount_std_cust"] + 1e-9)
    return out[FEATURE_ORDER]


def compute_infer_features(
    df: pd.DataFrame,
    state: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute features using state store aggregates when available.
    """
    out = df.copy()
    out["hour"] = pd.to_datetime(out["timestamp"], errors="coerce").dt.hour.fillna(0).astype(int)

    if state is not None and not state.empty:
        state_trim = state[STATE_COLUMNS].copy()
        out = out.merge(state_trim, on="customer_id", how="left")
        missing = out["amount_mean_cust"].isna() | out["amount_std_cust"].isna()
        if missing.any():
            grp = out.groupby("customer_id")["amount"]
            batch_mean = grp.transform("mean")
            batch_std = grp.transform("std").fillna(0.0)
            out["amount_mean_cust"] = out["amount_mean_cust"].fillna(batch_mean)
            out["amount_std_cust"] = out["amount_std_cust"].fillna(batch_std)
    else:
        grp = out.groupby("customer_id")["amount"]
        out["amount_mean_cust"] = grp.transform("mean")
        out["amount_std_cust"] = grp.transform("std").fillna(0.0)

    out["amount_z"] = (out["amount"] - out["amount_mean_cust"]) / (out["amount_std_cust"] + 1e-9)
    return out[FEATURE_ORDER]
