"""
Tabular fraud features with a strict, name-ordered schema.
Train-time uses full data; inference-time uses batch approximations.
"""
from __future__ import annotations
import pandas as pd

FEATURE_ORDER = ["amount", "amount_z", "amount_mean_cust", "amount_std_cust", "hour"]


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


# def compute_infer_features(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Compute features using only the batch (approximation).
#     """
#     out = df.copy()
#     out["hour"] = pd.to_datetime(out["timestamp"], errors="coerce").dt.hour.fillna(0).astype(int)
#     grp = out.groupby("customer_id")["amount"]
#     out["amount_mean_cust"] = grp.transform("mean")
#     out["amount_std_cust"] = grp.transform("std").fillna(0.0)
#     out["amount_z"] = (out["amount"] - out["amount_mean_cust"]) / (out["amount_std_cust"] + 1e-9)
#     return out[FEATURE_ORDER]
