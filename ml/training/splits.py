"""
Split utilities for supervised fraud training.

Goals
- Reduce leakage by supporting time based splits and group based splits.
- Provide a safe random split fallback for small datasets.
- Emit split metadata for metrics and model cards.

Notes
- Time splits are preferred when a timestamp column exists.
- Group splits are preferred when customer id exists and leakage risk is high.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


@dataclass(frozen=True)
class SplitResult:
    """
    Result container for train test splitting.
    """

    train_idx: np.ndarray
    test_idx: np.ndarray
    meta: Dict[str, Any]


def _safe_stratify_split(
    y: np.ndarray,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Attempt stratified split, fall back to non stratified when classes are too small.
    """
    y_series = pd.Series(y)
    test_count = max(1, int(round(len(y_series) * test_size)))
    class_counts = y_series.value_counts()
    can_stratify = (
        len(class_counts) <= test_count
        and (class_counts.min() if len(class_counts) else 0) >= 2
    )

    if can_stratify:
        idx_train, idx_test = train_test_split(
            np.arange(len(y)),
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )
        return (
            np.asarray(idx_train),
            np.asarray(idx_test),
            {"split_type": "random_stratified"},
        )

    idx_train, idx_test = train_test_split(
        np.arange(len(y)),
        test_size=test_size,
        random_state=seed,
        stratify=None,
    )
    return (
        np.asarray(idx_train),
        np.asarray(idx_test),
        {"split_type": "random_no_stratify"},
    )


def _time_split_indices(
    df: pd.DataFrame,
    timestamp_col: str,
    test_size: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Time split by sorting timestamp ascending and taking the last fraction as test.
    """
    ts = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
    valid = ts.notna().values
    if valid.sum() < 10:
        raise ValueError("Too few valid timestamps for time split.")

    order = np.argsort(ts.values.astype("datetime64[ns]"))
    n = len(df)
    n_test = max(1, int(round(n * test_size)))
    test_idx = order[-n_test:]
    train_idx = order[:-n_test]

    meta = {
        "split_type": "time",
        "timestamp_col": timestamp_col,
        "n_total": int(n),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }
    return np.asarray(train_idx), np.asarray(test_idx), meta


def _group_split_indices(
    df: pd.DataFrame,
    group_col: str,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Group split so the same group does not appear in both train and test.
    """
    groups = df[group_col].astype(str).values
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=groups))

    meta = {
        "split_type": "group",
        "group_col": group_col,
        "n_total": int(len(df)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_groups": int(pd.Series(groups).nunique()),
    }
    return np.asarray(train_idx), np.asarray(test_idx), meta


def _leakage_guard_remove_overlap(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    group_col: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Remove any train rows that share a group with the test set.

    This is useful when a time split is used but customers appear in both windows.
    """
    train_groups = set(df.iloc[train_idx][group_col].astype(str).tolist())
    test_groups = set(df.iloc[test_idx][group_col].astype(str).tolist())
    overlap = train_groups.intersection(test_groups)

    if not overlap:
        return train_idx, {"leakage_guard": "none", "overlap_groups": 0}

    keep_mask = ~df.iloc[train_idx][group_col].astype(str).isin(overlap).values
    filtered_train_idx = train_idx[keep_mask]

    meta = {
        "leakage_guard": "removed_overlap_groups",
        "overlap_groups": int(len(overlap)),
        "train_removed_rows": int(len(train_idx) - len(filtered_train_idx)),
    }
    return filtered_train_idx, meta


def split_train_test(
    df: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.25,
    seed: int = 42,
    split_mode: str = "time",
    timestamp_col: str = "timestamp",
    group_col: str = "customer_id",
    leakage_guard: bool = True,
) -> SplitResult:
    """
    Split indices for training and testing.

    split_mode options
    - time uses timestamp ordering
    - group uses customer id grouping
    - random uses a safe stratified split when possible

    leakage_guard removes overlapping customer ids from the training set after time split.
    """
    split_mode = split_mode.strip().lower()

    if split_mode == "time" and timestamp_col in df.columns:
        train_idx, test_idx, meta = _time_split_indices(df, timestamp_col, test_size)
        if leakage_guard and group_col in df.columns:
            train_idx, guard_meta = _leakage_guard_remove_overlap(
                df, train_idx, test_idx, group_col
            )
            meta.update(guard_meta)
            meta["n_train"] = int(len(train_idx))
        return SplitResult(train_idx=train_idx, test_idx=test_idx, meta=meta)

    if split_mode == "group" and group_col in df.columns:
        train_idx, test_idx, meta = _group_split_indices(df, group_col, test_size, seed)
        return SplitResult(train_idx=train_idx, test_idx=test_idx, meta=meta)

    train_idx, test_idx, meta = _safe_stratify_split(y=y, test_size=test_size, seed=seed)
    return SplitResult(train_idx=train_idx, test_idx=test_idx, meta=meta)
