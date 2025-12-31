from __future__ import annotations

import numpy as np
import pandas as pd

from ml.training.splits import split_train_test


def test_time_split_leakage_guard_removes_overlap_customers() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-03T00:00:00Z",
                "2025-01-04T00:00:00Z",
            ],
            "customer_id": ["C1", "C2", "C1", "C3"],
        }
    )
    y = np.array([0, 0, 1, 0])

    split = split_train_test(
        df=df,
        y=y,
        test_size=0.25,
        seed=42,
        split_mode="time",
        timestamp_col="timestamp",
        group_col="customer_id",
        leakage_guard=True,
    )

    train_customers = set(df.iloc[split.train_idx]["customer_id"].tolist())
    test_customers = set(df.iloc[split.test_idx]["customer_id"].tolist())
    assert train_customers.isdisjoint(test_customers)
