"""
Threshold monotonicity tests.

If threshold increases, flagged count must not increase.
"""
from __future__ import annotations

import pandas as pd

from ml.inference.scorer import Scorer


def test_threshold_monotonicity(monkeypatch) -> None:
    """
    Lower thresholds should flag the same or more rows than higher thresholds.
    """
    scorer = Scorer()

    class FakeCat:
        def predict(self, X):
            return ["X"] * len(X)

        def predict_proba(self, X):
            import numpy as np
            return np.c_[[0.5] * len(X), [0.5] * len(X)]

    class FakeFraud:
        def decision_function(self, X):
            import numpy as np
            return -np.array([0.1, 0.2, 0.3, 0.4])

    scorer._load = lambda task, version: (FakeCat()
                                          if task == "categorisation" else FakeFraud())
    # type: ignore
    scorer._registry["categorisation"]["latest"] = "fake_cat"
    scorer._registry["fraud"]["latest"] = "fake_fraud"
    scorer._registry["categorisation"]["fake_cat"] = {"text_cols": ["merchant", "description"]}
    scorer._registry["fraud"]["fake_fraud"] = {}

    df = pd.DataFrame(
        {
            "timestamp": ["t1", "t2", "t3", "t4"],
            "amount": [10, 20, 30, 40],
            "customer_id": ["c1", "c2", "c3", "c4"],
            "merchant": ["m"] * 4,
            "description": ["d"] * 4,
        }
    )

    out_low, _ = scorer.score(df, threshold=0.3)
    out_high, _ = scorer.score(df, threshold=0.8)

    assert out_low["flagged"].sum() >= out_high["flagged"].sum()
