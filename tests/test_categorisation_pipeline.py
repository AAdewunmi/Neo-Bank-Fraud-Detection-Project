"""
Tests for the categorisation baseline pipeline.

Week 1 intent:
- prove the pipeline fits and predicts
- prove schema_hash is stable and sensitive to column order
"""
from __future__ import annotations

import pandas as pd

from ml.training.train_categorisation import build_pipeline
from ml.training.utils import schema_hash


def test_pipeline_trains_and_predicts() -> None:
    """
    Pipeline should fit on tiny data and predict a single label.
    """
    X = pd.Series(["coffee shop", "weekly groceries", "monthly subscription"])
    y = pd.Series(["Food & Drink", "Groceries", "Subscriptions"])

    pipe = build_pipeline()
    pipe.fit(X, y)

    pred = pipe.predict(pd.Series(["coffee"]))
    assert len(pred) == 1
    assert isinstance(pred[0], str)
