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


def test_schema_hash_is_stable_and_ordered() -> None:
    """
    Column order must affect the schema hash (guards accidental reshuffles).
    """
    h1 = schema_hash(["merchant", "description", "category"])
    h2 = schema_hash(["merchant", "description", "category"])
    h3 = schema_hash(["description", "merchant", "category"])
    assert h1 == h2
    assert h1 != h3
