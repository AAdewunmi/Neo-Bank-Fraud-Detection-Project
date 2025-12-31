"""
Tests for dashboard.services scorer caching.
"""
from __future__ import annotations

import pandas as pd

from dashboard import services


def test_score_df_reuses_singleton(monkeypatch) -> None:
    calls = []

    class FakeScorer:
        def __init__(self) -> None:
            calls.append("init")

        def score(self, df, threshold):
            df = df.copy()
            df["category"] = "X"
            df["fraud_risk"] = 0.1
            df["flagged"] = False
            return df, {"n": len(df), "threshold": threshold, "pct_flagged": 0.0}

    monkeypatch.setattr(services, "Scorer", FakeScorer)
    monkeypatch.setattr(services, "_SCORER", None, raising=False)

    df = pd.DataFrame([{
        "timestamp": "t",
        "amount": 1.0,
        "customer_id": "c1",
        "merchant": "m",
        "description": "d",
    }])

    services.score_df(df, threshold=0.5)
    services.score_df(df, threshold=0.7)

    assert calls == ["init"]
