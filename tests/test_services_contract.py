"""
Service-level tests for score_df and diagnostics contract.
"""
from __future__ import annotations
import pandas as pd
from dashboard import services


def test_score_df_returns_expected_keys(monkeypatch):
    class FakeScorer:
        def score(self, df, threshold):
            df = df.copy()
            df["category"] = "X"
            df["fraud_risk"] = 0.1
            df["flagged"] = False
            return df, {"n": len(df), "threshold": threshold, "pct_flagged": 0.0}

    monkeypatch.setattr(services, "Scorer", lambda: FakeScorer())

    df = pd.DataFrame([{
        "timestamp": "t",
        "amount": 1.0,
        "customer_id": "c1",
        "merchant": "m",
        "description": "d"
    }])
    scored, diags = services.score_df(df, threshold=0.5)

    assert set(diags.keys()) == {"n", "threshold", "pct_flagged", "pct_auto_categorised"}
    assert diags["pct_auto_categorised"] == 1.0
    assert len(scored) == 1
