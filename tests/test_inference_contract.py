"""
Inference contract tests.

Week 1 goal: protect output columns + threshold behavior.
"""
from __future__ import annotations

import pandas as pd

from ml.inference.scorer import Scorer


def test_scorer_contract_monkeypatched(monkeypatch) -> None:
    """
    Score must return required columns and bounded risk.

    Uses fake models so the test does not depend on local artefacts.
    """
    scorer = Scorer()

    class FakeCat:
        def predict(self, X):
            return ["Food"] * len(X)

        def predict_proba(self, X):
            import numpy as np
            return np.c_[[0.2] * len(X), [0.8] * len(X)]

    class FakeFraud:
        def decision_function(self, X):
            import numpy as np
            return -np.linspace(0.1, 1.0, len(X))

    scorer._load = lambda task, version: (FakeCat()
                                          if task == "categorisation" else FakeFraud())
    # type: ignore
    scorer._registry["categorisation"]["latest"] = "fake_cat"
    scorer._registry["fraud"]["latest"] = "fake_fraud"
    scorer._registry["categorisation"]["fake_cat"] = {"text_cols": ["merchant", "description"]}
    scorer._registry["fraud"]["fake_fraud"] = {}

    df = pd.DataFrame(
        {
            "timestamp": ["t1", "t2", "t3"],
            "amount": [10, 100, 1000],
            "customer_id": ["c1", "c2", "c3"],
            "merchant": ["m1", "m2", "m3"],
            "description": ["d1", "d2", "d3"],
        }
    )

    out, diags = scorer.score(df, threshold=0.5)

    assert {"category", "category_confidence", "fraud_risk", "flagged"}.issubset(out.columns)
    assert out["fraud_risk"].between(0.0, 1.0).all()
    assert 0.0 <= diags["pct_flagged"] <= 1.0
