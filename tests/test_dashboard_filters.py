"""
Server-side filter tests.

Verifies filtering operates on session-backed scored rows (no re-score).
"""
from __future__ import annotations

import pandas as pd
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

pytestmark = pytest.mark.django_db


def _fake_scoring():
    scored = pd.DataFrame(
        [
            {
                "timestamp": "2025-01-01T10:00:00Z",
                "amount": 10.00,
                "customer_id": "C001",
                "merchant": "Coffee Shop",
                "description": "latte",
                "category": "Food",
                "category_confidence": 0.91,
                "fraud_risk": 0.12,
                "flagged": False,
            },
            {
                "timestamp": "2025-01-01T11:00:00Z",
                "amount": 999.00,
                "customer_id": "C002",
                "merchant": "Electronics",
                "description": "laptop",
                "category": "Retail",
                "category_confidence": 0.77,
                "fraud_risk": 0.93,
                "flagged": True,
            },
        ]
    )
    diags = {"threshold": 0.7, "pct_flagged": 0.5, "pct_auto_categorised": 1.0, "n": 2}
    return scored, diags


def test_filters_apply_to_session_run(monkeypatch, client, django_user_model) -> None:
    user = django_user_model.objects.create_user(username="ops", password="pass1234", is_staff=True)
    client.force_login(user)

    monkeypatch.setattr("dashboard.views.services.read_csv", lambda f: pd.DataFrame([{"x": 1}]))
    monkeypatch.setattr("dashboard.views.services.score_df", lambda df, threshold: _fake_scoring())

    upload = SimpleUploadedFile(
        "sample.csv",
        b"timestamp,amount,customer_id,merchant,description\n",
        content_type="text/csv",
    )
    resp = client.post("/ops/", data={"threshold": "0.7", "csv_file": upload})
    assert resp.status_code == 200

    filtered = client.get("/ops/?flagged_only=on")
    assert filtered.status_code == 200
    html = filtered.content.decode("utf-8")

    assert "Electronics" in html
    assert "Coffee Shop" not in html
