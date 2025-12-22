"""
Export flagged CSV behavior tests.

Verifies:
- Ops routes require authentication
- Export returns a CSV download
- Export includes only flagged rows (plus headers)
- Missing session payload returns 400
"""
from __future__ import annotations

import io
from typing import Tuple

import pandas as pd
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

pytestmark = pytest.mark.django_db


def _fake_scoring() -> Tuple[pd.DataFrame, dict]:
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


def test_ops_requires_login(client) -> None:
    resp = client.get("/ops/")
    assert resp.status_code == 302
    assert "/accounts/login/" in resp["Location"]


def test_export_missing_session_returns_400(client, django_user_model) -> None:
    user = django_user_model.objects.create_user(username="ops", password="pass1234", is_staff=True)
    client.force_login(user)

    resp = client.get("/ops/export/flagged/")
    assert resp.status_code == 400
    assert b"Upload and score a CSV first" in resp.content


def test_export_contains_only_flagged_rows(monkeypatch, client, django_user_model) -> None:
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

    export = client.get("/ops/export/flagged/")
    assert export.status_code == 200
    assert export["Content-Type"].startswith("text/csv")
    assert "attachment; filename=" in export["Content-Disposition"]

    content = export.content.decode("utf-8")
    lines = [line for line in io.StringIO(content).read().splitlines() if line.strip()]
    assert len(lines) == 2
    assert "Electronics" in content
    assert "Coffee Shop" not in content
