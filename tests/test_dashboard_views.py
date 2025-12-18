"""
Integration-style tests for dashboard views.
tests/test_dashboard_views.py

Week 1 intent:
- GET renders successfully
- POST with valid CSV renders KPIs and a table
- POST with invalid CSV shows a readable error

These tests do not require database access.
"""
from __future__ import annotations

from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse


def test_dashboard_get_renders(client) -> None:
    resp = client.get(reverse("dashboard"))
    assert resp.status_code == 200
    assert b"LedgerGuard" in resp.content


def test_dashboard_post_valid_csv_renders_table(monkeypatch, client) -> None:
    def fake_score_df(df, threshold):
        df = df.copy()
        df["category"] = "Test"
        df["category_confidence"] = 0.9
        df["fraud_risk"] = 0.1
        df["flagged"] = False
        diags = {"pct_flagged": 0.0, "pct_auto_categorised": 1.0, "threshold": threshold}
        return df, diags

    import dashboard.services as services

    monkeypatch.setattr(services, "score_df", fake_score_df)

    csv_bytes = (
        "timestamp,amount,customer_id,merchant,description\n"
        "2024-01-01T00:00:00Z,10.0,c1,m1,d1\n"
    ).encode("utf-8")

    upload = SimpleUploadedFile("tx.csv", csv_bytes, content_type="text/csv")
    resp = client.post(reverse("dashboard"), data={"threshold": 0.65, "csv_file": upload})

    assert resp.status_code == 200
    assert b"KPIs" in resp.content
    assert b"Transactions" in resp.content


def test_dashboard_post_missing_columns_shows_error(client) -> None:
    bad_csv = b"foo,bar\n1,2\n"
    upload = SimpleUploadedFile("bad.csv", bad_csv, content_type="text/csv")

    resp = client.post(reverse("dashboard"), data={"threshold": 0.65, "csv_file": upload})

    assert resp.status_code == 200
    assert b"Missing required columns" in resp.content
