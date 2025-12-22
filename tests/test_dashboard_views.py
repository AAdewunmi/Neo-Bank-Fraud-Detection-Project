"""
Ops dashboard view tests.

These tests log in because /ops/ is an internal area.
"""
from __future__ import annotations

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

pytestmark = pytest.mark.django_db


def test_dashboard_get_renders_after_login(client, django_user_model) -> None:
    user = django_user_model.objects.create_user(username="ops", password="pass1234", is_staff=True)
    client.force_login(user)

    resp = client.get(reverse("dashboard"))
    assert resp.status_code == 200


def test_dashboard_post_valid_csv_renders_table(monkeypatch, client, django_user_model) -> None:
    user = django_user_model.objects.create_user(username="ops", password="pass1234", is_staff=True)
    client.force_login(user)

    def fake_score_df(df, threshold):
        df = df.copy()
        df["category"] = "Test"
        df["category_confidence"] = 0.9
        df["fraud_risk"] = 0.1
        df["flagged"] = False
        diags = {
            "pct_flagged": 0.0,
            "pct_auto_categorised": 1.0,
            "threshold": threshold,
            "n": len(df),
        }
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
    assert b"Transactions" in resp.content


def test_dashboard_post_missing_columns_shows_error(client, django_user_model) -> None:
    user = django_user_model.objects.create_user(username="ops", password="pass1234", is_staff=True)
    client.force_login(user)

    bad_csv = b"foo,bar\n1,2\n"
    upload = SimpleUploadedFile("bad.csv", bad_csv, content_type="text/csv")

    resp = client.post(reverse("dashboard"), data={"threshold": 0.65, "csv_file": upload})
    assert resp.status_code == 200
    assert b"Missing required columns" in resp.content
