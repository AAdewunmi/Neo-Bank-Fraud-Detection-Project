"""Customer site transaction view tests."""
from __future__ import annotations

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

from dashboard.session_store import load_scored_run

pytestmark = pytest.mark.django_db


def test_customer_site_empty_state(client) -> None:
    resp = client.get("/customer/")
    assert resp.status_code == 200
    assert b"No transactions yet" in resp.content


def test_customer_site_renders_safe_rows_and_edits(monkeypatch, client, django_user_model) -> None:
    user = django_user_model.objects.create_user(
        username="ops", password="pass1234", is_staff=True
    )
    client.force_login(user)

    def fake_score_df(df, threshold):
        df = df.copy()
        df["category"] = "Travel"
        df["fraud_risk"] = 0.91
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
        "2024-01-01T00:00:00Z,42.5,c1,QuickDrop,Order 412\n"
    ).encode("utf-8")

    upload = SimpleUploadedFile("tx.csv", csv_bytes, content_type="text/csv")
    resp = client.post("/ops/", data={"threshold": 0.65, "csv_file": upload})
    assert resp.status_code == 200

    customer_resp = client.get("/customer/")
    assert customer_resp.status_code == 200
    assert b"QuickDrop" in customer_resp.content
    assert b"Order 412" in customer_resp.content
    assert b"Travel" in customer_resp.content
    assert b"fraud_risk" not in customer_resp.content

    scored_run = load_scored_run(client.session)
    assert scored_run is not None
    row_id = scored_run.rows[0]["row_id"]

    edit_resp = client.post("/ops/edit/", data={"row_id": row_id, "new_category": "Food"})
    assert edit_resp.status_code == 302

    edited_resp = client.get("/customer/")
    assert edited_resp.status_code == 200
    assert b"Food" in edited_resp.content
