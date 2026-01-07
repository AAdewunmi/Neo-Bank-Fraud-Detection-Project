# tests/test_export_views_final_branches.py
"""
Tests for final branches in export_views.py
"""

import pandas as pd
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from dashboard import services


@pytest.mark.django_db
def test_export_all_csv_success(monkeypatch, client, django_user_model):
    """Covers export success branch (export_views.py lines 95–117)."""
    user = django_user_model.objects.create_user(username="ops", password="p", is_staff=True)
    client.force_login(user)

    # Fake scored dataframe and diagnostics
    df = pd.DataFrame(
        [
            {"timestamp": "t1", "amount": 10.0, "customer_id": "c1",
             "merchant": "m1", "description": "ok", "flagged": True}
        ]
    )
    diags = {"n": 1, "pct_flagged": 1.0, "threshold": 0.7}

    # Stub scoring functions
    def fake_score_df(df, threshold):
        return df, diags

    monkeypatch.setattr(services, "read_csv", lambda f: df)
    monkeypatch.setattr(services, "score_df", fake_score_df)

    # Simulate upload + scoring to trigger session creation
    upload = SimpleUploadedFile(
        "tx.csv",
        b"timestamp,amount,customer_id,merchant,description\n",
        content_type="text/csv",
    )
    client.post("/ops/", data={"threshold": 0.7, "csv_file": upload})

    # Export should now succeed (DB-backed)
    resp = client.get("/ops/export/all/")
    assert resp.status_code == 200
    assert resp["Content-Type"].startswith("text/csv")
    body = resp.content.decode("utf-8")
    assert "customer_id" in body
    assert "flagged" in body
