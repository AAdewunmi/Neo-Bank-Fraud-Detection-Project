# tests/test_export_views_no_flagged_data.py
"""
Tests for export views when there is no flagged data.
"""

import pandas as pd
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from dashboard import services


@pytest.mark.django_db
def test_export_flagged_csv_no_flagged_rows(monkeypatch, client, django_user_model):
    """Simulates a scored run where no rows are flagged (covers export_views error branch)."""
    user = django_user_model.objects.create_user(username="ops", password="p", is_staff=True)
    client.force_login(user)

    df = pd.DataFrame(
        [
            {
                "timestamp": "t",
                "amount": 1.0,
                "customer_id": "c1",
                "merchant": "m",
                "description": "d",
                "flagged": False,
            }
        ]
    )
    diags = {"n": 1, "pct_flagged": 0.0, "threshold": 0.7}

    def fake_score_df(df, threshold):
        return df, diags

    monkeypatch.setattr(services, "read_csv", lambda f: df)
    monkeypatch.setattr(services, "score_df", fake_score_df)

    upload = SimpleUploadedFile(
        "tx.csv",
        b"timestamp,amount,customer_id,merchant,description\n",
        content_type="text/csv",
    )
    client.post("/ops/", data={"threshold": 0.7, "csv_file": upload})

    resp = client.get("/ops/export/flagged/")
    assert resp.status_code == 400
    assert b"No flagged rows available" in resp.content
