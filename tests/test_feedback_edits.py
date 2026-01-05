"""
tests/test_feedback_edits.py

Integration tests for the feedback loop:
- Upload CSV (creates table rows and session cache)
- Apply edits (merged by row_id)
- Export feedback contains stable identifiers and edits
- UI reflects edits after refresh
"""

from __future__ import annotations

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

pytestmark = pytest.mark.django_db


def _upload_csv(client, django_user_model, monkeypatch):
    user = django_user_model.objects.create_user(
        username="ops",
        password="pass1234",
        is_staff=True,
    )
    client.force_login(user)

    def fake_score_df(df, threshold):
        scored = df.copy()
        scored["category"] = ["Cafe", "Misc"]
        scored["category_confidence"] = [0.9, 0.6]
        scored["fraud_risk"] = [0.1, 0.8]
        scored["flagged"] = [False, True]
        diags = {
            "pct_flagged": 0.5,
            "pct_auto_categorised": 1.0,
            "threshold": threshold,
            "n": len(scored),
            "flagged_key": "flagged",
        }
        return scored, diags

    import dashboard.services as services

    monkeypatch.setattr(services, "score_df", fake_score_df)

    content = (
        "timestamp,amount,customer_id,merchant,description\n"
        "2024-01-01T12:00:00Z,10.0,c1,Coffee Co,flat white\n"
        "2024-01-02T12:00:00Z,999.0,c2,Weird Merchant,suspicious\n"
    ).encode("utf-8")
    upload = SimpleUploadedFile("tx.csv", content, content_type="text/csv")

    return client.post(
        reverse("dashboard:index"),
        data={"threshold": 0.5, "csv_file": upload},
    )


def test_apply_edit_merges_and_export_contains_stable_id(
    client, django_user_model, monkeypatch
):
    resp = _upload_csv(client, django_user_model, monkeypatch)
    assert resp.status_code == 200
    assert resp.context is not None
    rows = resp.context["table"]
    assert len(rows) == 2

    row_id_0 = rows[0]["row_id"]
    row_id_1 = rows[1]["row_id"]

    # Apply two edits in two separate submits; both must persist.
    resp2 = client.post(
        reverse("dashboard:apply_edit"),
        data={"row_id": row_id_0, "new_category": "Food & Drink"},
    )
    assert resp2.status_code == 302

    resp3 = client.post(
        reverse("dashboard:apply_edit"),
        data={"row_id": row_id_1, "new_category": "Other"},
    )
    assert resp3.status_code == 302

    export = client.get(reverse("dashboard:export_feedback"))
    assert export.status_code == 200

    body = export.content.decode("utf-8")
    assert (
        "row_id,timestamp,customer_id,amount,merchant,description,predicted_category,new_category"
        in body
    )
    assert row_id_0 in body
    assert row_id_1 in body
    assert "Food & Drink" in body
    assert "Other" in body


def test_overlay_shows_edited_value_on_refresh(client, django_user_model, monkeypatch):
    resp = _upload_csv(client, django_user_model, monkeypatch)
    rows = resp.context["table"]
    row_id = rows[1]["row_id"]

    client.post(
        reverse("dashboard:apply_edit"),
        data={"row_id": row_id, "new_category": "Other"},
    )

    # Refresh dashboard; should render edited category.
    resp2 = client.get(reverse("dashboard:index"))
    assert resp2.status_code == 200
    content = resp2.content.decode("utf-8")
    assert "Other" in content
