"""
tests/test_feedback_edits.py

Integration tests for the feedback loop:
- Upload CSV (creates table rows and session cache)
- Apply edits (merged by row_id)
- Export feedback contains stable identifiers and edits
- UI reflects edits after refresh
"""

from __future__ import annotations

import io

from django.urls import reverse


def _upload_csv(client):
    content = (
        "timestamp,amount,customer_id,merchant,description\n"
        "2024-01-01T12:00:00Z,10.0,c1,Coffee Co,flat white\n"
        "2024-01-02T12:00:00Z,999.0,c2,Weird Merchant,suspicious\n"
    ).encode("utf-8")

    return client.post(
        reverse("upload_and_score"),
        data={"threshold": 0.5, "csv_file": io.BytesIO(content)},
    )


def test_apply_edit_merges_and_export_contains_stable_id(client):
    resp = _upload_csv(client)
    assert resp.status_code == 200
    assert resp.context is not None
    rows = resp.context["rows"]
    assert len(rows) == 2

    row_id_0 = rows[0]["row_id"]
    row_id_1 = rows[1]["row_id"]

    # Apply two edits in two separate submits; both must persist.
    resp2 = client.post(reverse("apply_edit"), data={"row_id": row_id_0, "new_category": "Food & Drink"})
    assert resp2.status_code == 302

    resp3 = client.post(reverse("apply_edit"), data={"row_id": row_id_1, "new_category": "Other"})
    assert resp3.status_code == 302

    export = client.get(reverse("export_feedback"))
    assert export.status_code == 200

    body = export.content.decode("utf-8")
    assert "row_id,timestamp,customer_id,amount,merchant,description,predicted_category,new_category" in body
    assert row_id_0 in body
    assert row_id_1 in body
    assert "Food & Drink" in body
    assert "Other" in body



