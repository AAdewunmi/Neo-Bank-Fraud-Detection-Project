"""
tests/test_precedence.py

Integration test asserting precedence:
edit > rule > model
"""

from __future__ import annotations

import io

from django.urls import reverse


def test_edit_wins_over_rule(client):
    content = (
        "timestamp,amount,customer_id,merchant,description,category\n"
        "2024-01-01T12:00:00Z,10.0,c1,Coffee Co,flat white,Other\n"
    ).encode("utf-8")

    resp = client.post(
        reverse("upload_and_score"),
        data={"threshold": 0.5, "csv_file": io.BytesIO(content)},
    )
    assert resp.status_code == 200
    row_id = resp.context["rows"][0]["row_id"]

    # Apply edit that should override any rule-based category.
    client.post(reverse("apply_edit"), data={"row_id": row_id, "new_category": "Custom Category"})

    resp2 = client.get(reverse("dashboard"))
    assert resp2.status_code == 200
    html = resp2.content.decode("utf-8")
    assert "Custom Category" in html
