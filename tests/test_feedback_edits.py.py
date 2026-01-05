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



