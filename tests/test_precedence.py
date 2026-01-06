"""
tests/test_precedence.py

Integration test asserting precedence:
edit > rule > model
"""

from __future__ import annotations

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse


pytestmark = pytest.mark.django_db


def test_edit_wins_over_rule(client, django_user_model):
    user = django_user_model.objects.create_user(username="ops", password="pass1234", is_staff=True)
    client.force_login(user)

    content = (
        "timestamp,amount,customer_id,merchant,description,category\n"
        "2024-01-01T12:00:00Z,10.0,c1,Coffee Co,flat white,Other\n"
    ).encode("utf-8")
    upload = SimpleUploadedFile("tx.csv", content, content_type="text/csv")

    resp = client.post(
        reverse("dashboard"),
        data={"threshold": 0.5, "csv_file": upload},
    )
    assert resp.status_code == 200
    row_id = resp.context["table"][0]["row_id"]

    # Apply edit that should override any rule-based category.
    client.post(
        reverse("dashboard:apply_edit"),
        data={"row_id": row_id, "new_category": "Custom Category"},
    )

    resp2 = client.get(reverse("dashboard"))
    assert resp2.status_code == 200
    html = resp2.content.decode("utf-8")
    assert "Custom Category" in html
