"""
Extra coverage for export view error handling.
"""
from __future__ import annotations

import pytest


@pytest.mark.django_db
def test_export_all_csv_no_data_returns_400(client, django_user_model):
    user = django_user_model.objects.create_user(
        username="ops",
        password="p",
        is_staff=True,
    )
    client.force_login(user)

    resp = client.get("/ops/export/all/")
    assert resp.status_code == 400
    assert b"No export data found" in resp.content
