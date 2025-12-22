from __future__ import annotations

import pytest
from django.urls import reverse

pytestmark = pytest.mark.django_db

"""
Smoke tests for dashboard routing.
tests/test_dashboard_smoke.py
"""
"""
Smoke tests for the ops dashboard route.
"""


def test_index_smoke_requires_login(client) -> None:
    resp = client.get(reverse("dashboard"))
    assert resp.status_code == 302
    assert "/accounts/login/" in resp["Location"]


def test_index_smoke_ok_after_login(client, django_user_model) -> None:
    user = django_user_model.objects.create_user(username="ops", password="pass1234", is_staff=True)
    client.force_login(user)

    resp = client.get(reverse("dashboard"))
    assert resp.status_code == 200
