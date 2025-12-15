"""
Smoke tests for the Django scaffold.
"""
from __future__ import annotations

from django.urls import reverse


def test_index_smoke(client) -> None:
    """
    The root dashboard endpoint should respond successfully.
    """
    resp = client.get(reverse("dashboard"))
    assert resp.status_code == 200
    assert b"Week 1: scaffolding" in resp.content
