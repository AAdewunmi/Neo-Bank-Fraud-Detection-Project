"""
Smoke tests for dashboard routing.
tests/test_dashboard_smoke.py
"""
from __future__ import annotations

from django.urls import reverse


def test_index_smoke(client) -> None:
    """
    The root dashboard endpoint should respond successfully.
    """
    resp = client.get(reverse("dashboard"))
    assert resp.status_code == 200
    assert b"LedgerGuard" in resp.content
