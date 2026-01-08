"""
tests/test_health.py

Smoke test for the health endpoint.
"""

from __future__ import annotations


def test_health_route(client):
    resp = client.get("/health/")
    assert resp.status_code == 200
    assert b"ok" in resp.content
