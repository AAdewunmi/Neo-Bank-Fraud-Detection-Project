# tests/test_export_views_edge_cases.py
import pytest


@pytest.mark.django_db
def test_export_flagged_csv_no_session_returns_400(client):
    resp = client.get("/ops/export/flagged/")
    assert resp.status_code == 400
    assert b"No scored run" in resp.content


@pytest.mark.django_db
def test_export_all_csv_no_session_returns_400(client):
    resp = client.get("/ops/export/all/")
    assert resp.status_code == 400
    assert b"No scored run" in resp.content
