# tests/test_export_views_edge_cases.py
# tests/test_export_views_edge_cases.py
import pytest


@pytest.mark.django_db
def test_export_flagged_csv_no_session_redirects_to_login(client):
    resp = client.get("/ops/export/flagged/")
    assert resp.status_code == 302
    assert resp.url.startswith("/accounts/login")


@pytest.mark.django_db
def test_export_all_csv_no_session_redirects_to_login(client):
    resp = client.get("/ops/export/all/")
    assert resp.status_code == 302
    assert resp.url.startswith("/accounts/login")
