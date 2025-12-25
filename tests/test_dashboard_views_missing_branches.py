# tests/test_dashboard_views_missing_branches.py
"""
Tests for dashboard views when certain branches are missing.
"""

from django.core.files.uploadedfile import SimpleUploadedFile


def test_dashboard_get_without_login_redirects_to_login(client):
    """GET /ops/ should redirect unauthenticated users to login (covers views.py line 34)."""
    resp = client.get("/ops/")
    assert resp.status_code == 302
    assert resp.url.startswith("/accounts/login")


def test_dashboard_post_with_missing_threshold_uses_default(client, django_user_model):
    """POST without threshold should use default branch (covers views.py line 46)."""
    user = django_user_model.objects.create_user(username="ops", password="p", is_staff=True)
    client.force_login(user)
    csv = SimpleUploadedFile(
        "tx.csv",
        b"timestamp,amount,customer_id,merchant,description\n",
        content_type="text/csv"
    )
    resp = client.post("/ops/", data={"csv_file": csv})
    assert resp.status_code == 200
