# tests/test_dashboard_views_errors.py
from django.core.files.uploadedfile import SimpleUploadedFile


def test_dashboard_post_invalid_csv_shows_error(client, django_user_model):
    """Ensure invalid CSV uploads are handled gracefully (no crash)."""
    user = django_user_model.objects.create_user(username="ops", password="pass", is_staff=True)
    client.force_login(user)

    bad_csv = SimpleUploadedFile("bad.csv", b"col1,col2\n1,2\n", content_type="text/csv")
    resp = client.post("/ops/", data={"threshold": 0.7, "csv_file": bad_csv})
    assert resp.status_code == 200
    # just ensure page renders normally and contains upload form again
    assert b"<form" in resp.content
