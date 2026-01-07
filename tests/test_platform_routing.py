"""Routing smoke tests for customer and ops entrypoints."""

import pytest

pytestmark = pytest.mark.django_db


def test_customer_home_ok(client) -> None:
    """Public home page should be reachable without auth."""
    resp = client.get("/")
    assert resp.status_code == 200


def test_customer_site_ok(client) -> None:
    """Customer site should be reachable without auth."""
    resp = client.get("/customer/")
    assert resp.status_code == 200


def test_ops_requires_login(client) -> None:
    """Ops area should redirect unauthenticated users to login."""
    resp = client.get("/ops/")
    assert resp.status_code == 302
    assert "/accounts/login/" in resp["Location"]


def test_ops_ok_after_login(client, django_user_model) -> None:
    """Ops area should load for a staff user once authenticated."""
    user = django_user_model.objects.create_user(
        username="ops", password="pass1234", is_staff=True
    )
    # Force session auth to avoid relying on the login view behavior here.
    client.force_login(user)

    resp = client.get("/ops/")
    assert resp.status_code == 200
