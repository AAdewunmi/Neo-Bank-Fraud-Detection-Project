"""Auth boundary tests for customer and ops logins."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.django_db


def test_customer_login_rejects_staff_user(client, django_user_model) -> None:
    user = django_user_model.objects.create_user(
        username="ops_staff",
        password="pass1234",
        is_staff=True,
    )

    resp = client.post(
        "/customer/login/",
        data={"username": user.username, "password": "pass1234"},
    )

    assert resp.status_code == 200
    assert b"Staff accounts cannot sign in here" in resp.content
    assert "_auth_user_id" not in client.session
    follow_up = client.get("/customer/")
    assert not follow_up.wsgi_request.user.is_authenticated


def test_ops_login_rejects_customer_user(client, django_user_model) -> None:
    user = django_user_model.objects.create_user(
        username="cust_user",
        password="pass1234",
        is_staff=False,
    )

    resp = client.post(
        "/accounts/login/",
        data={"username": user.username, "password": "pass1234"},
    )

    assert resp.status_code == 200
    assert b"Customer accounts cannot sign in here" in resp.content
    assert "_auth_user_id" not in client.session
    follow_up = client.get("/ops/")
    assert not follow_up.wsgi_request.user.is_authenticated
