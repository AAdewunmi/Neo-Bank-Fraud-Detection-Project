"""
tests/test_dashboard_permissions.py

Tests for dashboard access-control rules.

Focus:
- ops_access_required decorator behaviour under different settings.
- Ensures login and staff requirements are enforced via simple, explicit cases.

These tests are intentionally isolated from URL routing. They exercise the
decorator contract directly using RequestFactory and lightweight fake users.
"""

from __future__ import annotations

import types

import pytest
from django.http import HttpRequest, HttpResponse
from django.test import RequestFactory, override_settings

from dashboard.decorators import ops_access_required


@pytest.fixture
def rf() -> RequestFactory:
    """
    Provide a Django RequestFactory instance for building HttpRequest objects.

    Returns:
        A RequestFactory instance.
    """
    return RequestFactory()


def _dummy_view(request: HttpRequest, *args, **kwargs) -> HttpResponse:
    """
    Simple view used as the target for the ops_access_required decorator.

    Args:
        request: Django HttpRequest.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Basic HTTP 200 response with a small body.
    """
    return HttpResponse("OK", status=200)


@override_settings(
    DASHBOARD_REQUIRE_LOGIN=True,
    DASHBOARD_REQUIRE_STAFF=True,
    LOGIN_URL="/accounts/login/",
)
def test_ops_access_redirects_anonymous_when_login_required(rf: RequestFactory) -> None:
    """
    Anonymous users must be redirected to the login page
    when DASHBOARD_REQUIRE_LOGIN is enabled.
    """
    request = rf.get("/ops/")
    # Lightweight fake user object with the attributes the decorator expects.
    request.user = types.SimpleNamespace(is_authenticated=False, is_staff=False)

    guarded_view = ops_access_required(_dummy_view)
    response = guarded_view(request)

    assert response.status_code == 302
    assert "/accounts/login/" in response["Location"]


@override_settings(
    DASHBOARD_REQUIRE_LOGIN=True,
    DASHBOARD_REQUIRE_STAFF=True,
)
def test_ops_access_forbids_non_staff_user_when_staff_required(rf: RequestFactory) -> None:
    """
    Authenticated non-staff users receive HTTP 403 when staff access is required.
    """
    request = rf.get("/ops/")
    request.user = types.SimpleNamespace(is_authenticated=True, is_staff=False)

    guarded_view = ops_access_required(_dummy_view)
    response = guarded_view(request)

    assert response.status_code == 403
    body = response.content.decode("utf-8")
    assert "Staff access required." in body


@override_settings(
    DASHBOARD_REQUIRE_LOGIN=True,
    DASHBOARD_REQUIRE_STAFF=True,
)
def test_ops_access_allows_staff_user_when_requirements_met(rf: RequestFactory) -> None:
    """
    Staff users who are authenticated pass through to the underlying view.
    """
    request = rf.get("/ops/")
    request.user = types.SimpleNamespace(is_authenticated=True, is_staff=True)

    guarded_view = ops_access_required(_dummy_view)
    response = guarded_view(request)

    assert response.status_code == 200
    assert response.content.decode("utf-8") == "OK"


@override_settings(
    DASHBOARD_REQUIRE_LOGIN=False,
    DASHBOARD_REQUIRE_STAFF=False,
)
def test_ops_access_allows_when_checks_disabled(rf: RequestFactory) -> None:
    """
    When both DASHBOARD_REQUIRE_LOGIN and DASHBOARD_REQUIRE_STAFF are disabled,
    the decorator should behave like a no-op for access control.
    """
    request = rf.get("/ops/")
    request.user = types.SimpleNamespace(is_authenticated=False, is_staff=False)

    guarded_view = ops_access_required(_dummy_view)
    response = guarded_view(request)

    assert response.status_code == 200
    assert response.content.decode("utf-8") == "OK"
