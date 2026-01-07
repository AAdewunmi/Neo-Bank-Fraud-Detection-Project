"""Tests for host-scoped session/CSRF cookies."""
from __future__ import annotations

from django.conf import settings
from django.http import HttpResponse
from django.test import RequestFactory

from neobank_site.middleware import HostScopedCookieMiddleware


def _make_response_with_default_cookies() -> HttpResponse:
    resp = HttpResponse()
    resp.set_cookie(settings.SESSION_COOKIE_NAME, "session-value")
    resp.set_cookie(settings.CSRF_COOKIE_NAME, "csrf-value")
    return resp


def test_ops_host_rewrites_cookies_and_maps_request() -> None:
    rf = RequestFactory()
    request = rf.get("/", HTTP_HOST="ops.localhost")
    request.COOKIES["ops_sessionid"] = "ops-session"
    request.COOKIES["ops_csrftoken"] = "ops-csrf"

    def get_response(req):
        assert req.COOKIES.get(settings.SESSION_COOKIE_NAME) == "ops-session"
        assert req.COOKIES.get(settings.CSRF_COOKIE_NAME) == "ops-csrf"
        return _make_response_with_default_cookies()

    middleware = HostScopedCookieMiddleware(get_response)
    response = middleware(request)

    assert "ops_sessionid" in response.cookies
    assert "ops_csrftoken" in response.cookies
    assert settings.SESSION_COOKIE_NAME not in response.cookies
    assert settings.CSRF_COOKIE_NAME not in response.cookies


def test_customer_host_rewrites_cookies_and_maps_request() -> None:
    rf = RequestFactory()
    request = rf.get("/", HTTP_HOST="customer.localhost")
    request.COOKIES["customer_sessionid"] = "customer-session"
    request.COOKIES["customer_csrftoken"] = "customer-csrf"

    def get_response(req):
        assert req.COOKIES.get(settings.SESSION_COOKIE_NAME) == "customer-session"
        assert req.COOKIES.get(settings.CSRF_COOKIE_NAME) == "customer-csrf"
        return _make_response_with_default_cookies()

    middleware = HostScopedCookieMiddleware(get_response)
    response = middleware(request)

    assert "customer_sessionid" in response.cookies
    assert "customer_csrftoken" in response.cookies
    assert settings.SESSION_COOKIE_NAME not in response.cookies
    assert settings.CSRF_COOKIE_NAME not in response.cookies
