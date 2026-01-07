"""Middleware helpers for host-scoped session and CSRF cookies."""

from __future__ import annotations

from typing import Tuple

from django.conf import settings


def _cookie_names_for_host(host: str) -> Tuple[str, str]:
    hostname = host.split(":", 1)[0].lower()
    if hostname.startswith("ops."):
        return ("ops_sessionid", "ops_csrftoken")
    if hostname.startswith("customer."):
        return ("customer_sessionid", "customer_csrftoken")
    return (settings.SESSION_COOKIE_NAME, settings.CSRF_COOKIE_NAME)


class HostScopedCookieMiddleware:
    """Map session/CSRF cookies to host-specific names for subdomains."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        host = request.get_host()
        session_cookie_name, csrf_cookie_name = _cookie_names_for_host(host)

        request._host_scoped_cookie_names = (session_cookie_name, csrf_cookie_name)

        # Map host-scoped cookies to the default names expected by Django middleware.
        if session_cookie_name != settings.SESSION_COOKIE_NAME:
            if session_cookie_name in request.COOKIES:
                request.COOKIES[settings.SESSION_COOKIE_NAME] = request.COOKIES[session_cookie_name]
        if csrf_cookie_name != settings.CSRF_COOKIE_NAME:
            if csrf_cookie_name in request.COOKIES:
                request.COOKIES[settings.CSRF_COOKIE_NAME] = request.COOKIES[csrf_cookie_name]

        response = self.get_response(request)

        # Rewrite Set-Cookie headers back to host-scoped names.
        if session_cookie_name != settings.SESSION_COOKIE_NAME:
            self._rewrite_cookie(response, settings.SESSION_COOKIE_NAME, session_cookie_name)
        if csrf_cookie_name != settings.CSRF_COOKIE_NAME:
            self._rewrite_cookie(response, settings.CSRF_COOKIE_NAME, csrf_cookie_name)

        return response

    @staticmethod
    def _rewrite_cookie(response, old_name: str, new_name: str) -> None:
        if old_name not in response.cookies:
            return
        morsel = response.cookies.pop(old_name)
        response.cookies[new_name] = morsel.value
        new_morsel = response.cookies[new_name]
        for key in morsel.keys():
            new_morsel[key] = morsel[key]
