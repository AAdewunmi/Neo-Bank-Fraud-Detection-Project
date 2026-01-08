"""Middleware helpers for host-scoped session and CSRF cookies."""

from __future__ import annotations

from typing import Tuple

from django.conf import settings
from django.shortcuts import redirect


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
                request.COOKIES[settings.SESSION_COOKIE_NAME] = request.COOKIES[
                    session_cookie_name
                ]
        if csrf_cookie_name != settings.CSRF_COOKIE_NAME:
            if csrf_cookie_name in request.COOKIES:
                request.COOKIES[settings.CSRF_COOKIE_NAME] = request.COOKIES[
                    csrf_cookie_name
                ]

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


class HostRoutingMiddleware:
    """Redirect to the correct subdomain for ops/customer routes."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if not settings.HOST_ROUTING_ENABLED:
            return self.get_response(request)
        response = self._redirect_for_host(request)
        if response is not None:
            return response
        return self.get_response(request)

    def _redirect_for_host(self, request):
        host = request.get_host()
        hostname, port = self._split_host(host)
        path = request.path

        if path.startswith(("/ops", "/accounts/", "/admin")):
            if hostname.startswith("ops."):
                return None
            target = settings.OPS_HOST
        elif path.startswith("/customer"):
            if hostname.startswith("customer."):
                return None
            if (
                hostname.startswith("ops.")
                and request.user.is_authenticated
                and request.user.is_staff
            ):
                return None
            target = settings.CUSTOMER_HOST
        else:
            return None

        target_host = self._with_port(target, port)
        url = f"{request.scheme}://{target_host}{request.get_full_path()}"
        return redirect(url)

    @staticmethod
    def _split_host(host: str) -> Tuple[str, str | None]:
        if ":" in host:
            name, port = host.split(":", 1)
            return name, port
        return host, None

    @staticmethod
    def _with_port(host: str, port: str | None) -> str:
        if ":" in host or not port:
            return host
        return f"{host}:{port}"
