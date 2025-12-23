"""
Dashboard access-control decorators.

Week 2 policy:
- Ops area is internal.
- Require login by default.
- Require staff by default (toggle via env/settings).

Settings:
- DASHBOARD_REQUIRE_LOGIN (bool): default True
- DASHBOARD_REQUIRE_STAFF (bool): default True
"""
from __future__ import annotations

from functools import wraps
from typing import Any, Callable, TypeVar

from django.conf import settings
from django.contrib.auth.views import redirect_to_login
from django.http import HttpRequest, HttpResponse, HttpResponseForbidden

F = TypeVar("F", bound=Callable[..., HttpResponse])


def ops_access_required(view_func: F) -> F:
    """
    Enforce access rules for ops endpoints.

    Args:
        view_func: Django view callable.

    Returns:
        Wrapped callable enforcing login/staff rules.
    """

    @wraps(view_func)
    def _wrapped(request: HttpRequest, *args: Any, **kwargs: Any) -> HttpResponse:
        require_login = bool(getattr(settings, "DASHBOARD_REQUIRE_LOGIN", True))
        require_staff = bool(getattr(settings, "DASHBOARD_REQUIRE_STAFF", True))

        if require_login and not request.user.is_authenticated:
            return redirect_to_login(
                request.get_full_path(),
                login_url=getattr(settings, "LOGIN_URL", "/accounts/login/"),
            )

        if require_staff and not bool(getattr(request.user, "is_staff", False)):
            return HttpResponseForbidden("Staff access required.")

        return view_func(request, *args, **kwargs)

    return _wrapped  # type: ignore[return-value]
