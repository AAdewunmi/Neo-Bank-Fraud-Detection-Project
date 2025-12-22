"""
Access control decorators for dashboard views.
"""
from __future__ import annotations

from functools import wraps
from typing import Any, Callable, TypeVar, cast

from django.http import HttpRequest, HttpResponse, HttpResponseForbidden

F = TypeVar("F", bound=Callable[..., HttpResponse])


def ops_access_required(view_func: F) -> F:
    """
    Allow access only to authenticated staff users.
    """

    @wraps(view_func)
    def _wrapped(request: HttpRequest, *args: Any, **kwargs: Any) -> HttpResponse:
        user = getattr(request, "user", None)
        if user is None or not getattr(user, "is_authenticated", False):
            return HttpResponseForbidden("Authentication required.")
        if not getattr(user, "is_staff", False):
            return HttpResponseForbidden("Staff access required.")
        return view_func(request, *args, **kwargs)

    return cast(F, _wrapped)
