# neobank_site/urls.py
"""
Project URL configuration.

Includes a lightweight healthcheck route suitable for platform probes.
"""

from django.contrib import admin
from django.http import JsonResponse
from django.urls import include, path
from dashboard import views as dashboard_views


def health(_request):
    """Healthcheck endpoint for liveness checks."""
    return JsonResponse({"ok": True})

urlpatterns = [
    # Public homepage
    path("", dashboard_views.public_home, name="public_home"),

    # Customer site
    path(
        "customer/",
        include(("customer_site.urls", "customer"), namespace="customer"),
    ),

    # Dashboard (main app)
    path("ops/", dashboard_views.index, name="dashboard"),
    path("ops/", include(("dashboard.urls", "dashboard"), namespace="dashboard")),

    # Auth endpoints
    path("accounts/login/", dashboard_views.ops_login, name="login"),
    path("accounts/", include("django.contrib.auth.urls")),

    # Admin site
    path("admin/", admin.site.urls),

    # Healthcheck
    path("health/", health, name="health"),
]
