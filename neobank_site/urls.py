"""
Project URL configuration.

Week 2 intent:
- Customer site at /
- Ops dashboard at /ops/ (internal)
- Auth pages at /accounts/ (login/logout)
"""
from __future__ import annotations

from django.contrib import admin
from django.urls import include, path

from dashboard.views import public_home

urlpatterns = [
    path("", public_home, name="home"),
    path("", public_home, name="public_home"),
    path("app/", public_home, name="app_home"),
    path("ops/", include("dashboard.urls")),
    path("accounts/", include("django.contrib.auth.urls")),
    path("admin/", admin.site.urls),
]
