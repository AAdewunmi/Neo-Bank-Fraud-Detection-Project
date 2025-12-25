# neobank_site/urls.py
"""
Project URL configuration.

"""

from django.contrib import admin
from django.urls import include, path
from dashboard import views as dashboard_views

urlpatterns = [
    # Public homepage
    path("", dashboard_views.public_home, name="public_home"),

    # Dashboard (namespaced and included)
    path("ops/", include(("dashboard.urls", "dashboard"), namespace="dashboard")),
    path("ops/", dashboard_views.index, name="dashboard"),  # plain alias

    # Auth endpoints
    path("accounts/", include("django.contrib.auth.urls")),
    path("admin/", admin.site.urls),
]
