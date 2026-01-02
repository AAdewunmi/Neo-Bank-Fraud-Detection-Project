# dashboard/urls.py
"""
Dashboard URL configuration.

"""

from __future__ import annotations
from django.urls import path
from dashboard import views, export_views

app_name = "dashboard"

urlpatterns = [
    # Main dashboard page
    path("", views.index, name="index"),
    path("performance/", views.performance, name="performance"),
    path("reset/", views.reset_run, name="reset_run"),

    # Namespaced export endpoints (for {% url 'dashboard:...' %})
    path("export/flagged/", export_views.export_flagged_csv, name="export_flagged"),
    path("export/all/", export_views.export_all_csv, name="export_all"),

    # Aliases (for {% url 'export_flagged' %} etc. — backwards compatibility)
    path("export/flagged/", export_views.export_flagged_csv, name="export_flagged_global"),
    path("export/all/", export_views.export_all_csv, name="export_all_global"),
]
