"""
URL routes for the dashboard app.

This app is mounted under /ops/ at the project level.
"""
from __future__ import annotations

from django.urls import path

from . import export_views, views

urlpatterns = [
    path("", views.index, name="dashboard"),
    path("export/flagged/", export_views.export_flagged_csv, name="export_flagged"),
]
