"""
Views for the Week 1 dashboard scaffold.

Week 1 goal: prove request routing + template rendering + test harness.
"""
from __future__ import annotations

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render


def index(request: HttpRequest) -> HttpResponse:
    """
    Render a placeholder page for Week 1 scaffolding.

    Args:
        request: Django request object.

    Returns:
        An HTTP response containing the rendered template.
    """
    return render(request, "dashboard/index.html")
