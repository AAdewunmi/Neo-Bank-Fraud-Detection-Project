"""Customer-facing views (read-only surface)."""

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render


def home(request: HttpRequest) -> HttpResponse:
    """Render the customer landing page."""
    return render(request, "customer/home.html")
