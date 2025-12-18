"""
Views for the dashboard app.
dashboard/views.py

Week 1 intent:
- render an upload form
- validate CSV via the ingestion contract
- run scoring and render a simple table with KPIs

Note:
- importing the services module (not individual functions) makes tests easier to patch.
"""
from __future__ import annotations

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from .forms import UploadForm
from . import services


def index(request: HttpRequest) -> HttpResponse:
    """
    Render the dashboard. On POST, validate the CSV, score it, and render results.

    Args:
        request: Django request object.

    Returns:
        HTML response.
    """
    context = {"form": UploadForm()}

    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        context["form"] = form

        if form.is_valid():
            try:
                df = services.read_csv(request.FILES["csv_file"])
                scored, diags = services.score_df(df, threshold=form.cleaned_data["threshold"])

                context.update(
                    {
                        "kpi": {
                            "tx_count": int(len(scored)),
                            "pct_flagged": f"{diags['pct_flagged'] * 100:.1f}%",
                            "pct_auto_cat": f"{diags['pct_auto_categorised'] * 100:.1f}%",
                        },
                        "table": scored.to_dict(orient="records"),
                        "threshold": float(diags["threshold"]),
                    }
                )
            except Exception as exc:
                context["error"] = str(exc)
        else:
            context["error"] = "Invalid form input."

    return render(request, "dashboard/index.html", context)
