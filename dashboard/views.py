"""
Views for the dashboard app.

Week 2 intent:
- /ops/ is an internal Fraud Ops dashboard.
- Upload -> validate contract -> score -> render table + KPIs.
- Store scored rows in session so export can work without re-scoring.
"""
from __future__ import annotations

from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

import dashboard.services as services
from .forms import UploadForm
from .session_store import load_diags, load_scored_rows, save_scored_run


def public_home(request: HttpRequest) -> HttpResponse:
    """
    Minimal customer-facing landing page.

    Args:
        request: Django request.

    Returns:
        Rendered response.
    """
    return render(request, "dashboard/public_home.html", {})


@login_required
def index(request: HttpRequest) -> HttpResponse:
    """
    Ops dashboard: upload and score a CSV, then display results.

    GET:
        - If a prior scored run is stored in session, render it.
        - Otherwise render empty state.

    POST:
        - Validate upload, run ingestion contract, score, store in session, render results.

    Args:
        request: Django request.

    Returns:
        Rendered response.
    """
    context = {"form": UploadForm()}

    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        context["form"] = form

        if form.is_valid():
            try:
                df = services.read_csv(request.FILES["csv_file"])
                scored, diags = services.score_df(df, threshold=form.cleaned_data["threshold"])

                scored_rows = scored.to_dict(orient="records")
                save_scored_run(request.session, scored_rows, diags)

                pct_flagged = float(diags.get("pct_flagged", 0.0)) * 100
                pct_auto_cat = float(diags.get("pct_auto_categorised", 0.0)) * 100
                context.update(
                    {
                        "kpi": {
                            "tx_count": int(diags.get("n", len(scored))),
                            "pct_flagged": f"{pct_flagged:.1f}%",
                            "pct_auto_cat": f"{pct_auto_cat:.1f}%",
                        },
                        "table": scored_rows,
                        "threshold": float(diags.get("threshold", form.cleaned_data["threshold"])),
                        "export_available": True,
                    }
                )
            except Exception as exc:
                context["error"] = str(exc)
        else:
            context["error"] = "Invalid form input."

    if request.method == "GET":
        rows = load_scored_rows(request.session)
        diags = load_diags(request.session)
        if rows and diags:
            pct_flagged = float(diags.get("pct_flagged", 0.0)) * 100
            pct_auto_cat = float(diags.get("pct_auto_categorised", 0.0)) * 100
            context.update(
                {
                    "kpi": {
                        "tx_count": int(diags.get("n", len(rows))),
                        "pct_flagged": f"{pct_flagged:.1f}%",
                        "pct_auto_cat": f"{pct_auto_cat:.1f}%",
                    },
                    "table": rows,
                    "threshold": float(diags.get("threshold", 0.0)),
                    "export_available": True,
                }
            )

    return render(request, "dashboard/index.html", context)
