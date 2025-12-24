"""
Views for the dashboard app.

Week 2 intent

- Ops dashboard under /ops/
- Upload -> score -> render table and KPIs
- Store the scored run in the session
- Export reads from the session
- Filters apply without re-scoring (GET-based)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from . import services
from .decorators import ops_access_required
from .forms import FilterForm, UploadForm
from .session_store import ScoredRun, build_scored_run, load_scored_run, save_scored_run


def public_home(request: HttpRequest) -> HttpResponse:
    """
    Public landing page.

    The fraud operations dashboard itself lives under /ops/ and is
    protected by access controls. This view remains safe to expose
    publicly while internal surfaces evolve.
    """
    return render(request, "dashboard/public_home.html")


def _apply_filters(rows: List[Dict[str, Any]], filter_form: FilterForm) -> List[Dict[str, Any]]:
    """
    Apply server-side filters to already-scored rows.

    This helper is side-effect free so it is simple to test. The
    dashboard view is responsible for supplying the cleaned filter
    data and for handling any validation errors at the form layer.
    """
    if not rows:
        return []

    if not filter_form.is_valid():
        # If filters are invalid fall back to the unfiltered table.
        return list(rows)

    cleaned = filter_form.cleaned_data
    flagged_only = cleaned.get("flagged_only")
    category = cleaned.get("category") or None

    filtered: List[Dict[str, Any]] = []

    for row in rows:
        if flagged_only and not row.get("flagged", False):
            continue

        if category is not None and row.get("category") != category:
            continue

        filtered.append(row)

    return filtered


@ops_access_required
def index(request: HttpRequest) -> HttpResponse:
    """
    Main fraud operations dashboard view.

    Responsibilities

    - Accept a CSV upload and invoke the scoring pipeline.
    - Persist the last scored run in the session.
    - Apply server-side filters for the visible table.
    """
    upload_form = UploadForm()
    filter_form = FilterForm(request.GET or None)

    error: Optional[str] = None
    threshold_value: Optional[float] = None
    table_rows: List[Dict[str, Any]] = []
    run_meta: Dict[str, Any] = {}
    kpi: Dict[str, Any] = {}
    visible_count: Optional[int] = None
    export_available = False

    if request.method == "POST":
        upload_form = UploadForm(request.POST, request.FILES)

        if upload_form.is_valid():
            try:
                csv_file = upload_form.cleaned_data["csv_file"]
                threshold = upload_form.cleaned_data["threshold"]

                df = services.read_csv(csv_file)
                scored_df, diags = services.score_df(df, threshold=threshold)

                flagged_key = diags.get("flagged_key", "flagged")
                scored_rows: List[Dict[str, Any]] = scored_df.to_dict(orient="records")

                run: ScoredRun = build_scored_run(
                    scored_rows,
                    threshold=diags.get("threshold", threshold),
                    pct_flagged=diags.get("pct_flagged", 0.0),
                    pct_auto_categorised=diags.get("pct_auto_categorised", 0.0),
                    flagged_key=flagged_key,
                )

                save_scored_run(request.session, run)
            except Exception as exc:  # pragma: no cover - defensive
                # Any unexpected failure in parsing or scoring is surfaced
                # as a single error string while leaving any previous run
                # in the session untouched.
                error = str(exc)
        else:
            # Form validation errors are rendered next to the fields.
            error = "Invalid form input."

    run = load_scored_run(request.session)

    if run is not None:
        run_meta = run.run_meta
        threshold_value = run_meta.get("threshold")

        # Filters are applied to the already-scored rows from the session.
        table_rows = _apply_filters(list(run.rows), filter_form)
        visible_count = len(table_rows)

        kpi = {
            "tx_count": run_meta.get("tx_count"),
            "pct_flagged": run_meta.get("pct_flagged"),
            "pct_auto_cat": run_meta.get("pct_auto_categorised"),
            "flagged_count": run_meta.get("flagged_count"),
        }

        export_available = bool(table_rows)

    view_meta: Dict[str, Any] = {
        "has_run": bool(run and run.rows),
        "visible_count": visible_count,
    }

    context = {
        "form": upload_form,
        "filter_form": filter_form,
        "error": error,
        "table": table_rows,
        "threshold": threshold_value,
        "kpi": kpi,
        "run_meta": run_meta,
        "view_meta": view_meta,
        "export_available": export_available,
    }

    return render(request, "dashboard/index.html", context)
