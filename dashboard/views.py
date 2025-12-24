"""
Views for the dashboard app.

Week 2 intent:
- ops dashboard under /ops/
- upload -> score -> render table + KPIs
- store the scored run in session
- export reads from session
- filters apply without re-scoring (GET-based)
"""
from __future__ import annotations

from typing import Any, Dict, List

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from . import services
from .decorators import ops_access_required
from .forms import FilterForm, UploadForm
from .session_store import build_scored_run, load_scored_run, save_scored_run


def public_home(request: HttpRequest) -> HttpResponse:
    """
    Customer-facing landing page.

    Args:
        request: Django request.

    Returns:
        Rendered HTML response.
    """
    return render(request, "dashboard/public_home.html", {})


def _apply_filters(rows: List[Dict[str, Any]], filter_form: FilterForm) -> List[Dict[str, Any]]:
    """
    Apply server-side filters to session-backed scored rows.

    Args:
        rows: Scored rows.
        filter_form: FilterForm bound to request.GET.

    Returns:
        Filtered rows.
    """
    if not filter_form.is_valid():
        return rows

    data = filter_form.cleaned_data
    out = rows

    if bool(data.get("flagged_only")):
        out = [r for r in out if bool(r.get("flagged")) is True]

    customer_id = (data.get("customer_id") or "").strip()
    if customer_id:
        out = [r for r in out if str(r.get("customer_id", "")).strip() == customer_id]

    merchant = (data.get("merchant") or "").strip().lower()
    if merchant:
        out = [r for r in out if merchant in str(r.get("merchant", "")).lower()]

    category = (data.get("category") or "").strip().lower()
    if category:
        out = [r for r in out if category in str(r.get("category", "")).lower()]

    min_fraud_risk = data.get("min_fraud_risk")
    if min_fraud_risk is not None:
        try:
            m = float(min_fraud_risk)
            out = [r for r in out if float(r.get("fraud_risk", 0.0) or 0.0) >= m]
        except Exception:
            pass

    return out


@ops_access_required
def index(request: HttpRequest) -> HttpResponse:
    """
    Ops dashboard: upload, score, filter, export.

    GET:
      - if a run exists in session, show it (filtered by GET params).
      - otherwise show empty state.

    POST:
      - validate ingestion contract
      - score
      - store session payload
      - render results

    Args:
        request: Django request.

    Returns:
        Rendered HTML response.
    """
    upload_form = UploadForm()
    filter_form = FilterForm(request.GET or None)

    error: str | None = None
    threshold_value: float | None = None
    table_rows: List[Dict[str, Any]] | None = None
    kpi: Dict[str, Any] | None = None
    run_meta: Dict[str, Any] | None = None
    visible_count: int | None = None
    export_available = False

    if request.method == "POST":
        upload_form = UploadForm(request.POST, request.FILES)

        if upload_form.is_valid():
            try:
                df = services.read_csv(request.FILES["csv_file"])
                threshold = upload_form.cleaned_data["threshold"]
                scored_df, diags = services.score_df(df, threshold=threshold)

                scored_rows = scored_df.to_dict(orient="records")

                run = build_scored_run(
                    scored_rows,
                    threshold=float(diags["threshold"]),
                    pct_flagged=float(diags["pct_flagged"]),
                    pct_auto_categorised=float(diags["pct_auto_categorised"]),
                    flagged_key="flagged",
                )
                save_scored_run(request.session, run)
            except Exception as exc:
                error = str(exc)
        else:
            error = "Invalid form input."

    run = load_scored_run(request.session)
    if run is not None and run.rows:
        threshold_value = float(run.run_meta.get("threshold", 0.0) or 0.0)
        run_meta = dict(run.run_meta)

        filtered = _apply_filters(list(run.rows), filter_form)
        table_rows = filtered
        visible_count = len(filtered)

        tx_count = int(run_meta.get("tx_count", 0) or 0)
        flagged_count = int(run_meta.get("flagged_count", 0) or 0)
        pct_flagged = float(run_meta.get("pct_flagged", 0.0) or 0.0)
        pct_auto_cat = float(run_meta.get("pct_auto_categorised", 0.0) or 0.0)

        kpi = {
            "tx_count": tx_count,
            "pct_flagged": f"{pct_flagged * 100:.1f}%",
            "pct_auto_cat": f"{pct_auto_cat * 100:.1f}%",
            "flagged_count": flagged_count,
        }

        export_available = True

    context: Dict[str, Any] = {
        "form": upload_form,
        "filter_form": filter_form,
        "error": error,
        "table": table_rows,
        "threshold": threshold_value,
        "kpi": kpi,
        "run_meta": run_meta,
        "view_meta": {"visible_count": visible_count} if visible_count is not None else None,
        "export_available": export_available,
    }
    return render(request, "dashboard/index.html", context)
