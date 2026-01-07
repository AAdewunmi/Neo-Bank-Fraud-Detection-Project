"""
Export views for Ops dashboard.

Exports are DB-backed:
- no re-scoring on export
- export reflects the latest persisted scoring run
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse

from customer_site.models import CustomerTransaction


def _filename_ts() -> str:
    """
    Build a UTC timestamp for the export filename.

    Returns:
        Timestamp string YYYYMMDD-HHMMSS.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


@login_required
def export_flagged_csv(request: HttpRequest) -> HttpResponse:
    """
    Export only flagged rows from the last scored run.

    Rules:
    - 400 if no scored data in session
    - CSV includes headers
    - CSV contains only rows where flagged is True

    Args:
        request: Django request.

    Returns:
        CSV download response.
    """
    latest_scored_at = (
        CustomerTransaction.objects.order_by("-scored_at")
        .values_list("scored_at", flat=True)
        .first()
    )
    if not latest_scored_at:
        return HttpResponse(
            "No export data found. Upload and score a CSV first.",
            status=400,
            content_type="text/plain; charset=utf-8",
        )

    flagged_qs = CustomerTransaction.objects.filter(
        scored_at=latest_scored_at,
        flagged=True,
    ).order_by("row_id")
    if not flagged_qs.exists():
        return HttpResponse(
            "No flagged rows available for export.",
            status=400,
            content_type="text/plain; charset=utf-8",
        )

    fieldnames = [
        "row_id",
        "timestamp",
        "customer_id",
        "amount",
        "merchant",
        "description",
        "category",
        "predicted_category",
        "category_source",
        "fraud_risk",
        "flagged",
    ]
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in flagged_qs.values(*fieldnames):
        writer.writerow({k: row.get(k, "") for k in fieldnames})

    csv_text = buf.getvalue()
    buf.close()

    filename = f"flagged-transactions-{_filename_ts()}.csv"
    resp = HttpResponse(csv_text, content_type="text/csv; charset=utf-8")
    resp["Content-Disposition"] = f'attachment; filename="{filename}"'
    resp["Cache-Control"] = "no-store"
    return resp


@login_required
def export_all_csv(request: HttpRequest) -> HttpResponse:
    """
    Export all rows from the last scored run.

    Rules:
    - 400 if no scored data in session
    - CSV includes headers

    Args:
        request: Django request.

    Returns:
        CSV download response.
    """
    latest_scored_at = (
        CustomerTransaction.objects.order_by("-scored_at")
        .values_list("scored_at", flat=True)
        .first()
    )
    if not latest_scored_at:
        return HttpResponse(
            "No export data found. Upload and score a CSV first.",
            status=400,
            content_type="text/plain; charset=utf-8",
        )

    fieldnames = [
        "row_id",
        "timestamp",
        "customer_id",
        "amount",
        "merchant",
        "description",
        "category",
        "predicted_category",
        "category_source",
        "fraud_risk",
        "flagged",
    ]
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    all_rows = CustomerTransaction.objects.filter(
        scored_at=latest_scored_at
    ).order_by("row_id")
    for row in all_rows.values(*fieldnames):
        writer.writerow({k: row.get(k, "") for k in fieldnames})

    csv_text = buf.getvalue()
    buf.close()

    filename = f"all-transactions-{_filename_ts()}.csv"
    resp = HttpResponse(csv_text, content_type="text/csv; charset=utf-8")
    resp["Content-Disposition"] = f'attachment; filename="{filename}"'
    resp["Cache-Control"] = "no-store"
    return resp
