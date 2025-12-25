"""
Export views for Ops dashboard.

Exports are session-backed:
- no re-scoring on export
- export is consistent with what the user last scored
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO
from typing import Dict, List

from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse

from .session_store import load_scored_rows


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
    rows = load_scored_rows(request.session)
    if not rows:
        return HttpResponse(
            "No export data found. Upload and score a CSV first.",
            status=400,
            content_type="text/plain; charset=utf-8",
        )

    flagged: List[Dict] = [r for r in rows if bool(r.get("flagged")) is True]
    fieldnames = list(rows[0].keys()) if rows else []

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for r in flagged:
        writer.writerow({k: r.get(k, "") for k in fieldnames})

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
    rows = load_scored_rows(request.session)
    if not rows:
        return HttpResponse(
            "No export data found. Upload and score a CSV first.",
            status=400,
            content_type="text/plain; charset=utf-8",
        )

    fieldnames = list(rows[0].keys())
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r.get(k, "") for k in fieldnames})

    csv_text = buf.getvalue()
    buf.close()

    filename = f"all-transactions-{_filename_ts()}.csv"
    resp = HttpResponse(csv_text, content_type="text/csv; charset=utf-8")
    resp["Content-Disposition"] = f'attachment; filename="{filename}"'
    resp["Cache-Control"] = "no-store"
    return resp
