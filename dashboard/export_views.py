"""
CSV export views for the dashboard app.

Contract:
- Export reads from session payload written during scoring.
- Export does not re-run scoring.
- Export includes only flagged rows.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from django.http import HttpRequest, HttpResponse

from .decorators import ops_access_required
from .session_store import filter_flagged_rows, load_scored_run


def _filename_ts() -> str:
    """
    UTC timestamp for filenames.

    Returns:
        Timestamp string YYYYMMDD-HHMMSS.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


@ops_access_required
def export_flagged_csv(request: HttpRequest) -> HttpResponse:
    """
    Export only flagged rows from the last scoring run.

    Behavior:
    - If no scored run exists: 400 with helpful message.
    - If run exists but 0 flagged rows: CSV contains headers only.
    - Content-Type: text/csv
    - Content-Disposition set for file download.

    Args:
        request: Django request.

    Returns:
        HttpResponse with CSV body.
    """
    run = load_scored_run(request.session)
    if run is None or not run.fieldnames:
        return HttpResponse(
            "No export data found. Upload and score a CSV first.",
            status=400,
            content_type="text/plain; charset=utf-8",
        )

    flagged_rows = filter_flagged_rows(run)

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=run.fieldnames, extrasaction="ignore")
    writer.writeheader()

    for row in flagged_rows:
        writer.writerow({k: row.get(k, "") for k in run.fieldnames})

    csv_text = buf.getvalue()
    buf.close()

    filename = f"flagged-transactions-{_filename_ts()}.csv"
    resp = HttpResponse(csv_text, content_type="text/csv; charset=utf-8")
    resp["Content-Disposition"] = f'attachment; filename="{filename}"'
    resp["Cache-Control"] = "no-store"
    return resp
