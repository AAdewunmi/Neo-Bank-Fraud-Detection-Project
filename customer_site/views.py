"""Customer-facing views (read-only surface)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from dashboard.session_store import load_scored_run
from dashboard.views import _ensure_row_ids, _get_category_edits, _overlay_category_edits
from customer_site.services import build_spend_summary

SAFE_FIELDS = ("row_id", "timestamp", "merchant", "description", "amount", "category")


def _build_customer_rows(
    rows: List[Mapping[str, Any]],
    edits: Dict[str, Dict[str, Any]],
    max_rows: int,
) -> List[Dict[str, Any]]:
    rows_with_ids = _ensure_row_ids(list(rows))
    display_rows = _overlay_category_edits(rows_with_ids, edits)

    safe_rows: List[Dict[str, Any]] = []
    for row in display_rows[:max_rows]:
        safe_rows.append({field: row.get(field, "") for field in SAFE_FIELDS})
    return safe_rows


def home(request: HttpRequest) -> HttpResponse:
    """Render the customer landing page."""
    scored_run = load_scored_run(request.session)
    base_rows = scored_run.rows if scored_run else []
    run_meta = scored_run.run_meta if scored_run else {}

    if not base_rows:
        context = {"rows": [], "has_rows": False, "total_count": 0, "rows_shown": 0}
        return render(request, "customer/home.html", context)

    max_rows = int(os.environ.get("LEDGERGUARD_CUSTOMER_MAX_ROWS", "200"))
    edits = _get_category_edits(request.session)
    safe_rows = _build_customer_rows(base_rows, edits, max_rows)
    summary = build_spend_summary(safe_rows, max_categories=6)

    total_count = int(run_meta.get("tx_count") or len(base_rows))
    context = {
        "rows": safe_rows,
        "has_rows": bool(safe_rows),
        "total_count": total_count,
        "rows_shown": len(safe_rows),
        "summary": summary,
    }
    return render(request, "customer/home.html", context)
