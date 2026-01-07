"""Customer-facing views (read-only surface)."""

from __future__ import annotations

import csv
import io
import logging
import os
import re
from typing import Any, Dict, List, Mapping

from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from dashboard.session_store import load_scored_run
from dashboard.views import _ensure_row_ids, _get_category_edits, _overlay_category_edits
from customer_site.services import build_spend_summary

SAFE_FIELDS = ("row_id", "timestamp", "merchant", "description", "amount", "category")
CUSTOMER_FLAGS_SESSION_KEY = "customer_flags"
MAX_REASON_LENGTH = 200
ROW_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SAFE_FIELDS_SET = set(SAFE_FIELDS)

logger = logging.getLogger(__name__)


def _get_customer_flags(session: Any) -> Dict[str, Dict[str, Any]]:
    raw = session.get(CUSTOMER_FLAGS_SESSION_KEY, {})
    if isinstance(raw, dict):
        return raw
    return {}


def _is_valid_row_id(row_id: str) -> bool:
    return bool(ROW_ID_PATTERN.match(row_id))


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


def _filter_customer_rows(rows: List[Mapping[str, Any]]) -> tuple[List[Dict[str, Any]], int]:
    filtered: List[Dict[str, Any]] = []
    dropped = 0
    for idx, row in enumerate(rows):
        unexpected = set(row.keys()) - SAFE_FIELDS_SET
        if unexpected:
            unexpected_list = ", ".join(sorted(unexpected))
            logger.warning(
                "Dropping customer row %s due to unexpected keys: %s",
                idx,
                unexpected_list,
            )
            dropped += 1
            continue
        filtered.append(dict(row))
    return filtered, dropped


def home(request: HttpRequest) -> HttpResponse:
    """Render the customer landing page."""
    scored_run = load_scored_run(request.session)
    base_rows = scored_run.rows if scored_run else []
    run_meta = scored_run.run_meta if scored_run else {}
    flags = _get_customer_flags(request.session)

    if not base_rows:
        context = {
            "rows": [],
            "has_rows": False,
            "total_count": 0,
            "rows_shown": 0,
            "flags": flags,
        }
        return render(request, "customer/home.html", context)

    max_rows = int(os.environ.get("LEDGERGUARD_CUSTOMER_MAX_ROWS", "200"))
    edits = _get_category_edits(request.session)
    safe_rows = _build_customer_rows(base_rows, edits, max_rows)
    safe_rows, dropped_rows = _filter_customer_rows(safe_rows)
    summary = build_spend_summary(safe_rows, max_categories=6)

    total_count = int(run_meta.get("tx_count") or len(base_rows))
    context = {
        "rows": safe_rows,
        "has_rows": bool(safe_rows),
        "total_count": total_count,
        "rows_shown": len(safe_rows),
        "summary": summary,
        "flags": flags,
        "warning_message": (
            f"{dropped_rows} transactions were hidden to protect your privacy."
            if dropped_rows
            else None
        ),
    }
    return render(request, "customer/home.html", context)


@require_POST
def flag_transaction(request: HttpRequest) -> HttpResponse:
    row_id = str(request.POST.get("row_id", "")).strip().lower()
    if not _is_valid_row_id(row_id):
        return redirect("customer:home")

    reason = str(request.POST.get("reason", "")).strip()
    if len(reason) > MAX_REASON_LENGTH:
        reason = reason[:MAX_REASON_LENGTH]

    scored_run = load_scored_run(request.session)
    if not scored_run or not scored_run.rows:
        return redirect("customer:home")

    rows_with_ids = _ensure_row_ids(list(scored_run.rows))
    row_lookup = {str(r.get("row_id", "")).lower(): r for r in rows_with_ids}
    base = row_lookup.get(row_id)
    if base is None:
        return redirect("customer:home")

    flags = _get_customer_flags(request.session)
    # Overwrite allowed so the latest customer note is captured.
    flags[row_id] = {
        "row_id": row_id,
        "timestamp": str(base.get("timestamp", "")),
        "customer_id": str(base.get("customer_id", "")),
        "amount": str(base.get("amount", "")),
        "merchant": str(base.get("merchant", "")),
        "description": str(base.get("description", "")),
        "reason": reason,
        "flagged_at": timezone.now().isoformat(),
    }
    request.session[CUSTOMER_FLAGS_SESSION_KEY] = flags
    request.session.modified = True
    return redirect("customer:home")


def export_flags(request: HttpRequest) -> HttpResponse:
    flags = _get_customer_flags(request.session)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "row_id",
            "timestamp",
            "customer_id",
            "amount",
            "merchant",
            "description",
            "reason",
            "flagged_at",
        ]
    )

    for row_id in sorted(flags.keys()):
        payload = flags.get(row_id) or {}
        writer.writerow(
            [
                payload.get("row_id", row_id),
                payload.get("timestamp", ""),
                payload.get("customer_id", ""),
                payload.get("amount", ""),
                payload.get("merchant", ""),
                payload.get("description", ""),
                payload.get("reason", ""),
                payload.get("flagged_at", ""),
            ]
        )

    resp = HttpResponse(buf.getvalue(), content_type="text/csv")
    resp["Content-Disposition"] = "attachment; filename=customer_flags.csv"
    return resp
