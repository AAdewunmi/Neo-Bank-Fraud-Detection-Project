"""Customer-facing views (read-only surface)."""

from __future__ import annotations

import csv
import io
import logging
import os
import re
from typing import Any, Dict, List, Mapping

from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from dashboard.views import _get_category_edits, _overlay_category_edits
from customer_site.models import CustomerTransaction
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
    display_rows = _overlay_category_edits(list(rows), edits)

    safe_rows: List[Dict[str, Any]] = []
    for row in display_rows[:max_rows]:
        safe_rows.append({field: row.get(field, "") for field in SAFE_FIELDS})
    return safe_rows


def _load_latest_transactions(user, max_rows: int) -> tuple[List[Dict[str, Any]], int]:
    if user.is_staff:
        base_qs = CustomerTransaction.objects.all()
    else:
        base_qs = CustomerTransaction.objects.filter(customer_id=user.username)

    latest_scored_at = (
        base_qs.order_by("-scored_at").values_list("scored_at", flat=True).first()
    )
    if not latest_scored_at:
        return [], 0

    scoped = base_qs.filter(scored_at=latest_scored_at)
    total_count = scoped.count()
    rows = list(
        scoped.order_by("-timestamp").values(
            "row_id",
            "timestamp",
            "merchant",
            "description",
            "amount",
            "category",
        )[:max_rows]
    )
    return rows, total_count


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


def customer_login(request: HttpRequest) -> HttpResponse:
    if request.user.is_authenticated:
        return redirect("customer:dashboard")

    form = AuthenticationForm(request, data=request.POST or None)
    form.fields["username"].widget.attrs.update({"class": "form-control"})
    form.fields["password"].widget.attrs.update({"class": "form-control"})
    if request.method == "POST" and form.is_valid():
        user = form.get_user()
        if user.is_staff:
            form.add_error(
                None,
                "Staff accounts cannot sign in here. Use Ops sign in instead.",
            )
        else:
            login(request, user)
            return redirect("customer:dashboard")

    return render(
        request,
        "customer/login.html",
        {"form": form, "brand_link": True},
    )


def customer_logout(request: HttpRequest) -> HttpResponse:
    logout(request)
    return redirect("public_home")


@login_required(login_url="customer:login")
def dashboard(request: HttpRequest) -> HttpResponse:
    """Render the customer dashboard."""
    flags = _get_customer_flags(request.session)

    max_rows = int(os.environ.get("LEDGERGUARD_CUSTOMER_MAX_ROWS", "200"))
    edits = _get_category_edits(request.session)
    base_rows, total_count = _load_latest_transactions(request.user, max_rows)
    if not base_rows:
        context = {
            "rows": [],
            "has_rows": False,
            "total_count": 0,
            "rows_shown": 0,
            "flags": flags,
            "warning_message": None,
        }
        return render(request, "customer/dashboard.html", context)

    safe_rows = _build_customer_rows(base_rows, edits, max_rows)
    safe_rows, dropped_rows = _filter_customer_rows(safe_rows)
    summary = build_spend_summary(safe_rows, max_categories=6)

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
    return render(request, "customer/dashboard.html", context)


@require_POST
@login_required(login_url="customer:login")
def flag_transaction(request: HttpRequest) -> HttpResponse:
    row_id = str(request.POST.get("row_id", "")).strip().lower()
    if not _is_valid_row_id(row_id):
        return redirect("customer:dashboard")

    reason = str(request.POST.get("reason", "")).strip()
    if len(reason) > MAX_REASON_LENGTH:
        reason = reason[:MAX_REASON_LENGTH]

    if request.user.is_staff:
        base = CustomerTransaction.objects.filter(row_id=row_id).values(
            "row_id",
            "timestamp",
            "customer_id",
            "amount",
            "merchant",
            "description",
        ).first()
    else:
        base = CustomerTransaction.objects.filter(
            row_id=row_id,
            customer_id=request.user.username,
        ).values(
            "row_id",
            "timestamp",
            "customer_id",
            "amount",
            "merchant",
            "description",
        ).first()
    if base is None:
        return redirect("customer:dashboard")

    flags = _get_customer_flags(request.session)
    # Overwrite allowed so the latest customer note is captured.
    flags[row_id] = {
        "row_id": str(base.get("row_id", row_id)),
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
    return redirect("customer:dashboard")


@login_required(login_url="customer:login")
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
