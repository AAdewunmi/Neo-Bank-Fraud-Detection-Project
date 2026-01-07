"""Helpers for customer-facing summaries."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


def _parse_amount(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "").replace("$", "")
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _format_money(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def build_spend_summary(
    rows: Iterable[Mapping[str, Any]],
    *,
    now: Optional[datetime] = None,
    max_categories: int = 6,
) -> Dict[str, Any]:
    now_dt = now or datetime.now(timezone.utc)
    category_totals: Dict[str, Decimal] = {}
    mtd_total = Decimal("0")

    for row in rows:
        category = str(row.get("category", "")).strip()
        amount = _parse_amount(row.get("amount"))
        if not category or amount is None:
            continue

        category_totals[category] = category_totals.get(category, Decimal("0")) + amount

        timestamp = _parse_timestamp(row.get("timestamp"))
        if timestamp and timestamp.year == now_dt.year and timestamp.month == now_dt.month:
            mtd_total += amount

    sorted_categories = sorted(
        category_totals.items(),
        key=lambda item: (-item[1], item[0].lower()),
    )[: max(0, max_categories)]

    categories = [
        {"name": name, "total": _format_money(total)} for name, total in sorted_categories
    ]

    total_all = sum(category_totals.values(), Decimal("0"))

    return {
        "categories": categories,
        "mtd_total": _format_money(mtd_total),
        "total_all": _format_money(total_all),
        "has_data": bool(categories),
    }


def persist_scored_transactions(
    rows: Sequence[Mapping[str, Any]],
    *,
    scored_at: datetime,
) -> None:
    from dashboard.utils import compute_row_id
    from customer_site.models import CustomerTransaction

    records = []
    for row in rows:
        row_id = str(row.get("row_id") or compute_row_id(row))
        flagged = bool(row.get("flagged") or row.get("fraud_flag"))
        fraud_risk = row.get("fraud_risk")
        records.append(
            CustomerTransaction(
                row_id=row_id,
                customer_id=str(row.get("customer_id", "")),
                timestamp=str(row.get("timestamp", "")),
                amount=str(row.get("amount", "")),
                merchant=str(row.get("merchant", "")),
                description=str(row.get("description", "")),
                category=str(row.get("category", "")),
                predicted_category=str(
                    row.get("predicted_category", row.get("category", ""))
                ),
                category_source=str(row.get("category_source", "model")),
                fraud_risk=float(fraud_risk) if fraud_risk is not None else None,
                flagged=flagged,
                scored_at=scored_at,
            )
        )

    if not records:
        return

    CustomerTransaction.objects.bulk_create(
        records,
        update_conflicts=True,
        unique_fields=["row_id"],
        update_fields=[
            "customer_id",
            "timestamp",
            "amount",
            "merchant",
            "description",
            "category",
            "predicted_category",
            "category_source",
            "fraud_risk",
            "flagged",
            "scored_at",
        ],
    )
