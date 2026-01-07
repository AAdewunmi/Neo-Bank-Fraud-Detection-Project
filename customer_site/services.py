"""Helpers for customer-facing summaries."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Dict, Iterable, Mapping, Optional


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
