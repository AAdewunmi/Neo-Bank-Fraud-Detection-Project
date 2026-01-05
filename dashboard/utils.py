"""
dashboard/utils.py

Utility helpers for the dashboard layer.

This module provides stable row identifiers for transaction rows so that
analyst feedback can be exported and matched reliably later, even if the
table order changes.

Design notes:
- We avoid using row_index as an identifier because sorting/filtering/re-upload
  can change index positions.
- We compute a deterministic row_id from a canonical subset of fields.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping


ROW_ID_FIELDS: tuple[str, ...] = (
    "timestamp",
    "customer_id",
    "amount",
    "merchant",
    "description",
)


def _canonicalise_text(value: Any) -> str:
    """
    Convert a value into a stable text representation.

    This normalises whitespace and guards against None.
    """
    if value is None:
        return ""
    return str(value).strip()


def _canonicalise_amount(value: Any) -> str:
    """
    Canonicalise amount to a stable decimal string where possible.

    CSV inputs sometimes vary between "10", "10.0", and 10.0.
    Canonicalising reduces accidental row_id drift.
    """
    if value is None:
        return ""
    try:
        d = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return _canonicalise_text(value)

    # Normalize removes exponent and trailing zeros in most cases.
    # Example: Decimal("10.0") -> "10"
    return format(d.normalize(), "f")


def compute_row_id(row: Mapping[str, Any]) -> str:
    """
    Compute a stable row_id from a transaction row.

    Args:
        row: A mapping containing at least the ROW_ID_FIELDS keys.

    Returns:
        Hex string SHA-256 hash representing the stable row identifier.
    """
    parts: list[str] = []
    for key in ROW_ID_FIELDS:
        if key == "amount":
            parts.append(_canonicalise_amount(row.get(key)))
        else:
            parts.append(_canonicalise_text(row.get(key)))

    canonical = "|".join(parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RowIdentity:
    """
    Stable identifiers for a row, suitable for export as feedback.

    This is the minimum we want to carry forward for retraining later.
    """
    row_id: str
    timestamp: str
    customer_id: str
    amount: str
    merchant: str
    description: str


def extract_identity(row: Mapping[str, Any]) -> RowIdentity:
    """
    Extract RowIdentity from a row dict.

    The row_id is computed from the canonical fields.
    """
    base = {
        "timestamp": _canonicalise_text(row.get("timestamp")),
        "customer_id": _canonicalise_text(row.get("customer_id")),
        "amount": _canonicalise_amount(row.get("amount")),
        "merchant": _canonicalise_text(row.get("merchant")),
        "description": _canonicalise_text(row.get("description")),
    }
    rid = compute_row_id(base)
    return RowIdentity(row_id=rid, **base)
