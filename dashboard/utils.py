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

