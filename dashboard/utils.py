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



