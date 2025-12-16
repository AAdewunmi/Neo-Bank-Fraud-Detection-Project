"""
Training utilities for reproducibility:
- stable schema hashing
- model registry I/O

These are intentionally small. Week 1 goal is reliability, not a framework.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable


def schema_hash(columns: Iterable[str]) -> str:
    """
    Create a stable schema hash from an ordered list of column names.

    Args:
        columns: Ordered column names (order matters).

    Returns:
        Short SHA-256 prefix to store in registry and tests.
    """
    joined = "|".join(columns)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]
