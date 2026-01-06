"""
dashboard/rules.py

Rules overlay for categorisation.

Simple, transparent logic:
- Each rule is a case-insensitive substring match on merchant or description.
- If a rule matches, it overrides category and sets category_source='rule'.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def load_rules(path: str = "rules/category_overrides.json") -> list[dict[str, str]]:
    """
    Load rules from a JSON file.

    Returns:
        List of dicts with keys: contains, category.
    """
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))
