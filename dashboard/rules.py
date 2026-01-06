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
from typing import Iterable

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


def apply_rules(scored_df: pd.DataFrame, rules: Iterable[dict[str, str]]) -> pd.DataFrame:
    """
    Apply substring rules on merchant and description to override category.

    Args:
        scored_df: DataFrame containing at least merchant, description, category.
        rules: Iterable of rule dicts.

    Returns:
        DataFrame with category and category_source updated.
    """
    out = scored_df.copy()

    if "category_source" not in out.columns:
        out["category_source"] = "model"

    # Ensure columns exist to avoid KeyError on small inputs.
    out["merchant"] = out.get("merchant", "").astype(str)
    out["description"] = out.get("description", "").astype(str)

    for rule in rules:
        needle = str(rule.get("contains", "")).strip().lower()
        cat = str(rule.get("category", "")).strip()
        if not needle or not cat:
            continue

        mask = (
            out["merchant"].str.lower().str.contains(needle, na=False)
            | out["description"].str.lower().str.contains(needle, na=False)
        )

        # Override and mark as rule-driven.
        out.loc[mask, "category"] = cat
        out.loc[mask, "category_source"] = "rule"

    return out
