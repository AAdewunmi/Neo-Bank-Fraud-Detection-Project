"""
Targeted unit tests to cover rules + utils helpers.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dashboard import rules
from dashboard.forms import EditCategoryForm
from dashboard.utils import _canonicalise_amount, compute_row_id, extract_identity


def test_load_rules_missing_file_returns_empty(tmp_path):
    missing = tmp_path / "nope.json"
    assert rules.load_rules(str(missing)) == []


def test_apply_rules_overrides_and_sets_source():
    df = pd.DataFrame(
        [
            {"merchant": "Coffee Co", "description": "flat white", "category": "Other"},
            {"merchant": "Grocery", "description": "weekly shop", "category": "Other"},
        ]
    )
    applied = rules.apply_rules(df, [{"contains": "coffee", "category": "Food"}])
    assert applied.loc[0, "category"] == "Food"
    assert applied.loc[0, "category_source"] == "rule"
    assert applied.loc[1, "category"] == "Other"
    assert applied.loc[1, "category_source"] == "model"


def test_apply_rules_skips_empty_rule_fields():
    df = pd.DataFrame([{"merchant": "Coffee Co", "description": "", "category": "Other"}])
    applied = rules.apply_rules(df, [{"contains": "", "category": "Food"}])
    assert applied.loc[0, "category"] == "Other"
    assert applied.loc[0, "category_source"] == "model"


def test_edit_category_form_rejects_blank():
    form = EditCategoryForm(data={"row_id": "abc", "new_category": "  "})
    assert not form.is_valid()
    assert "new_category" in form.errors


def test_canonicalise_amount_falls_back_on_invalid():
    assert _canonicalise_amount("not-a-number") == "not-a-number"


def test_extract_identity_builds_row_id():
    row = {
        "timestamp": "2024-01-01T00:00:00Z",
        "customer_id": "c1",
        "amount": "10.0",
        "merchant": "Coffee Co",
        "description": "flat white",
    }
    ident = extract_identity(row)
    assert ident.row_id == compute_row_id(row)
    assert ident.amount == "10"
