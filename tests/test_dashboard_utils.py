"""
Tests for dashboard utils and form helpers.
"""

from __future__ import annotations

from dataclasses import asdict

from dashboard import utils
from dashboard.forms import EditCategoryForm


def test_compute_row_id_is_stable_across_amount_formats() -> None:
    row_a = {
        "timestamp": "2024-01-01T12:00:00Z",
        "customer_id": "c1",
        "amount": "10.0",
        "merchant": " Coffee Co ",
        "description": "flat white",
    }
    row_b = {
        "timestamp": "2024-01-01T12:00:00Z",
        "customer_id": "c1",
        "amount": 10,
        "merchant": "Coffee Co",
        "description": "flat white",
    }

    assert utils.compute_row_id(row_a) == utils.compute_row_id(row_b)


def test_extract_identity_canonicalises_fields() -> None:
    row = {
        "timestamp": " 2024-02-02T09:30:00Z ",
        "customer_id": " c9 ",
        "amount": "10.00",
        "merchant": "Shop",
        "description": None,
    }

    identity = utils.extract_identity(row)

    assert identity.timestamp == "2024-02-02T09:30:00Z"
    assert identity.customer_id == "c9"
    assert identity.amount == "10"
    assert identity.description == ""
    assert identity.row_id == utils.compute_row_id(asdict(identity))


def test_compute_row_id_uses_text_for_non_decimal_amounts() -> None:
    row_a = {
        "timestamp": "t1",
        "customer_id": "c1",
        "amount": " ten ",
        "merchant": "m1",
        "description": "d1",
    }
    row_b = {
        "timestamp": "t1",
        "customer_id": "c1",
        "amount": "ten",
        "merchant": "m1",
        "description": "d1",
    }

    assert utils.compute_row_id(row_a) == utils.compute_row_id(row_b)


def test_edit_category_form_strips_whitespace() -> None:
    form = EditCategoryForm(data={"row_id": "a" * 64, "new_category": "  Food  "})
    assert form.is_valid()
    assert form.cleaned_data["new_category"] == "Food"


def test_edit_category_form_rejects_blank() -> None:
    form = EditCategoryForm(data={"row_id": "a" * 64, "new_category": "   "})
    assert not form.is_valid()
    assert "new_category" in form.errors
