"""Unit tests for customer spend summary aggregation."""
from __future__ import annotations

from datetime import datetime, timezone

from customer_site.services import build_spend_summary


def test_build_spend_summary_handles_mtd_and_categories() -> None:
    rows = [
        {
            "timestamp": "2024-01-05T10:00:00Z",
            "amount": "10.50",
            "category": "Groceries",
        },
        {
            "timestamp": "2024-01-12T10:00:00+00:00",
            "amount": "20",
            "category": "Groceries",
        },
        {
            "timestamp": "2024-02-01T10:00:00Z",
            "amount": "15",
            "category": "Travel",
        },
        {
            "timestamp": "bad-timestamp",
            "amount": "5",
            "category": "Dining",
        },
        {
            "timestamp": "2024-01-03T10:00:00Z",
            "amount": None,
            "category": "Dining",
        },
        {
            "timestamp": "2024-01-03T10:00:00Z",
            "amount": "9",
            "category": "",
        },
    ]

    now = datetime(2024, 1, 20, tzinfo=timezone.utc)
    summary = build_spend_summary(rows, now=now, max_categories=3)

    assert summary["mtd_total"] == "30.50"
    assert summary["total_all"] == "50.50"
    assert [c["name"] for c in summary["categories"]] == [
        "Groceries",
        "Travel",
        "Dining",
    ]
    assert [c["total"] for c in summary["categories"]] == [
        "30.50",
        "15.00",
        "5.00",
    ]
