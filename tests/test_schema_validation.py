"""
Schema validation tests for Week 1 ingestion.
"""
from __future__ import annotations

import io

import pytest

from dashboard.services import read_csv


def test_read_csv_missing_columns_raises() -> None:
    """
    Missing required columns should fail fast with a clear error.
    """
    bad = b"foo,bar\n1,2\n"
    with pytest.raises(ValueError) as exc:
        read_csv(io.BytesIO(bad))
    assert "Missing required columns" in str(exc.value)


def test_read_csv_amount_coercion_invalid_to_zero() -> None:
    """
    Invalid numeric values must coerce to 0.0.
    """
    csv = (
        "timestamp,amount,customer_id,merchant,description\n"
        "2024-01-01T00:00:00Z,notanumber,c1,m1,d1\n"
    )
    df = read_csv(io.BytesIO(csv.encode("utf-8")))
    assert df.loc[0, "amount"] == 0.0


def test_read_csv_empty_customer_id_rejected() -> None:
    """
    customer_id is required and must not be empty.
    """
    csv = (
        "timestamp,amount,customer_id,merchant,description\n"
        "2024-01-01T00:00:00Z,10.0,,m1,d1\n"
    )
    with pytest.raises(ValueError) as exc:
        read_csv(io.BytesIO(csv.encode("utf-8")))
    assert "customer_id must be non-empty" in str(exc.value)


def test_read_csv_rejects_large_files(monkeypatch) -> None:
    monkeypatch.setenv("LEDGERGUARD_DASHBOARD_MAX_SCORE_ROWS", "2")
    csv = (
        "timestamp,amount,customer_id,merchant,description\n"
        "2024-01-01T00:00:00Z,1.0,c1,m1,d1\n"
        "2024-01-01T00:01:00Z,2.0,c2,m2,d2\n"
        "2024-01-01T00:02:00Z,3.0,c3,m3,d3\n"
    )
    with pytest.raises(ValueError) as exc:
        read_csv(io.BytesIO(csv.encode("utf-8")))
    assert "Dashboard scoring supports up to" in str(exc.value)
