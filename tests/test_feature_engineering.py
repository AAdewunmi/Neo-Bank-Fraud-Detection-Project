"""
Feature engineering coverage for dashboard.services.read_csv.
"""
from __future__ import annotations

import io

import pandas as pd

from dashboard.services import read_csv


def test_read_csv_engineers_fraud_features() -> None:
    csv = (
        "timestamp,amount,customer_id,merchant,description\n"
        "2024-01-06T10:00:00Z,5.00,c1,m1,d1\n"
        "2024-01-06T11:00:00Z,50.00,c1,m1,d2\n"
        "2024-01-06T12:00:00Z,500.00,c2,m2,d3\n"
        "2024-01-06T13:00:00Z,1500.00,c3,m3,d4\n"
    )
    df = read_csv(io.BytesIO(csv.encode("utf-8")))

    assert set(
        ["hour", "is_weekend", "amount_bucket", "velocity_24h", "is_international"]
    ).issubset(df.columns)

    assert df.loc[0, "hour"] == 10
    assert df.loc[0, "is_weekend"] == 1

    assert df.loc[0, "amount_bucket"] == 0
    assert df.loc[1, "amount_bucket"] == 1
    assert df.loc[2, "amount_bucket"] == 2
    assert df.loc[3, "amount_bucket"] == 3

    assert df.loc[0, "velocity_24h"] == 1
    assert df.loc[1, "velocity_24h"] == 2

    assert df["is_international"].eq(0).all()
