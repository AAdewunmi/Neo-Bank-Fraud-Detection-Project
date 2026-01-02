from __future__ import annotations

import io

import pandas as pd

from dashboard.services import read_csv


def test_read_csv_accepts_paysim_schema() -> None:
    csv = (
        "step,amount,nameOrig,nameDest,type,isFraud\n"
        "1,10.5,cust_1,merchant_1,PAYMENT,0\n"
    )
    df = read_csv(io.BytesIO(csv.encode("utf-8")))

    for col in ["timestamp", "amount", "customer_id", "merchant", "description"]:
        assert col in df.columns

    assert df.loc[0, "customer_id"] == "cust_1"
    assert df.loc[0, "merchant"] == "merchant_1"
    assert df.loc[0, "description"] == "PAYMENT"
    assert pd.notna(df.loc[0, "timestamp"])
