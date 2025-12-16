"""
Trainer integration tests.

These tests intentionally exercise the trainer main paths so coverage reflects
real shipped behavior (artefact creation + registry updates).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ml.training.train_categorisation import main as cat_main
from ml.training.train_fraud_baseline import main as fraud_main


def _write_categorisation_csv(path: Path) -> None:
    """
    Write a small dataset with enough samples per class to support
    stratification.

    We use 5 classes with 2 examples each (10 rows total) to force the trainer
    to adjust test_size above 0.25 to satisfy stratified split constraints.
    """
    df = pd.DataFrame(
        [
            (
                "2024-01-01T10:00:00Z",
                3.50,
                "c001",
                "Coffee Co",
                "flat white",
                "Food & Drink",
            ),
            (
                "2024-01-01T11:00:00Z",
                4.10,
                "c002",
                "Coffee Co",
                "iced latte",
                "Food & Drink",
            ),
            (
                "2024-01-01T12:00:00Z",
                52.00,
                "c003",
                "Grocer Ltd",
                "weekly groceries",
                "Groceries",
            ),
            (
                "2024-01-01T13:00:00Z",
                34.75,
                "c004",
                "Grocer Ltd",
                "supermarket groceries",
                "Groceries",
            ),
            (
                "2024-01-01T14:00:00Z",
                12.99,
                "c005",
                "Streamr Inc",
                "monthly subscription",
                "Subscriptions",
            ),
            (
                "2024-01-01T15:00:00Z",
                8.99,
                "c006",
                "Streamr Inc",
                "video streaming plan",
                "Subscriptions",
            ),
            (
                "2024-01-01T16:00:00Z",
                499.00,
                "c007",
                "Weird Merchant",
                "high-value item",
                "Other",
            ),
            (
                "2024-01-01T17:00:00Z",
                250.00,
                "c008",
                "Weird Merchant",
                "online purchase",
                "Other",
            ),
            (
                "2024-01-01T18:00:00Z",
                18.20,
                "c009",
                "Transit",
                "city travel",
                "Transport",
            ),
            (
                "2024-01-01T19:00:00Z",
                2.40,
                "c010",
                "Transit",
                "bus fare",
                "Transport",
            ),
        ],
        columns=[
            "timestamp",
            "amount",
            "customer_id",
            "merchant",
            "description",
            "category",
        ],
    )
    df.to_csv(path, index=False)


def _read_registry(path: Path) -> dict:
    """
    Read registry JSON.

    Args:
        path: Path to model_registry.json

    Returns:
        Loaded JSON as dict.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def test_categorisation_trainer_writes_artefact_and_registry(
    tmp_path, monkeypatch
) -> None:
    """
    Categorisation trainer should:
    - create artefacts/ and write a .joblib model
    - update model_registry.json with a 'latest' pointer
    """
    monkeypatch.chdir(tmp_path)

    data_path = tmp_path / "sample.csv"
    registry_path = tmp_path / "model_registry.json"

    _write_categorisation_csv(data_path)

    args = SimpleNamespace(
        input=str(data_path),
        target_col="category",
        text_cols=["merchant", "description"],
        registry=str(registry_path),
        test_size=0.25,
    )

    cat_main(args)

    reg = _read_registry(registry_path)
    assert "categorisation" in reg
    assert "latest" in reg["categorisation"]

    latest = reg["categorisation"]["latest"]
    assert latest.startswith("cat_lr_")
    assert latest in reg["categorisation"]

    artefact_path = Path(reg["categorisation"][latest]["artefact"])
    assert artefact_path.exists()
    assert artefact_path.suffix == ".joblib"


def test_fraud_trainer_writes_artefact_and_registry(
    tmp_path, monkeypatch
) -> None:
    """
    Fraud trainer should:
    - create artefacts/ and write a .joblib model
    - update model_registry.json with a 'latest' pointer
    """
    monkeypatch.chdir(tmp_path)

    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"],
            "amount": [10.0, 1000.0],
            "customer_id": ["c1", "c2"],
            "merchant": ["m1", "m2"],
            "description": ["d1", "d2"],
        }
    )
    data_path = tmp_path / "fraud.csv"
    registry_path = tmp_path / "model_registry.json"
    df.to_csv(data_path, index=False)

    args = SimpleNamespace(
        input=str(data_path),
        amount_col="amount",
        registry=str(registry_path),
    )

    fraud_main(args)

    reg = _read_registry(registry_path)
    assert "fraud" in reg
    assert "latest" in reg["fraud"]

    latest = reg["fraud"]["latest"]
    assert latest.startswith("fraud_if_")
    assert latest in reg["fraud"]

    artefact_path = Path(reg["fraud"][latest]["artefact"])
    assert artefact_path.exists()
    assert artefact_path.suffix == ".joblib"
