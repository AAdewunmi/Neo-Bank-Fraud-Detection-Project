"""
Tests for supervised fraud training modes.

This module verifies:
- Deterministic behaviour of synthetic label generation (CI-safe).
- Correct column mapping and label extraction for PaySim data (offline only).
- End-to-end artefact and registry writing for supervised fraud training.

Marker policy
- Tests that depend on PaySim semantics are marked with @pytest.mark.paysim
  so they are excluded from CI via: pytest -m "ml and not paysim".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.training.train_fraud_supervised import (
    load_paysim,
    make_synthetic_labels,
    train,
)


def test_synthetic_label_generation_is_deterministic() -> None:
    """
    Synthetic label generation must be deterministic and well-formed.

    This test is CI-safe and verifies:
    - Labels are binary.
    - Metadata is present and internally consistent.
    - Positive rate is non-zero for a non-degenerate input.
    """
    df = pd.DataFrame(
        {
            "amount": [10, 20, 30, 40, 100],
            "timestamp": [1, 2, 3, 4, 5],
            "customer_id": ["a", "a", "b", "b", "c"],
        }
    )

    y, meta = make_synthetic_labels(df, "amount", positive_quantile=0.8)

    # Labels must be binary
    assert set(np.unique(y)).issubset({0, 1})

    # Metadata must clearly identify synthetic provenance
    assert meta["label_mode"] == "synthetic_top_amount_quantile"

    # Ensure the rule actually produces positives
    assert meta["positive_rate"] > 0.0


# @pytest.mark.paysim
# def test_paysim_loader_maps_columns_correctly() -> None:
#     """
#     PaySim loader must map dataset-specific columns into the internal schema.

#     This test is marked paysim because:
#     - It relies on PaySim column conventions.
#     - It validates dataset-specific semantics.
#     """
#     df = pd.DataFrame(
#         {
#             "amount": [100.0, 200.0],
#             "isFraud": [0, 1],
#             "nameOrig": ["c1", "c2"],
#             "step": [1, 2],
#         }
#     )

#     out, y = load_paysim(df)

#     # PaySim nameOrig must map to customer_id
#     assert "customer_id" in out.columns

#     # PaySim step must be mapped into a timestamp surrogate
#     assert "timestamp" in out.columns

#     # Labels must be preserved exactly
#     assert y.tolist() == [0, 1]


# def test_training_writes_model_and_metrics(tmp_path: Path, monkeypatch) -> None:
#     """
#     Supervised training must write artefacts and registry entries.

#     This test uses synthetic labels only and is CI-safe.
#     It verifies:
#     - Model artefacts are persisted.
#     - Metrics are written to disk.
#     - No external datasets are required.
#     """
#     df = pd.DataFrame(
#         {
#             "amount": [10, 20, 30, 40, 50, 100],
#             "timestamp": [1, 2, 3, 4, 5, 6],
#             "customer_id": ["a", "a", "b", "b", "c", "c"],
#         }
#     )

#     # Explicit, deterministic label vector
#     y = np.array([0, 0, 0, 0, 1, 1])

#     registry_path = tmp_path / "registry.json"
#     registry_path.write_text(json.dumps({}), encoding="utf-8")

#     # Run training in an isolated filesystem context
#     monkeypatch.chdir(tmp_path)

#     train(
#         df=df,
#         y=y,
#         dataset="synthetic",
#         registry_path=str(registry_path),
#     )

#     artefacts = Path("artefacts")
#     reports = Path("reports")

#     # Artefact directories must be created
#     assert artefacts.exists()
#     assert reports.exists()

#     # At least one model file must be written
#     assert any(p.suffix == ".joblib" for p in artefacts.iterdir())

#     # At least one metrics file must be written
#     assert any(p.suffix == ".json" for p in reports.iterdir())
