"""
Tests for model card emission.

Purpose
- Assert that training emits a model card JSON artefact.
- Keep the test CI-safe and deterministic.
- Avoid depending on wall-clock timestamps or exact artefact naming.

Design notes
- Uses a temporary working directory so artefacts do not leak between tests.
- Invokes the trainer as a subprocess to match real execution.
- Asserts structural correctness of the model card rather than filename patterns.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_model_card_written_for_categorisation(tmp_path, monkeypatch):
    """
    Train the categorisation embeddings model and verify that a model card is written.

    Guarantees checked
    - At least one *.card.json file is created.
    - The card contains required top-level keys.
    - Dataset hash and metrics are present.
    """

    # Run the trainer in an isolated working directory.
    monkeypatch.chdir(tmp_path)

    artefacts_dir = tmp_path / "artefacts"
    artefacts_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    input_path = Path(__file__).resolve().parents[1] / "data" / "sample_transactions.csv"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "ml.training.train_categorisation_embeddings",
            "--input",
            str(input_path),
            "--target_col",
            "category",
            "--text_cols",
            "merchant",
            "description",
            "--use_embeddings",
            "no",
            "--safe_threads",
            "yes",
            "--registry",
            "model_registry.json",
        ],
        env=env,
    )

    cards = list(artefacts_dir.glob("*.card.json"))
    assert cards, "No model card JSON was created"

    # Validate structure of the card
    card = json.loads(cards[0].read_text(encoding="utf-8"))

    assert "artefact" in card
    assert "dataset_sha1" in card
    assert "metrics" in card

    # Categorisation-specific metric
    assert "macro_f1" in card["metrics"]
    assert isinstance(card["metrics"]["macro_f1"], float)
