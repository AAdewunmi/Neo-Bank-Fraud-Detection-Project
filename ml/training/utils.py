# ml/training/utils.py
"""
Training utilities used across Week 1–3 labs.

Includes:
- schema_hash: stable hashing of input column lists and feature lists
- registry helpers: load_registry/save_registry for model_registry.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def schema_hash(columns: Iterable[str]) -> str:
    """
    Compute a stable hash for the model input schema.

    The caller should pass the exact ordered list of columns/features used by the trainer,
    for example:
      - list(text_cols) + [target_col]
      - feature_names (fraud engineered feature list)

    Normalisation:
      - trims whitespace
      - lowercases

    Returns:
      A SHA256 hex digest.
    """
    normalised = [str(c).strip().lower() for c in columns]
    payload = json.dumps(
        {"columns": normalised},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_registry(path: str | Path) -> dict[str, Any]:
    """
    Load model_registry.json. If it does not exist, create a minimal structure.
    """
    p = Path(path)
    if not p.exists():
        return {"categorisation": {}, "fraud": {}}

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    data.setdefault("categorisation", {})
    data.setdefault("fraud", {})
    return data


def save_registry(registry: dict[str, Any], path: str | Path) -> None:
    """
    Save model_registry.json with deterministic formatting.
    """
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, sort_keys=True)
        f.write("\n")
