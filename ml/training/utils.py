"""
Training utilities for reproducibility:
- stable schema hashing
- model registry I/O

These are intentionally small. Week 1 goal is reliability, not a framework.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable


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


def load_registry(path: str | Path) -> Dict[str, Any]:
    """
    Load the model registry JSON, returning a default structure if missing.

    Args:
        path: Path to model_registry.json.

    Returns:
        Registry dictionary.
    """
    p = Path(path)
    if not p.exists():
        return {"categorisation": {}, "fraud": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def save_registry(registry: Dict[str, Any], path: str | Path) -> None:
    """
    Save registry JSON deterministically (sorted keys, stable formatting).

    Args:
        registry: Registry dictionary to persist.
        path: Destination path.
    """
    Path(path).write_text(
        json.dumps(registry, indent=2, sort_keys=True),
        encoding="utf-8",
    )
