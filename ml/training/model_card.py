"""
Write a small model-card JSON with dataset hash, metrics, and provenance.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Any


def file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_model_card(artefact_path: str, dataset_path: str, metrics: Dict[str, Any]) -> str:
    card = {
        "artefact": artefact_path,
        "dataset_sha1": file_sha1(dataset_path),
        "metrics": metrics
    }
    out = Path(artefact_path).with_suffix(".card.json")
    out.write_text(json.dumps(card, indent=2))
    return str(out)
