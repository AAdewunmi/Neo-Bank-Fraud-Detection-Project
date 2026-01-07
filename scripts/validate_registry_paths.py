"""Validate model_registry.json artefact paths exist."""

from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REGISTRY_PATH = BASE_DIR / "model_registry.json"

PATH_FIELDS = (
    "artefact",
    "metrics_path",
    "model_card",
    "card_json",
    "dataset_path",
    "feature_state_path",
)


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (BASE_DIR / path)


def main() -> int:
    if not REGISTRY_PATH.exists():
        print(f"Registry not found: {REGISTRY_PATH}")
        return 1

    registry = json.loads(REGISTRY_PATH.read_text())
    missing = []

    for section in ("fraud", "categorisation", "fraud_synthetic"):
        entries = registry.get(section, {})
        for key, value in entries.items():
            if key == "latest" or not isinstance(value, dict):
                continue
            for field in PATH_FIELDS:
                if field not in value:
                    continue
                candidate = str(value.get(field, "")).strip()
                if not candidate:
                    continue
                resolved = _resolve(candidate)
                if not resolved.exists():
                    missing.append((section, key, field, candidate))

    if missing:
        print("Missing registry paths:")
        for section, key, field, candidate in missing:
            print(f"- [{section}] {key} {field}: {candidate}")
        return 1

    print("All registry paths exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
