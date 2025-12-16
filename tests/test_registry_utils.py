"""
Tests for model registry utilities.

Intent:
- Cover the "missing file" branch in load_registry
- Cover save_registry -> load_registry round-trip
"""
from __future__ import annotations

from ml.training.utils import load_registry, save_registry


def test_load_registry_returns_default_when_missing(tmp_path) -> None:
    """
    If the registry file does not exist, load_registry should return the default structure.
    """
    missing = tmp_path / "does_not_exist.json"
    reg = load_registry(missing)
    assert reg == {"categorisation": {}, "fraud": {}}


def test_save_and_load_registry_round_trip(tmp_path) -> None:
    """
    Registry should save deterministically and load back with the same content.
    """
    path = tmp_path / "model_registry.json"
    original = {"categorisation": {"latest": "v1"}, "fraud": {"latest": "v2"}}

    save_registry(original, path)
    loaded = load_registry(path)

    assert loaded == original
    assert path.read_text(encoding="utf-8").strip().startswith("{")
