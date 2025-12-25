# tests/test_session_store_core.py
"""
Covers dashboard.session_store core logic (save/load/clear paths)
to raise overall coverage without touching implementation details.
"""
import pytest
import pandas as pd
from dashboard import session_store


@pytest.mark.django_db
def test_save_and_load_scored_run(client):
    """Covers save_scored_run and load_scored_run with valid session mapping."""
    df = pd.DataFrame([{"x": 1, "y": 2}])
    diags = {"n": 1, "pct_flagged": 0.0, "threshold": 0.7}

    session = {}  # direct mapping
    # save_scored_run requires rows and diags separately
    session_store.save_scored_run(session, df.to_dict(orient="records"), diags)
    assert "scored_rows" in session or "scored_run" in session

    loaded = session_store.load_scored_run(session)
    assert loaded is not None
    assert "rows" in loaded
    assert "diags" in loaded
    assert loaded["diags"]["threshold"] == 0.7


def test_clear_scored_run_removes_keys():
    """Covers clear_scored_run() instead of missing reset_scored_run()."""
    session = {"scored_rows": [{"dummy": True}], "scored_diags": {"ok": True}}
    session_store.clear_scored_run(session)
    assert "scored_rows" not in session
    assert "scored_diags" not in session


def test_load_scored_run_handles_missing_session_key():
    """When no scored data exists, load_scored_run() should return None."""
    session = {}
    result = session_store.load_scored_run(session)
    assert result is None
