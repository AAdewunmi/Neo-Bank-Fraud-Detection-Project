"""
Additional coverage for dashboard.session_store helpers.
"""
from __future__ import annotations

import pytest
import pandas as pd

from dashboard import session_store
from dashboard.session_store import ScoredRun, build_scored_run


def test_json_helpers_and_coercion():
    assert session_store._is_jsonable({"a": 1}) is True
    assert session_store._is_jsonable({1, 2}) is False

    df = pd.DataFrame([{"a": 1, "b": 2}])
    series = pd.Series([1, 2, 3])
    assert session_store._coerce_jsonable(df) == [{"a": 1, "b": 2}]
    assert session_store._coerce_jsonable(series) == [1, 2, 3]

    complex_value = {"meta": {"x": 1}, "items": (1, 2)}
    coerced = session_store._coerce_jsonable(complex_value)
    assert coerced["meta"]["x"] == 1
    assert coerced["items"] == (1, 2)

    class NotJsonable:
        def __repr__(self) -> str:
            return "NotJsonable()"

    assert session_store._coerce_jsonable(NotJsonable()) == "NotJsonable()"


def test_build_scored_run_flags_and_meta():
    rows = [
        {"flagged": "yes", "amount": 1},
        {"flagged": "0", "amount": 2},
        {"flagged": True, "amount": 3},
    ]
    run = build_scored_run(
        rows,
        threshold=0.7,
        pct_flagged=0.5,
        pct_auto_categorised=0.0,
    )
    assert run.run_meta["flagged_count"] == 2
    assert run.run_meta["tx_count"] == 3
    assert run.rows[0]["flagged"] is True
    assert run.rows[1]["flagged"] is False


def test_save_scored_run_with_scored_run_instance():
    rows = [{"flagged": False, "amount": 5}]
    run = build_scored_run(
        rows,
        threshold=0.5,
        pct_flagged=0.0,
        pct_auto_categorised=0.0,
    )
    session = {}
    session_store.save_scored_run(session, run)

    assert session_store.SESSION_SCORED_ROWS_KEY in session
    assert session_store.SESSION_DIAGS_KEY in session
    assert session_store.LEGACY_SCORED_ROWS_KEY in session
    assert session_store.LEGACY_DIAGS_KEY in session

    loaded = session_store.load_scored_run(session)
    assert isinstance(loaded, ScoredRun)
    assert loaded["rows"] == run.rows
    assert loaded["diags"]["threshold"] == 0.5


def test_save_scored_run_requires_diags_for_raw_rows():
    with pytest.raises(TypeError):
        session_store.save_scored_run({}, [{"x": 1}], None)
    with pytest.raises(TypeError):
        session_store.save_scored_run({}, [{"x": 1}], "oops")


def test_load_scored_rows_and_diags_legacy_and_invalid():
    session = {session_store.LEGACY_SCORED_ROWS_KEY: [{"a": 1}, "bad"]}
    assert session_store.load_scored_rows(session) == [{"a": 1}]

    session = {session_store.SESSION_SCORED_ROWS_KEY: "nope"}
    assert session_store.load_scored_rows(session) is None

    session = {session_store.LEGACY_DIAGS_KEY: {"threshold": 0.9}}
    assert session_store.load_diags(session)["threshold"] == 0.9

    session = {"scored_run": {"rows": [], "diags": {"threshold": 0.8}}}
    assert session_store.load_diags(session)["threshold"] == 0.8

    session = {session_store.SESSION_DIAGS_KEY: "nope"}
    assert session_store.load_diags(session) is None


def test_clear_scored_run_removes_all_keys():
    session = {
        session_store.SESSION_SCORED_ROWS_KEY: [{"x": 1}],
        session_store.SESSION_DIAGS_KEY: {"threshold": 0.7},
        session_store.LEGACY_SCORED_ROWS_KEY: [{"x": 1}],
        session_store.LEGACY_DIAGS_KEY: {"threshold": 0.7},
        "scored_run": {"rows": [{"x": 1}], "diags": {"threshold": 0.7}},
    }
    session_store.clear_scored_run(session)
    assert session == {}
