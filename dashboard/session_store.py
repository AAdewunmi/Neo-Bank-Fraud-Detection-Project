"""
Session helpers for storing scored transaction runs.

This module keeps the shape of data written to the Django session small
and JSON-friendly so that it works cleanly with the standard JSON-based
session backends.

There are two layers of API:

1. Row / diagnostics helpers
   - save_scored_rows / load_scored_rows
   - save_diags / load_diags

2. Higher-level aggregate
   - ScoredRun dataclass
   - build_scored_run, save_scored_run, load_scored_run

The higher-level API is used by the dashboard view. The row helpers are
used by the CSV export view and remain available for backwards
compatibility.
"""

# dashboard/session_store.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

try:
    import numpy as np
except Exception:
    np = None

try:
    import pandas as pd
except Exception:
    pd = None

SESSION_SCORED_ROWS_KEY = "ledgerguard_scored_rows"
SESSION_SCORED_DIAGS_KEY = "ledgerguard_scored_diags"
SESSION_DIAGS_KEY = SESSION_SCORED_DIAGS_KEY
LEGACY_SCORED_ROWS_KEY = "dashboard_scored_rows"
LEGACY_DIAGS_KEY = "dashboard_diags"
LEGACY_SCORED_RUN_KEY = "scored_run"
LEGACY_SIMPLE_ROWS_KEY = "scored_rows"
LEGACY_SIMPLE_DIAGS_KEY = "scored_diags"


def _is_jsonable(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, Mapping):
        return all(_is_jsonable(v) for v in value.values())
    if isinstance(value, list):
        return all(_is_jsonable(v) for v in value)
    if isinstance(value, tuple):
        return all(_is_jsonable(v) for v in value)
    return False


def _coerce_iterable(values: Iterable[Any], as_tuple: bool = False) -> Any:
    coerced = [_coerce_jsonable(v) for v in values]
    return tuple(coerced) if as_tuple else coerced


def _coerce_jsonable(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if pd is not None:
        if isinstance(value, pd.DataFrame):
            return [_coerce_jsonable(r) for r in value.to_dict(orient="records")]
        if isinstance(value, pd.Series):
            return _coerce_iterable(value.tolist())

    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()

    if isinstance(value, Mapping):
        return {str(k): _coerce_jsonable(v) for k, v in value.items()}

    if isinstance(value, list):
        return _coerce_iterable(value)
    if isinstance(value, tuple):
        return _coerce_iterable(value, as_tuple=True)
    if isinstance(value, set):
        return _coerce_iterable(sorted(value, key=str))

    return str(value)


@dataclass(frozen=True)
class ScoredRun:
    """
    Container for a scored transaction run stored in the session.

    rows holds the preview rows only.
    run_meta holds totals and KPIs for the full run.
    """

    rows: List[Dict[str, Any]]
    run_meta: Dict[str, Any]

    def __contains__(self, key: object) -> bool:
        return key in {"rows", "diags"}

    def __getitem__(self, key: str) -> Any:
        if key == "rows":
            return self.rows
        if key == "diags":
            return self.run_meta
        raise KeyError(key)


def build_scored_run(
    scored_rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    pct_flagged: float,
    pct_auto_categorised: float,
    flagged_key: str = "flagged",
    total_tx_count: Optional[int] = None,
    flagged_count_total: Optional[int] = None,
    rows_truncated: bool = False,
) -> ScoredRun:
    normalised_rows: List[Dict[str, Any]] = []
    flagged_count_preview = 0

    for row in scored_rows:
        normalised = dict(row)
        if flagged_key in normalised:
            flag_val = _coerce_flag_value(normalised.get(flagged_key))
            normalised[flagged_key] = flag_val
        else:
            flag_val = False
        if flag_val:
            flagged_count_preview += 1
        normalised_rows.append(_coerce_jsonable(normalised))

    rows_shown = int(len(normalised_rows))
    tx_count = (
        int(total_tx_count) if total_tx_count is not None else rows_shown
    )
    flagged_count = (
        int(flagged_count_total)
        if flagged_count_total is not None
        else flagged_count_preview
    )

    run_meta = {
        "threshold": float(threshold),
        "pct_flagged": float(pct_flagged),
        "pct_auto_categorised": float(pct_auto_categorised),
        "tx_count": tx_count,
        "flagged_count": flagged_count,
        "rows_shown": rows_shown,
        "rows_truncated": bool(rows_truncated),
        "flagged_key": str(flagged_key),
    }

    return ScoredRun(rows=normalised_rows, run_meta=_coerce_jsonable(run_meta))


def save_scored_run(session: Any, run: Any, diags: Optional[Mapping[str, Any]] = None) -> None:
    if isinstance(run, ScoredRun):
        rows = run.rows
        run_diags = run.run_meta
    else:
        if diags is None or not isinstance(diags, Mapping):
            raise TypeError("save_scored_run requires diagnostics when saving raw rows.")
        rows = run
        run_diags = diags

    session[SESSION_SCORED_ROWS_KEY] = _coerce_jsonable(rows)
    session[SESSION_SCORED_DIAGS_KEY] = _coerce_jsonable(run_diags)
    session[LEGACY_SCORED_ROWS_KEY] = _coerce_jsonable(rows)
    session[LEGACY_DIAGS_KEY] = _coerce_jsonable(run_diags)
    session[LEGACY_SCORED_RUN_KEY] = {
        "rows": _coerce_jsonable(rows),
        "diags": _coerce_jsonable(run_diags),
    }
    _mark_session_modified(session)


def load_scored_run(session: Any) -> Optional[ScoredRun]:
    rows = load_scored_rows(session)
    diags = load_diags(session)
    if rows is None and diags is None:
        return None
    if rows is None or diags is None:
        return None
    return ScoredRun(rows=rows, run_meta=diags)


def clear_scored_run(session: Any) -> None:
    session.pop(SESSION_SCORED_ROWS_KEY, None)
    session.pop(SESSION_SCORED_DIAGS_KEY, None)
    session.pop(LEGACY_SCORED_ROWS_KEY, None)
    session.pop(LEGACY_DIAGS_KEY, None)
    session.pop(LEGACY_SCORED_RUN_KEY, None)
    session.pop(LEGACY_SIMPLE_ROWS_KEY, None)
    session.pop(LEGACY_SIMPLE_DIAGS_KEY, None)
    _mark_session_modified(session)


def save_scored_rows(session: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    """
    Backwards-compatible helper to persist scored rows to session.
    """
    payload = [_coerce_jsonable(row) for row in rows]
    session[SESSION_SCORED_ROWS_KEY] = payload
    session[LEGACY_SCORED_ROWS_KEY] = payload
    session[LEGACY_SCORED_RUN_KEY] = {
        "rows": payload,
        "diags": session.get(SESSION_SCORED_DIAGS_KEY) or session.get(LEGACY_DIAGS_KEY) or {},
    }
    _mark_session_modified(session)


def load_scored_rows(session: Any) -> Optional[List[Dict[str, Any]]]:
    """
    Backwards-compatible helper to load scored rows from session.
    """
    if SESSION_SCORED_ROWS_KEY in session:
        rows = session.get(SESSION_SCORED_ROWS_KEY)
    elif LEGACY_SCORED_ROWS_KEY in session:
        rows = session.get(LEGACY_SCORED_ROWS_KEY)
    elif LEGACY_SCORED_RUN_KEY in session:
        run = session.get(LEGACY_SCORED_RUN_KEY) or {}
        rows = run.get("rows")
    else:
        return None

    if not isinstance(rows, list):
        return None

    cleaned: List[Dict[str, Any]] = []
    for row in rows:
        if isinstance(row, Mapping):
            cleaned.append(dict(row))
    return cleaned


def save_diags(session: Any, diags: Mapping[str, Any]) -> None:
    """
    Backwards-compatible helper to persist diagnostics to session.
    """
    payload = _coerce_jsonable(diags)
    session[SESSION_SCORED_DIAGS_KEY] = payload
    session[LEGACY_DIAGS_KEY] = payload
    session[LEGACY_SCORED_RUN_KEY] = {
        "rows": session.get(SESSION_SCORED_ROWS_KEY) or session.get(LEGACY_SCORED_ROWS_KEY) or [],
        "diags": payload,
    }
    _mark_session_modified(session)


def load_diags(session: Any) -> Optional[Dict[str, Any]]:
    """
    Backwards-compatible helper to load diagnostics from session.
    """
    if SESSION_SCORED_DIAGS_KEY in session:
        diags = session.get(SESSION_SCORED_DIAGS_KEY)
    elif LEGACY_DIAGS_KEY in session:
        diags = session.get(LEGACY_DIAGS_KEY)
    elif LEGACY_SCORED_RUN_KEY in session:
        run = session.get(LEGACY_SCORED_RUN_KEY) or {}
        diags = run.get("diags")
    else:
        return None

    if not isinstance(diags, Mapping):
        return None
    return dict(diags)


def _coerce_flag_value(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "t"}:
            return True
        if lowered in {"0", "false", "no", "n", ""}:
            return False
    return bool(value)


def _mark_session_modified(session: Any) -> None:
    if hasattr(session, "modified"):
        session.modified = True
