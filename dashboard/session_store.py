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
from typing import Any, Dict, List, Mapping, Optional, Sequence

try:
    import numpy as np
except Exception:
    np = None


SESSION_SCORED_ROWS_KEY = "ledgerguard_scored_rows"
SESSION_SCORED_DIAGS_KEY = "ledgerguard_scored_diags"


def _coerce_jsonable(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()

    if isinstance(value, Mapping):
        return {str(k): _coerce_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_coerce_jsonable(v) for v in value]

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
        flag_val = normalised.get(flagged_key)
        if bool(flag_val):
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


def save_scored_run(session: Any, run: ScoredRun) -> None:
    session[SESSION_SCORED_ROWS_KEY] = run.rows
    session[SESSION_SCORED_DIAGS_KEY] = run.run_meta
    session.modified = True


def load_scored_run(session: Any) -> ScoredRun:
    rows = session.get(SESSION_SCORED_ROWS_KEY, []) or []
    diags = session.get(SESSION_SCORED_DIAGS_KEY, {}) or {}
    return ScoredRun(rows=rows, run_meta=diags)


def clear_scored_run(session: Any) -> None:
    session.pop(SESSION_SCORED_ROWS_KEY, None)
    session.pop(SESSION_SCORED_DIAGS_KEY, None)
    session.modified = True


def save_scored_rows(session: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    """
    Backwards-compatible helper to persist scored rows to session.
    """
    session[SESSION_SCORED_ROWS_KEY] = [_coerce_jsonable(row) for row in rows]
    session.modified = True


def load_scored_rows(session: Any) -> List[Dict[str, Any]]:
    """
    Backwards-compatible helper to load scored rows from session.
    """
    rows = session.get(SESSION_SCORED_ROWS_KEY, []) or []
    return [dict(row) for row in rows]


def save_diags(session: Any, diags: Mapping[str, Any]) -> None:
    """
    Backwards-compatible helper to persist diagnostics to session.
    """
    session[SESSION_SCORED_DIAGS_KEY] = _coerce_jsonable(diags)
    session.modified = True


def load_diags(session: Any) -> Dict[str, Any]:
    """
    Backwards-compatible helper to load diagnostics from session.
    """
    diags = session.get(SESSION_SCORED_DIAGS_KEY, {}) or {}
    return dict(diags)
