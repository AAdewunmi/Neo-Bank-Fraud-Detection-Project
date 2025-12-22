"""
Session storage helpers for dashboard scoring runs.

Goal:
- When a CSV is scored, persist the scored rows in session so:
  - export can pull from session without re-scoring
  - UI remains consistent with what the user just saw

Constraints:
- Payload must be JSON-serialisable.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

SESSION_SCORED_ROWS_KEY = "dashboard_scored_rows"
SESSION_DIAGS_KEY = "dashboard_last_diags"


def _coerce_jsonable(value: Any) -> Any:
    """
    Coerce common non-JSON values to JSON-friendly values.

    Args:
        value: Python object.

    Returns:
        JSON-serialisable representation.
    """
    if value is None:
        return ""

    if isinstance(value, (str, bool, int, float)):
        return value

    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.bool_):  # pragma: no cover
            return bool(value)
        if isinstance(value, np.integer):  # pragma: no cover
            return int(value)
        if isinstance(value, np.floating):  # pragma: no cover
            return float(value)
    except Exception:  # pragma: no cover
        pass

    return str(value)


def save_scored_run(session: Mapping[str, Any],
                    scored_rows: List[Dict[str, Any]], diags: Dict[str, Any]) -> None:
    """
    Save scored rows + diagnostics into session.

    Args:
        session: Django session object.
        scored_rows: list of dict rows (scored).
        diags: diagnostics dict from services.score_df.

    Returns:
        None.
    """
    safe_rows: List[Dict[str, Any]] = []
    for row in scored_rows:
        safe_rows.append({k: _coerce_jsonable(v) for k, v in row.items()})

    safe_diags = {k: _coerce_jsonable(v) for k, v in diags.items()}

    session[SESSION_SCORED_ROWS_KEY] = safe_rows
    session[SESSION_DIAGS_KEY] = safe_diags
    session.modified = True


def load_scored_rows(session: Mapping[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    Load scored rows from session.

    Args:
        session: Django session object.

    Returns:
        List of scored rows or None.
    """
    rows = session.get(SESSION_SCORED_ROWS_KEY)
    if not isinstance(rows, list):
        return None
    return rows


def load_diags(session: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Load diagnostics from session.

    Args:
        session: Django session object.

    Returns:
        Diagnostics dict or None.
    """
    diags = session.get(SESSION_DIAGS_KEY)
    if not isinstance(diags, dict):
        return None
    return diags
