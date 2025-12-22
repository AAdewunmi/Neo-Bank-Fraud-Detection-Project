"""
Session storage for the Ops dashboard.

Purpose:
- Persist the last scored run in session so filtering and export are consistent with the UI.
- Keep payload JSON-serialisable (no DataFrames / numpy types).

Week 2 requirements:
- export reads from session
- export includes only flagged rows
- metadata is stored for "what did I just run?" traceability
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional


SESSION_SCORED_KEY = "dashboard_last_scored_run"


@dataclass(frozen=True)
class ScoredRun:
    """
    Serializable representation of the last scoring run.

    Attributes:
        fieldnames: Ordered headers for CSV export.
        rows: Scored rows rendered in the table (JSON-safe).
        flagged_key: Column name for flagged boolean.
        run_meta: Run metadata (threshold, counts, percentages, timestamp).
    """

    fieldnames: List[str]
    rows: List[Dict[str, Any]]
    flagged_key: str
    run_meta: Dict[str, Any]


def _utc_now_iso() -> str:
    """
    Return current UTC timestamp as an ISO string.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _coerce_jsonable(value: Any) -> Any:
    """
    Coerce common non-JSON types into JSON-safe values.

    Args:
        value: Any Python object.

    Returns:
        JSON-serialisable value where possible.
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


def build_scored_run(
    scored_rows: List[Dict[str, Any]],
    *,
    threshold: float,
    pct_flagged: float,
    pct_auto_categorised: float,
    flagged_key: str = "flagged",
) -> ScoredRun:
    """
    Build a ScoredRun payload suitable for session storage.

    Args:
        scored_rows: List of row dicts (what you render in the table).
        threshold: Fraud threshold used for scoring.
        pct_flagged: Fraction in [0,1] flagged by the model.
        pct_auto_categorised: Fraction in [0,1] where category is non-empty.
        flagged_key: Column name containing flagged boolean.

    Returns:
        ScoredRun payload for session.
    """
    safe_rows: List[Dict[str, Any]] = []
    for row in scored_rows:
        safe_rows.append({k: _coerce_jsonable(v) for k, v in row.items()})

    fieldnames = list(safe_rows[0].keys()) if safe_rows else []
    flagged_count = sum(1 for r in safe_rows if bool(r.get(flagged_key)) is True)

    run_meta = {
        "scored_at": _utc_now_iso(),
        "threshold": float(threshold),
        "tx_count": int(len(safe_rows)),
        "flagged_count": int(flagged_count),
        "pct_flagged": float(pct_flagged),
        "pct_auto_categorised": float(pct_auto_categorised),
    }

    return ScoredRun(
        fieldnames=fieldnames,
        rows=safe_rows,
        flagged_key=flagged_key,
        run_meta=run_meta,
    )


def save_scored_run(session: Mapping[str, Any], run: ScoredRun) -> None:
    """
    Save the last scored run into session.

    Args:
        session: Django request.session mapping-like object.
        run: ScoredRun payload.

    Returns:
        None.
    """
    session[SESSION_SCORED_KEY] = {
        "fieldnames": run.fieldnames,
        "rows": run.rows,
        "flagged_key": run.flagged_key,
        "run_meta": run.run_meta,
    }
    session.modified = True


def load_scored_run(session: Mapping[str, Any]) -> Optional[ScoredRun]:
    """
    Load last scored run from session.

    Args:
        session: Django request.session.

    Returns:
        ScoredRun if present and valid, otherwise None.
    """
    raw = session.get(SESSION_SCORED_KEY)
    if not isinstance(raw, dict):
        return None

    fieldnames = raw.get("fieldnames")
    rows = raw.get("rows")
    flagged_key = raw.get("flagged_key", "flagged")
    run_meta = raw.get("run_meta")

    if (
        not isinstance(fieldnames, list)
        or not isinstance(rows, list)
        or not isinstance(run_meta, dict)
    ):
        return None

    return ScoredRun(
        fieldnames=list(fieldnames),
        rows=list(rows),
        flagged_key=str(flagged_key),
        run_meta=dict(run_meta),
    )


def filter_flagged_rows(run: ScoredRun) -> List[Dict[str, Any]]:
    """
    Return only flagged rows from a scoring run.

    Args:
        run: ScoredRun payload.

    Returns:
        List of rows where flagged_key is True.
    """
    return [r for r in run.rows if bool(r.get(run.flagged_key)) is True]
