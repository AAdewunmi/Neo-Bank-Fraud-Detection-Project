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

from __future__ import annotations

import json
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

SESSION_SCORED_ROWS_KEY = "dashboard_scored_rows"
SESSION_DIAGS_KEY = "dashboard_diags"
LEGACY_SCORED_ROWS_KEY = "scored_rows"
LEGACY_DIAGS_KEY = "scored_diags"


def _is_jsonable(value: Any) -> bool:
    """
    Return True if the value can be serialised by json.dumps.

    This check is defensive and used to decide when coercion is needed.
    """
    try:
        json.dumps(value)
    except TypeError:
        return False
    return True


def _coerce_jsonable(value: Any) -> Any:
    """
    Convert common scientific Python types into JSON-serialisable
    structures.

    The dashboard stores scored runs in the Django session. Built-in
    types are already safe. This helper deals with pandas objects,
    NumPy types and nested containers.
    """
    # Local imports avoid importing heavy libraries when they are not
    # installed in a minimal environment that only reads plain Python
    # types from the session.
    try:  # pragma: no cover - import guard
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - import guard
        np = None  # type: ignore[assignment]

    try:  # pragma: no cover - import guard
        import pandas as pd  # type: ignore
    except Exception:  # pragma: no cover - import guard
        pd = None  # type: ignore[assignment]

    if _is_jsonable(value):
        return value

    if pd is not None:
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
        if isinstance(value, pd.Series):
            return value.tolist()

    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()

    if isinstance(value, Mapping):
        return {str(k): _coerce_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_coerce_jsonable(v) for v in value]

    # Fallback: store a string representation so that the session layer
    # never sees unsupported objects.
    return str(value)


@dataclass(frozen=True)
class ScoredRun:
    """
    Container for a scored transaction run stored in the session.

    Attributes
    ----------
    rows:
        List of dicts representing scored transactions. Each row is safe
        to serialise to JSON.
    run_meta:
        Aggregate metrics used for KPIs and diagnostics. Values are
        JSON-friendly.
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
) -> ScoredRun:
    """
    Build a ScoredRun instance from scored rows and diagnostics.

    Parameters
    ----------
    scored_rows:
        Sequence of row mappings returned from the scoring pipeline.
    threshold:
        Decision threshold used during scoring.
    pct_flagged:
        Percentage of transactions flagged by the model.
    pct_auto_categorised:
        Percentage of transactions that can be auto-categorised.
    flagged_key:
        Name of the boolean flag column that marks suspicious rows.

    Returns
    -------
    ScoredRun
        A container with JSON-friendly rows and metadata.
    """
    normalised_rows: List[Dict[str, Any]] = []
    flagged_count = 0

    for row in scored_rows:
        normalised = dict(row)
        flag_val = normalised.get(flagged_key)

        if isinstance(flag_val, str):
            lowered = flag_val.strip().lower()
            flag_val = lowered in {"1", "true", "yes", "y"}

        normalised[flagged_key] = bool(flag_val)
        if normalised[flagged_key]:
            flagged_count += 1

        normalised_rows.append(_coerce_jsonable(normalised))

    tx_count = len(normalised_rows)

    run_meta: Dict[str, Any] = {
        "threshold": float(threshold),
        "tx_count": tx_count,
        "flagged_key": flagged_key,
        "flagged_count": flagged_count,
        "pct_flagged": float(pct_flagged),
        "pct_auto_categorised": float(pct_auto_categorised),
    }

    return ScoredRun(rows=normalised_rows, run_meta=_coerce_jsonable(run_meta))


def save_scored_rows(
    session: MutableMapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """
    Persist scored transaction rows into the session.

    Parameters
    ----------
    session:
        Django session object (or any mapping-like object).
    rows:
        Sequence of row dictionaries to persist.
    """
    payload = [_coerce_jsonable(r) for r in rows]
    session[SESSION_SCORED_ROWS_KEY] = payload
    session[LEGACY_SCORED_ROWS_KEY] = payload


def save_diags(
    session: MutableMapping[str, Any],
    diags: Mapping[str, Any],
) -> None:
    """
    Persist diagnostic metrics for the last scored run into the session.

    Parameters
    ----------
    session:
        Django session object (or any mapping-like object).
    diags:
        Mapping of diagnostic values to store.
    """
    payload = _coerce_jsonable(dict(diags))
    session[SESSION_DIAGS_KEY] = payload
    session[LEGACY_DIAGS_KEY] = payload


def save_scored_run(
    session: MutableMapping[str, Any],
    run_or_rows: Any,
    diags: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Store the last scored run in the session.

    This helper supports both the legacy and the new calling
    conventions to keep older tests and views working:

    - New style:
        save_scored_run(session, run)

      where ``run`` is a ScoredRun instance.

    - Legacy style:
        save_scored_run(session, scored_rows, diags)

      where ``scored_rows`` is a sequence of row mappings and
      ``diags`` is a diagnostics mapping.
    """
    if isinstance(run_or_rows, ScoredRun) and diags is None:
        run = run_or_rows
        rows = run.rows
        meta = run.run_meta
    else:
        rows = run_or_rows
        if diags is None:
            raise TypeError(
                "diags must be provided when save_scored_run is called "
                "with raw rows."
            )
        if not isinstance(diags, Mapping):
            raise TypeError("diags must be a mapping.")
        meta = dict(diags)
        run = ScoredRun(rows=list(rows), run_meta=meta)

    save_scored_rows(session, rows)
    save_diags(session, meta)

    # The ScoredRun instance is returned for convenience in situations
    # where callers want a strongly typed object as well as session
    # side-effects.
    # This return is not used by the current views but keeps the helper
    # broadly useful.
    return run  # type: ignore[return-value]


def load_scored_rows(session: Mapping[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    Load scored rows from the session.

    Parameters
    ----------
    session:
        Django session object (or any mapping-like object).

    Returns
    -------
    list[dict] | None
        A list of row dictionaries if present and well-formed,
        otherwise None.
    """
    rows = session.get(SESSION_SCORED_ROWS_KEY)
    if rows is None:
        legacy_run = session.get("scored_run")
        if isinstance(legacy_run, Mapping):
            rows = legacy_run.get("rows")
    if rows is None:
        rows = session.get(LEGACY_SCORED_ROWS_KEY)
    if not isinstance(rows, list):
        return None

    cleaned: List[Dict[str, Any]] = []
    for row in rows:
        if isinstance(row, Mapping):
            cleaned.append(dict(row))

    if not cleaned:
        return None

    return cleaned


def load_diags(session: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Load diagnostics for the last run from the session.

    Parameters
    ----------
    session:
        Django session object (or any mapping-like object).

    Returns
    -------
    dict | None
        A dictionary of diagnostics if present and well-formed,
        otherwise None.
    """
    diags = session.get(SESSION_DIAGS_KEY)
    if diags is None:
        legacy_run = session.get("scored_run")
        if isinstance(legacy_run, Mapping):
            diags = legacy_run.get("diags")
    if diags is None:
        diags = session.get(LEGACY_DIAGS_KEY)
    if not isinstance(diags, Mapping):
        return None
    return dict(diags)


def clear_scored_run(session: MutableMapping[str, Any]) -> None:
    """
    Remove all scored-run related keys from the session.

    This clears both the current and legacy session keys.
    """
    for key in (
        SESSION_SCORED_ROWS_KEY,
        SESSION_DIAGS_KEY,
        LEGACY_SCORED_ROWS_KEY,
        LEGACY_DIAGS_KEY,
        "scored_run",
    ):
        session.pop(key, None)


def load_scored_run(session: Mapping[str, Any]) -> Optional[ScoredRun]:
    """
    Reconstruct a ScoredRun from session data.

    The function uses the same underlying keys as the row / diagnostics
    helpers so older code that only knows about those keys still works.

    Parameters
    ----------
    session:
        Django session object (or any mapping-like object).

    Returns
    -------
    ScoredRun | None
        A ScoredRun instance if both rows and diagnostics are available,
        otherwise None.
    """
    rows = load_scored_rows(session)
    diags = load_diags(session)

    if not rows or diags is None:
        return None

    return ScoredRun(rows=rows, run_meta=diags)
