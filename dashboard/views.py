"""
dashboard/views.py

Views for the dashboard app.

This module preserves existing Week 1–3 behaviour:
- CSV upload and scoring via dashboard.services
- Session-backed scored run via dashboard.session_store
- Filters via FilterForm
- KPI summary and insights timestamp display
- Performance view sourced from model_registry.json

Week 4 additions (Day 1 feedback loop):
- Stable row_id per row to make feedback durable across ordering changes
- Session-backed category edits keyed by row_id
- Merge edits rather than overwrite
- Overlay edits on render so Ops sees changes immediately
- Export feedback edits as feedback_edits.csv with stable identifiers

Notes:
- Precedence in this file is edit > existing category values.
- Rules overlay (edit > rule > model) can be layered later by stamping category_source
  and adjusting precedence, without changing edit persistence mechanics.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import os
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pandas as pd
from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST

from dashboard import services
from dashboard.decorators import ops_access_required
from dashboard.forms import FilterForm, UploadForm
from dashboard.session_store import (
    build_scored_run,
    clear_scored_run,
    load_scored_run,
    save_scored_run,
)
from ml.training.utils import load_registry

logger = logging.getLogger(__name__)

CATEGORY_EDITS_SESSION_KEY = "category_edits"


def public_home(request: HttpRequest) -> HttpResponse:
    """
    Public landing page.
    """
    return render(request, "dashboard/public_home.html")


@ops_access_required
def reset_run(request: HttpRequest) -> HttpResponse:
    """
    Clears the current scored run from session.
    """
    if request.method == "POST":
        clear_scored_run(request.session)
    return redirect("dashboard:index")


@ops_access_required
def performance(request: HttpRequest) -> HttpResponse:
    """
    Model performance page driven by model_registry.json.
    """
    registry_path = Path(settings.BASE_DIR) / "model_registry.json"
    registry = load_registry(str(registry_path))

    fraud_info = _build_model_summary(registry, "fraud")
    cat_info = _build_model_summary(registry, "categorisation")

    timestamps = [t for t in [fraud_info.get("updated_at"), cat_info.get("updated_at")] if t]
    updated_at = max(timestamps) if timestamps else None

    context = {
        "fraud": fraud_info,
        "categorisation": cat_info,
        "updated_at": updated_at,
        "insights_generated_at": updated_at,
    }
    return render(request, "dashboard/performance.html", context)


@ops_access_required
def index(request: HttpRequest) -> HttpResponse:
    """
    Dashboard index.

    GET:
        - Loads the current scored run from session.
        - Ensures each row has a stable row_id.
        - Overlays any stored category edits so the UI reflects corrections.
        - Applies filter form (flagged only, customer, merchant, category, min risk).

    POST:
        - Handles CSV upload and scoring using existing Week 1–3 services.
        - Stores a preview run in session_store.
        - Ensures stable row_id values exist in preview rows.
        - Overlays existing edits so previously corrected rows show as edited.
    """
    stored_run = load_scored_run(request.session)

    # Ensure current rows have stable row_id, then overlay edits for display.
    base_rows: List[Mapping[str, Any]] = stored_run.rows if stored_run else []
    run_meta = stored_run.run_meta if stored_run else {}

    base_rows_with_ids = _ensure_row_ids(list(base_rows))
    edits = _get_category_edits(request.session)
    edits = _migrate_index_based_edits_if_needed(request.session, base_rows_with_ids, edits)
    display_rows = _overlay_category_edits(base_rows_with_ids, edits)

    filtered_rows, view_meta = _apply_filters(
        request, display_rows, run_meta.get("flagged_key", "flagged")
    )

    context: Dict[str, Any] = {
        "form": UploadForm(),
        "filter_form": FilterForm(request.GET or None),
        "run": stored_run,
        "has_results": bool(base_rows),
        "table": list(filtered_rows),
        "view_meta": view_meta,
        "run_meta": run_meta,
        "threshold": run_meta.get("threshold"),
        "export_available": bool(base_rows),
        "kpi": _build_kpis(run_meta),
        "insights_generated_at": _format_insights_timestamp(
            _resolve_insights_timestamp(run_meta)
        ),
        "error": None,
    }

    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        context["form"] = form

        if not form.is_valid():
            context["error"] = "Invalid form input."
            return render(request, "dashboard/index.html", context)

        uploaded = form.cleaned_data["csv_file"]
        threshold = float(form.cleaned_data["threshold"])

        try:
            df = services.read_csv(uploaded)
            scored_df, diags = services.score_df(df=df, threshold=threshold)

            diags_payload = {
                "threshold": float(threshold),
                "n": int(diags.get("n", len(scored_df))),
                "pct_flagged": float(diags.get("pct_flagged", 0.0)),
                "pct_auto_categorised": float(diags.get("pct_auto_categorised", 0.0)),
                "flagged_key": str(diags.get("flagged_key", "flagged")),
            }

            flagged_key = diags_payload["flagged_key"]
            total_tx_count = int(len(scored_df))

            if flagged_key in scored_df.columns:
                flagged_count_total = int(scored_df[flagged_key].astype(bool).sum())
            elif "fraud_flag" in scored_df.columns:
                flagged_count_total = int(scored_df["fraud_flag"].astype(bool).sum())
            else:
                flagged_count_total = int(round(diags_payload["pct_flagged"] * total_tx_count))

            max_preview_rows = int(os.environ.get("LEDGERGUARD_DASHBOARD_MAX_ROWS", "500"))
            preview_df = scored_df.head(max_preview_rows)
            rows_truncated = bool(total_tx_count > len(preview_df))

            # Preserve your current behaviour: prioritise flagged rows in preview.
            if rows_truncated:
                flagged_col = None
                if flagged_key in scored_df.columns:
                    flagged_col = flagged_key
                elif "fraud_flag" in scored_df.columns:
                    flagged_col = "fraud_flag"

                if flagged_col:
                    flagged_rows = scored_df[scored_df[flagged_col].astype(bool)]
                    if not flagged_rows.empty:
                        remaining_slots = max_preview_rows - len(flagged_rows)
                        if remaining_slots > 0:
                            remainder = scored_df[~scored_df[flagged_col].astype(bool)].head(
                                remaining_slots
                            )
                            preview_df = pd.concat([flagged_rows, remainder], ignore_index=True)
                        else:
                            preview_df = flagged_rows.head(max_preview_rows)
                        rows_truncated = bool(total_tx_count > len(preview_df))

            # Convert to records and attach stable row_id so edits can key reliably.
            preview_records: List[Dict[str, Any]] = preview_df.to_dict(orient="records")
            preview_records = _ensure_row_ids(preview_records)

            # Stamp audit defaults if your scoring pipeline does not set them.
            for r in preview_records:
                r.setdefault("category_source", "model")
                r.setdefault("predicted_category", r.get("category"))

            scored_run = build_scored_run(
                preview_records,
                threshold=diags_payload["threshold"],
                pct_flagged=diags_payload["pct_flagged"],
                pct_auto_categorised=diags_payload["pct_auto_categorised"],
                flagged_key=flagged_key,
                total_tx_count=total_tx_count,
                flagged_count_total=flagged_count_total,
                rows_truncated=rows_truncated,
                scored_at=timezone.now().isoformat(),
            )

            save_scored_run(request.session, scored_run)

            # Re-load and display with edit overlay (so previously edited rows show immediately).
            stored_run = load_scored_run(request.session)
            base_rows = stored_run.rows if stored_run else []
            run_meta = stored_run.run_meta if stored_run else {}

            base_rows_with_ids = _ensure_row_ids(list(base_rows))
            edits = _get_category_edits(request.session)
            edits = _migrate_index_based_edits_if_needed(
                request.session, base_rows_with_ids, edits
            )
            display_rows = _overlay_category_edits(base_rows_with_ids, edits)

            filtered_rows, view_meta = _apply_filters(
                request,
                display_rows,
                run_meta.get("flagged_key", "flagged"),
            )

            context.update(
                {
                    "run": stored_run,
                    "has_results": bool(base_rows),
                    "table": list(filtered_rows),
                    "view_meta": view_meta,
                    "run_meta": run_meta,
                    "threshold": run_meta.get("threshold"),
                    "export_available": bool(base_rows),
                    "kpi": _build_kpis(run_meta),
                    "insights_generated_at": _format_insights_timestamp(
                        _resolve_insights_timestamp(run_meta)
                    ),
                }
            )
            return render(request, "dashboard/index.html", context)

        except Exception as exc:
            logger.exception("Scoring failed")
            context["error"] = str(exc)
            return render(request, "dashboard/index.html", context)

    return render(request, "dashboard/index.html", context)


@require_POST
@ops_access_required
def apply_edit(request: HttpRequest) -> HttpResponse:
    """
    Persist a single inline category edit.

    Expected POST fields:
        - row_id: stable 64-hex identifier for the row
        - new_category: edited category string

    Storage:
        - request.session[CATEGORY_EDITS_SESSION_KEY] is a dict keyed by row_id
        - Each value includes stable identity columns needed for later retraining/export

    Behaviour:
        - Merges into existing edits, does not overwrite the whole edit set.
        - Latest edit for a given row_id wins.
    """
    row_id = str(request.POST.get("row_id", "")).strip()
    new_category = str(request.POST.get("new_category", "")).strip()

    if not row_id or len(row_id) != 64:
        return redirect("dashboard:index")

    if not new_category:
        return redirect("dashboard:index")

    # Small production-style guardrail, keeps categories sane in a demo.
    if len(new_category) > 50:
        new_category = new_category[:50]

    stored_run = load_scored_run(request.session)
    if not stored_run or not stored_run.rows:
        return redirect("dashboard:index")

    rows_with_ids = _ensure_row_ids(list(stored_run.rows))
    row_lookup = {str(r.get("row_id", "")): r for r in rows_with_ids}
    base = row_lookup.get(row_id)

    if base is None:
        return redirect("dashboard:index")

    edits = _get_category_edits(request.session)

    payload = {
        "row_id": row_id,
        # Stable identifiers for retraining join.
        "timestamp": str(base.get("timestamp", "")),
        "customer_id": str(base.get("customer_id", "")),
        "amount": str(base.get("amount", "")),
        "merchant": str(base.get("merchant", "")),
        "description": str(base.get("description", "")),
        # Audit of what the system predicted before human correction.
        "predicted_category": str(base.get("predicted_category", base.get("category", ""))),
        # The analyst correction.
        "new_category": new_category,
        # Optional metadata, useful later.
        "edited_at": timezone.now().isoformat(),
    }

    edits[row_id] = payload
    request.session[CATEGORY_EDITS_SESSION_KEY] = edits
    request.session.modified = True

    return redirect("dashboard:index")


@require_GET
@ops_access_required
def export_feedback(request: HttpRequest) -> HttpResponse:
    """
    Export analyst edits as feedback_edits.csv.

    CSV columns:
        row_id,timestamp,customer_id,amount,merchant,description,predicted_category,new_category,edited_at

    Notes:
        - Deterministic ordering by row_id to make diffs stable.
        - This export is designed as a retraining input in later labs.
    """
    edits = _get_category_edits(request.session)

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(
        [
            "row_id",
            "timestamp",
            "customer_id",
            "amount",
            "merchant",
            "description",
            "predicted_category",
            "new_category",
            "edited_at",
        ]
    )

    for rid in sorted(edits.keys()):
        e = edits[rid] or {}
        w.writerow(
            [
                e.get("row_id", rid),
                e.get("timestamp", ""),
                e.get("customer_id", ""),
                e.get("amount", ""),
                e.get("merchant", ""),
                e.get("description", ""),
                e.get("predicted_category", ""),
                e.get("new_category", ""),
                e.get("edited_at", ""),
            ]
        )

    resp = HttpResponse(buf.getvalue(), content_type="text/csv")
    resp["Content-Disposition"] = "attachment; filename=feedback_edits.csv"
    return resp


def _apply_filters(
    request: HttpRequest,
    rows: List[Mapping[str, Any]],
    flagged_key: str,
) -> tuple[List[Mapping[str, Any]], Dict[str, Any]]:
    """
    Apply FilterForm filters to the displayed table rows.

    Important:
        This function expects rows already have edits overlaid so that filters reflect
        the edited category values.
    """
    form = FilterForm(request.GET or None)
    if not form.is_valid() or not rows:
        return list(rows), {"visible_count": len(rows)}

    filtered = list(rows)
    if form.cleaned_data.get("flagged_only"):
        filtered = [r for r in filtered if _coerce_flagged(r.get(flagged_key))]

    customer_id = form.cleaned_data.get("customer_id")
    if customer_id:
        filtered = [r for r in filtered if str(r.get("customer_id", "")) == str(customer_id)]

    merchant = form.cleaned_data.get("merchant")
    if merchant:
        needle = str(merchant).lower()
        filtered = [r for r in filtered if needle in str(r.get("merchant", "")).lower()]

    category = form.cleaned_data.get("category")
    if category:
        needle = str(category).lower()
        filtered = [r for r in filtered if needle in str(r.get("category", "")).lower()]

    min_risk = form.cleaned_data.get("min_fraud_risk")
    if min_risk is not None:
        filtered = [
            r
            for r in filtered
            if _coerce_float(r.get("fraud_risk")) is not None
            and _coerce_float(r.get("fraud_risk")) >= float(min_risk)
        ]

    return filtered, {"visible_count": len(filtered)}


def _coerce_flagged(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "t"}:
            return True
        if lowered in {"0", "false", "no", "n", ""}:
            return False
    return bool(value)


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _canonicalise_text(value: Any) -> str:
    """
    Convert a value into a stable string representation.

    Normalises whitespace and guards None.
    """
    if value is None:
        return ""
    return str(value).strip()


def _canonicalise_amount(value: Any) -> str:
    """
    Canonicalise amount to reduce row_id drift due to formatting.

    Example: "10.0" and "10" should map to the same canonical amount.
    """
    if value is None:
        return ""
    try:
        d = Decimal(str(value).strip())
        return format(d.normalize(), "f")
    except (InvalidOperation, ValueError):
        return _canonicalise_text(value)


def _compute_row_id(row: Mapping[str, Any]) -> str:
    """
    Compute a deterministic row_id from a canonical subset of fields.

    This is stable across filtering and ordering, and is suitable as a join key
    for feedback edits.
    """
    parts = [
        _canonicalise_text(row.get("timestamp")),
        _canonicalise_text(row.get("customer_id")),
        _canonicalise_amount(row.get("amount")),
        _canonicalise_text(row.get("merchant")),
        _canonicalise_text(row.get("description")),
    ]
    canonical = "|".join(parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _ensure_row_ids(rows: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure each row dict includes a stable row_id.

    Args:
        rows: list of row mappings (likely loaded from session_store rows)

    Returns:
        A list of mutable dicts where each has a "row_id" field.
    """
    out: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        if not d.get("row_id"):
            d["row_id"] = _compute_row_id(d)
        out.append(d)
    return out


def _get_category_edits(session: Any) -> Dict[str, Dict[str, Any]]:
    """
    Load edits from session.

    Current format:
        { row_id: { row_id, ..., new_category, ... } }

    Backward compatibility:
        If older format exists, migration is handled elsewhere.
    """
    raw = session.get(CATEGORY_EDITS_SESSION_KEY, {})
    if isinstance(raw, dict):
        return raw
    return {}


def _migrate_index_based_edits_if_needed(
    session: Any,
    rows: List[Mapping[str, Any]],
    edits: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Backward compatibility helper.

    If session stored edits as a list of {"row_index": idx, "new_category": "..."},
    migrate it to the row_id keyed format using the current row ordering.

    This is best-effort migration and only applies when:
    - session[CATEGORY_EDITS_SESSION_KEY] is a list
    - rows exist and indices are in range

    Once migrated, session is updated to the dict format.
    """
    raw = session.get(CATEGORY_EDITS_SESSION_KEY)

    if isinstance(raw, list) and rows:
        migrated: Dict[str, Dict[str, Any]] = dict(edits)
        rows_with_ids = _ensure_row_ids(list(rows))

        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("row_index"))
            except (TypeError, ValueError):
                continue
            if idx < 0 or idx >= len(rows_with_ids):
                continue

            base = rows_with_ids[idx]
            rid = str(base.get("row_id", "")).strip()
            if not rid:
                continue

            new_category = str(item.get("new_category", "")).strip()
            if not new_category:
                continue

            migrated[rid] = {
                "row_id": rid,
                "timestamp": str(base.get("timestamp", "")),
                "customer_id": str(base.get("customer_id", "")),
                "amount": str(base.get("amount", "")),
                "merchant": str(base.get("merchant", "")),
                "description": str(base.get("description", "")),
                "predicted_category": str(base.get("predicted_category", base.get("category", ""))),
                "new_category": new_category,
                "edited_at": timezone.now().isoformat(),
            }

        session[CATEGORY_EDITS_SESSION_KEY] = migrated
        session.modified = True
        return migrated

    return edits


def _overlay_category_edits(
    rows: List[Mapping[str, Any]],
    edits: Dict[str, Dict[str, Any]],
) -> List[Mapping[str, Any]]:
    """
    Overlay edits onto the displayed rows.

    If an edit exists for a row_id:
        - set category to new_category
        - set category_source to 'edit'
        - preserve predicted_category for audit

    If no edit exists:
        - ensure category_source is present (default 'model')
    """
    rows_with_ids = _ensure_row_ids(rows)
    out: List[Dict[str, Any]] = []

    for r in rows_with_ids:
        rid = str(r.get("row_id", "")).strip()
        row = dict(r)

        if rid and rid in edits:
            e = edits[rid]
            row.setdefault("predicted_category", row.get("category"))
            row["category"] = e.get("new_category", row.get("category"))
            row["category_source"] = "edit"
        else:
            row.setdefault("category_source", "model")
            row.setdefault("predicted_category", row.get("category"))

        out.append(row)

    return out


def _build_model_summary(registry: dict[str, Any], section: str) -> Dict[str, Any]:
    entry = None
    latest = registry.get(section, {}).get("latest")
    if latest:
        entry = registry[section].get(latest)

    updated_at = _parse_registry_timestamp(latest) if latest else None
    summary = {
        "latest": latest,
        "entry": entry,
        "metrics": {},
        "card_json": None,
        "card_pretty": None,
        "summary_rows": [],
        "card_path": None,
        "metrics_path": None,
        "updated_at": updated_at,
    }

    if not entry:
        return summary

    metrics_path = entry.get("metrics_path")
    if metrics_path:
        summary["metrics_path"] = metrics_path
        metrics = _read_json_file(metrics_path)
        if metrics:
            summary["metrics"] = metrics
    else:
        summary["metrics"] = entry.get("metrics", {})

    summary["summary_rows"] = _build_summary_rows(section, entry, summary["metrics"])

    model_path = entry.get("artefact")
    if model_path:
        card_path = Path(model_path).with_suffix(".card.json")
        if card_path.exists():
            summary["card_path"] = str(card_path)
            card_json = _read_json_file(str(card_path))
            summary["card_json"] = card_json
            summary["card_pretty"] = _pretty_json(card_json)

    return summary


def _build_summary_rows(
    section: str, entry: dict[str, Any], metrics: dict[str, Any]
) -> List[tuple[str, Any]]:
    rows: List[tuple[str, Any]] = []
    if section == "fraud":
        rows.append(("Average precision", metrics.get("average_precision")))
        rows.append(("Dataset", entry.get("dataset")))
        rows.append(("Label source", entry.get("label_source") or metrics.get("label_source")))
        rows.append(("Split", entry.get("split_type")))
    elif section == "categorisation":
        rows.append(("Macro F1", metrics.get("macro_f1")))
        rows.append(("Embeddings status", metrics.get("embeddings_status")))
        rows.append(("Label mode", entry.get("label_mode")))
    return rows


def _read_json_file(path: str) -> Dict[str, Any]:
    try:
        import json

        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _pretty_json(payload: Dict[str, Any]) -> str:
    try:
        import json

        return json.dumps(payload, indent=2, sort_keys=True)
    except Exception:
        return ""


def _parse_registry_timestamp(version: str) -> datetime | None:
    if not version:
        return None
    digits = "".join(ch for ch in version if ch.isdigit())
    if len(digits) < 14:
        return None
    try:
        ts = datetime.strptime(digits[-14:], "%Y%m%d%H%M%S")
        return timezone.make_aware(ts, timezone.get_current_timezone())
    except ValueError:
        return None


def _build_kpis(run_meta: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "tx_count": int(run_meta.get("tx_count", run_meta.get("n", 0)) or 0),
        "pct_flagged": float(run_meta.get("pct_flagged", 0.0) or 0.0),
        "pct_auto_cat": float(run_meta.get("pct_auto_categorised", 0.0) or 0.0),
    }


def _load_insights_timestamp() -> str | None:
    base_dir = Path(getattr(settings, "BASE_DIR", "."))
    for path in (
        base_dir / "artefacts" / "fraud_insights_timestamp.txt",
        base_dir / "neobank_site" / "static" / "artefacts" / "fraud_insights_timestamp.txt",
    ):
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = handle.read().strip()
                return value or None
        except FileNotFoundError:
            continue
    return None


def _resolve_insights_timestamp(run_meta: Mapping[str, Any]) -> str | None:
    file_ts = _load_insights_timestamp()
    run_ts = run_meta.get("scored_at") if isinstance(run_meta, Mapping) else None
    if not file_ts:
        return run_ts
    if not run_ts:
        return file_ts
    file_dt = pd.to_datetime(file_ts, errors="coerce")
    run_dt = pd.to_datetime(run_ts, errors="coerce")
    if pd.isna(file_dt) or pd.isna(run_dt):
        return run_ts or file_ts
    return run_ts if run_dt >= file_dt else file_ts


def _format_insights_timestamp(value: str | None) -> str | None:
    if not value:
        return None
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return value
    return dt.strftime("%b %-d, %Y %H:%M")
