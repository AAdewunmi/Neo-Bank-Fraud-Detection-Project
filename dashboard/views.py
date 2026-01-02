"""
Views for the dashboard app.

"""

# dashboard/views.py
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping
from datetime import datetime

import pandas as pd
from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.utils import timezone

from dashboard.decorators import ops_access_required
from dashboard.forms import FilterForm, UploadForm
from dashboard import services
from dashboard.session_store import (
    build_scored_run,
    load_scored_run,
    save_scored_run,
)
from ml.training.utils import load_registry

logger = logging.getLogger(__name__)


def public_home(request: HttpRequest) -> HttpResponse:
    """
    Public landing page.
    """
    return render(request, "dashboard/public_home.html")


@ops_access_required
def performance(request: HttpRequest) -> HttpResponse:
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
    stored_run = load_scored_run(request.session)
    table_rows: List[Mapping[str, Any]] = stored_run.rows if stored_run else []
    run_meta = stored_run.run_meta if stored_run else {}

    filtered_rows, view_meta = _apply_filters(
        request, table_rows, run_meta.get("flagged_key", "flagged")
    )

    context: Dict[str, Any] = {
        "form": UploadForm(),
        "filter_form": FilterForm(request.GET or None),
        "run": stored_run,
        "has_results": bool(table_rows),
        "table": list(filtered_rows),
        "view_meta": view_meta,
        "run_meta": run_meta,
        "threshold": run_meta.get("threshold"),
        "export_available": bool(table_rows),
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

            scored_run = build_scored_run(
                preview_df.to_dict(orient="records"),
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
            filtered_rows, view_meta = _apply_filters(
                request,
                scored_run.rows,
                scored_run.run_meta.get("flagged_key", "flagged"),
            )
            context.update(
                {
                    "run": scored_run,
                    "has_results": bool(scored_run.rows),
                    "table": list(filtered_rows),
                    "view_meta": view_meta,
                    "run_meta": scored_run.run_meta,
                    "threshold": scored_run.run_meta.get("threshold"),
                    "export_available": bool(scored_run.rows),
                    "kpi": _build_kpis(scored_run.run_meta),
                    "insights_generated_at": _format_insights_timestamp(
                        _resolve_insights_timestamp(scored_run.run_meta)
                    ),
                }
            )
            return render(request, "dashboard/index.html", context)

        except Exception as exc:
            logger.exception("Scoring failed")
            context["error"] = str(exc)
            return render(request, "dashboard/index.html", context)

    return render(request, "dashboard/index.html", context)


def _apply_filters(
    request: HttpRequest,
    rows: List[Mapping[str, Any]],
    flagged_key: str,
) -> tuple[List[Mapping[str, Any]], Dict[str, Any]]:
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
