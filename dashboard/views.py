"""
Views for the dashboard app.

"""

# dashboard/views.py
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Mapping

import pandas as pd
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from dashboard.decorators import ops_access_required
from dashboard.forms import FilterForm, UploadForm
from dashboard import services
from dashboard.session_store import (
    build_scored_run,
    load_scored_run,
    save_scored_run,
)

logger = logging.getLogger(__name__)


def public_home(request: HttpRequest) -> HttpResponse:
    """
    Public landing page.
    """
    return render(request, "dashboard/public_home.html")


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
        "insights_generated_at": _load_insights_timestamp(),
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
                    "insights_generated_at": _load_insights_timestamp(),
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


def _build_kpis(run_meta: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "tx_count": int(run_meta.get("tx_count", run_meta.get("n", 0)) or 0),
        "pct_flagged": float(run_meta.get("pct_flagged", 0.0) or 0.0),
        "pct_auto_cat": float(run_meta.get("pct_auto_categorised", 0.0) or 0.0),
    }


def _load_insights_timestamp() -> str | None:
    for path in (
        os.path.join("artefacts", "fraud_insights_timestamp.txt"),
        os.path.join("neobank_site", "static", "artefacts", "fraud_insights_timestamp.txt"),
    ):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                value = handle.read().strip()
                return value or None
        except FileNotFoundError:
            continue
    return None
