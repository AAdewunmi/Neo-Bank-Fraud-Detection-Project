"""
Views for the dashboard app.

"""

# dashboard/views.py
from __future__ import annotations

import logging
import os
from typing import Any, Dict

from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render

from dashboard.decorators import ops_access_required
from dashboard.forms import ScoreForm
from dashboard.services import read_csv_file, score_transactions
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

    context: Dict[str, Any] = {
        "form": ScoreForm(),
        "run": stored_run,
        "has_results": bool(stored_run.rows),
        "error": None,
    }

    if request.method == "POST":
        form = ScoreForm(request.POST, request.FILES)
        context["form"] = form

        if not form.is_valid():
            context["error"] = "Invalid form input."
            return render(request, "dashboard/index.html", context)

        uploaded = form.cleaned_data["file"]
        threshold = float(form.cleaned_data["threshold"])

        try:
            df = read_csv_file(uploaded)
            scored_df, diags = score_transactions(df=df, threshold=threshold)

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
            return redirect("dashboard:index")

        except Exception as exc:
            logger.exception("Scoring failed")
            context["error"] = str(exc)
            return render(request, "dashboard/index.html", context)

    return render(request, "dashboard/index.html", context)
