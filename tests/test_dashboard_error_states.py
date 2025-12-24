"""
Error-state tests for the fraud operations dashboard.

These tests focus on upload and scoring failures so that the Week 2
server-side error handling paths remain covered and predictable.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

DASHBOARD_URL_NAME = "dashboard"
SESSION_ROWS_KEY = "dashboard_scored_rows"
SESSION_DIAGS_KEY = "dashboard_diags"


@pytest.mark.django_db
def test_invalid_upload_sets_error_and_does_not_persist_run(
    client,
    django_user_model,
) -> None:
    """
    When the upload form is invalid the view reports an error and
    leaves the session without any scored run.
    """
    user = django_user_model.objects.create_user(
        username="ops_invalid",
        password="pass1234",
        is_staff=True,
    )
    client.force_login(user)

    url = reverse(DASHBOARD_URL_NAME)
    # No file and no threshold fields supplied.
    response = client.post(url, data={})

    assert response.status_code == 200
    context = response.context
    assert context is not None

    # The view signals a generic form validation issue.
    assert context["error"] == "Invalid form input."
    assert context["table"] == []

    session = client.session
    assert SESSION_ROWS_KEY not in session
    assert SESSION_DIAGS_KEY not in session


@pytest.mark.django_db
def test_read_csv_failure_sets_error_and_does_not_store_run(
    monkeypatch,
    client,
    django_user_model,
) -> None:
    """
    If the CSV cannot be parsed the view surfaces the exception message
    and skips creating a new scored run in the session.
    """
    user = django_user_model.objects.create_user(
        username="ops_parse_error",
        password="pass1234",
        is_staff=True,
    )
    client.force_login(user)

    url = reverse(DASHBOARD_URL_NAME)

    def fake_read_csv(file_obj: Any) -> None:
        raise ValueError("CSV parsing failed in test")

    monkeypatch.setattr("dashboard.views.services.read_csv", fake_read_csv)

    upload = SimpleUploadedFile(
        "transactions.csv",
        b"account_id,amount\n1,10.0\n",
        content_type="text/csv",
    )

    response = client.post(
        url,
        data={"threshold": "0.5", "csv_file": upload},
    )

    assert response.status_code == 200
    context = response.context
    assert context is not None
    assert context["error"] == "CSV parsing failed in test"

    session = client.session
    assert SESSION_ROWS_KEY not in session
    assert SESSION_DIAGS_KEY not in session


@pytest.mark.django_db
def test_score_df_failure_sets_error_and_does_not_store_run(
    monkeypatch,
    client,
    django_user_model,
) -> None:
    """
    If scoring fails after CSV parsing the view reports the error and
    avoids writing partial state for the run into the session.
    """
    user = django_user_model.objects.create_user(
        username="ops_scoring_error",
        password="pass1234",
        is_staff=True,
    )
    client.force_login(user)

    url = reverse(DASHBOARD_URL_NAME)

    def fake_read_csv(file_obj: Any) -> pd.DataFrame:
        # Minimal frame; content is not important for this test.
        return pd.DataFrame([{"amount": 10.0}])

    def fake_score_df(df: pd.DataFrame, threshold: float) -> None:
        raise ValueError("Scoring failed in test")

    monkeypatch.setattr("dashboard.views.services.read_csv", fake_read_csv)
    monkeypatch.setattr("dashboard.views.services.score_df", fake_score_df)

    upload = SimpleUploadedFile(
        "transactions.csv",
        b"amount\n10.0\n",
        content_type="text/csv",
    )

    response = client.post(
        url,
        data={"threshold": "0.5", "csv_file": upload},
    )

    assert response.status_code == 200
    context = response.context
    assert context is not None
    assert context["error"] == "Scoring failed in test"

    session = client.session
    assert SESSION_ROWS_KEY not in session
    assert SESSION_DIAGS_KEY not in session
