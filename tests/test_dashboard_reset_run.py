from __future__ import annotations

import pytest
from django.urls import reverse

from dashboard.session_store import (
    LEGACY_SCORED_RUN_KEY,
    SESSION_SCORED_DIAGS_KEY,
    SESSION_SCORED_ROWS_KEY,
)

pytestmark = pytest.mark.django_db


def test_reset_run_clears_session(client, django_user_model) -> None:
    user = django_user_model.objects.create_user(
        username="ops", password="pass1234", is_staff=True
    )
    client.force_login(user)

    session = client.session
    session[SESSION_SCORED_ROWS_KEY] = [{"x": 1}]
    session[SESSION_SCORED_DIAGS_KEY] = {"threshold": 0.7}
    session[LEGACY_SCORED_RUN_KEY] = {"rows": [{"x": 1}], "diags": {"threshold": 0.7}}
    session.save()

    resp = client.post(reverse("dashboard:reset_run"))
    assert resp.status_code == 302

    session = client.session
    assert SESSION_SCORED_ROWS_KEY not in session
    assert SESSION_SCORED_DIAGS_KEY not in session
    assert LEGACY_SCORED_RUN_KEY not in session
