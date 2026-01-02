from __future__ import annotations

import json
from pathlib import Path

from django.conf import settings


def test_performance_page_renders_latest_metrics(
    client, django_user_model, tmp_path, monkeypatch
) -> None:
    user = django_user_model.objects.create_user(
        username="ops", password="pass1234", is_staff=True
    )
    client.login(username="ops", password="pass1234")

    monkeypatch.setattr(settings, "BASE_DIR", tmp_path)

    artefacts_dir = tmp_path / "artefacts"
    reports_dir = tmp_path / "reports"
    artefacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    fraud_model = artefacts_dir / "fraud_xgb_20250101010101.joblib"
    fraud_model.write_text("stub", encoding="utf-8")
    fraud_card = fraud_model.with_suffix(".card.json")
    fraud_card.write_text(
        json.dumps({"artefact": str(fraud_model), "dataset_sha1": "abc", "metrics": {}}),
        encoding="utf-8",
    )
    fraud_metrics_path = reports_dir / "fraud_metrics.json"
    fraud_metrics_path.write_text(
        json.dumps({"average_precision": 0.42}),
        encoding="utf-8",
    )

    cat_model = artefacts_dir / "cat_model_20250101010101.joblib"
    cat_model.write_text("stub", encoding="utf-8")
    cat_card = cat_model.with_suffix(".card.json")
    cat_card.write_text(
        json.dumps({"artefact": str(cat_model), "dataset_sha1": "def", "metrics": {}}),
        encoding="utf-8",
    )

    registry = {
        "fraud": {
            "latest": "fraud_xgb_20250101010101",
            "fraud_xgb_20250101010101": {
                "artefact": str(fraud_model),
                "metrics_path": str(fraud_metrics_path),
                "dataset": "paysim",
                "label_source": "paysim",
                "split_type": "time",
                "type": "supervised_xgb",
            },
        },
        "categorisation": {
            "latest": "cat_model_20250101010101",
            "cat_model_20250101010101": {
                "artefact": str(cat_model),
                "metrics": {"macro_f1": 0.12, "embeddings_status": "disabled"},
                "label_mode": "real",
                "type": "tfidf_logreg",
            },
        },
    }

    registry_path = Path(tmp_path) / "model_registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    resp = client.get("/ops/performance/")
    assert resp.status_code == 200
    content = resp.content.decode("utf-8")
    assert "Model Performance" in content
    assert "Average precision" in content
    assert "Macro F1" in content
