# 💳 FinSight: Neo-Bank Transaction Categorisation & Fraud Risk Dashboard

**Duration:** 4 Weeks (20 Lab Days, Mon–Fri)
**Format:** Hands-on postgraduate programming lab
**Focus:** NLP for Categorisation • Fraud Modelling • Django Dashboards • CI/CD • Risk Communication

---

## 🎯 Course Overview

Neo-banks need reliable spend categorisation and low-friction fraud detection to inform customers and protect revenue. This lab guides you through building **FinSight** — an end-to-end pipeline that auto-labels merchant transactions and assigns a fraud risk score, surfaced in a Django + Bootstrap review dashboard with exportable workflows and threshold analysis.

You’ll progress from a strict data contract and deterministic baselines to embedding-based classifiers, supervised fraud models, model insights (PR curves), and a deployed, reviewer-friendly tool with rules overlays and a feedback loop.

---

## 📆 Weekly Structure

| Week                             | Theme                                                       | Core Skills                                                                                         | Key Deliverables                                                                                                  |
| -------------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **1 — Reproducible Foundations** | Data contract, baselines, tests, CI                         | Schema design • TF-IDF+LR • Isolation Forest • pytest • coverage • GitHub Actions                   | `docs/data_contract.md`, `train_categorisation.py`, `train_fraud_baseline.py`, `model_registry.json`, CI badge    |
| **2 — Django MVP Dashboard**     | Upload → score → filter → export                            | Django forms/views/templates • Bootstrap UI • server-side filtering • UX & error states             | `dashboard/` MVP, KPIs, flag export, screenshots, `tests/test_dashboard_*.py`, tag `v0.2`                         |
| **3 — Signal Uplift & Insights** | Embeddings + LightGBM; supervised fraud; PR/threshold plots | Sentence-Transformers (MiniLM) • LightGBM • XGBoost • PR-AUC • Matplotlib insights • feature parity | `train_categorisation_embeddings.py`, `train_fraud_supervised.py`, PNG plots, feature parity tests, model cards   |
| **4 — Feedback, Rules & Deploy** | Edit loop, rules overlay, container deploy, perf polish     | Human-in-the-loop design • Rules precedence • Gunicorn/Docker • Healthcheck • Perf instrumentation  | Inline edits + `feedback_edits.csv`, `rules/category_overrides.json`, `/health`, live URL, postmortem, tag `v1.0` |

---

## 🧪 Learning Outcomes

By completing FinSight, you will be able to:

1. **Engineer reproducible ML pipelines** with deterministic text vectorisers, persisted artefacts, and a model registry.
2. **Build categorisers and fraud detectors** (TF-IDF+LR → MiniLM+LightGBM; Isolation Forest → XGBoost) with imbalance handling.
3. **Evaluate trade-offs** using PR-AUC and threshold vs precision/recall to inform business policy.
4. **Develop a reviewer-ready Django dashboard** with upload, filtering, confidence cues, and CSV exports.
5. **Operationalise responsibly** with CI, tests, containerised deploys, model cards, and a concise **Risk Memo**.

---

## 📚 Assessment & Artifacts

* ✅ **Source with tests** (pytest/pytest-django, coverage ≥ 80%, CI passing)
* ✅ **Model artefacts + registry** (`artefacts/*.joblib`, `model_registry.json`, model cards)
* ✅ **Deployed dashboard** (cloud URL + `/health`) with screenshots/GIF
* ✅ **Insights pack** (PR curve, threshold trade-offs) and **Risk Memo** (`docs/risk_memo.md`)
* ✅ **Reflective posts** (weekly LinkedIn/Medium summaries)

---

## ✍️ Reflective Practice

Weekly short reflections to consolidate technical and communication skills:

* *Week 1 – “Data Contracts & Determinism: Making ML Reproducible”*
* *Week 2 – “From Pipeline to People: Shipping a Useful MVP”*
* *Week 3 – “Signal, Not Hype: Embeddings, PR-AUC, and Parity”*
* *Week 4 – “Rules, Feedback, and the Path to Production”*

---

## 🧩 Tools & Stack

**Languages:** Python 3.11 • HTML/CSS/JS (Bootstrap)
**Libraries:** Django, pandas, scikit-learn, sentence-transformers (MiniLM), LightGBM, XGBoost, imbalanced-learn, Matplotlib, pytest/pytest-django, factory_boy
**Infrastructure:** GitHub Actions, Docker, Gunicorn, Render/Railway (cloud)
**Data:** Kaggle PaySim / Credit Card Fraud (labels optional) + synthetic merchant descriptions

---

## 💬 Final Deliverable

A reproducible, deployed **Neo-bank categorisation & fraud risk dashboard** with transparent model insights, rules overlays, and a feedback export — ready for portfolio review, recruiter demos, or capstone assessment.

