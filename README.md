[![CI](https://github.com/AAdewunmi/Neo-Bank-Fraud-Detection-Project/actions/workflows/ci.yml/badge.svg)](https://github.com/AAdewunmi/Neo-Bank-Fraud-Detection-Project/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-80%25%20min-brightgreen)](https://github.com/AAdewunmi/Neo-Bank-Fraud-Detection-Project/actions)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![Django](https://img.shields.io/badge/django-5.x-darkgreen)](https://www.djangoproject.com/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED)](https://www.docker.com/)
[![Docker Compose](https://img.shields.io/badge/docker%20compose-supported-2496ED)](https://docs.docker.com/compose/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)


```markdown
# LedgerGuard (Neo-Bank Fraud Detection Project)

LedgerGuard is a portfolio-grade, production-minded fraud and transaction categorisation prototype for a neo-bank setting. Week 1 focuses on correctness, repeatability, and clean interfaces: deterministic baselines, schema-validated ingestion, and a small dashboard that exercises the full flow end to end.

Status: Work in progress (Week 1). This README will be revised as new weeks add persistence, monitoring, and deployment.

---

## What exists in Week 1

### Capabilities
- Transaction categorisation baseline (TF IDF with Logistic Regression)
- Fraud risk baseline (Isolation Forest)
- Unified scorer interface (one entry point that returns category, confidence, fraud risk, and flagged status)
- CSV ingestion with schema validation and coercion rules
- Minimal Django dashboard to upload a CSV and view KPIs and a scored table
- Test suite with coverage gate in CI

### Design goals (Week 1)
- Deterministic, reproducible runs where possible (stable tests and diffs in CI)
- Explicit input and output contracts between ingestion, scoring, and UI
- Fail fast, fail informative for bad CSV uploads

---

## Tech stack (Week 1)
- Python 3.11
- Django 5.x
- pandas, scikit-learn
- pytest, pytest-django, coverage
- Docker, Docker Compose (for container workflow)

---

## Repository layout (high level)
- `dashboard/` Django app for upload, ingestion services, and rendering results
- `ml/` training and inference code (baselines, scorer, utilities)
- `tests/` unit and integration-style tests
- `docs/` contracts and operational notes (grows over time)
- `data/` local sample datasets (not committed if large or sensitive)

---

## Data contract (Week 1)

Expected CSV columns:
- `timestamp`
- `amount`
- `customer_id`
- `merchant`
- `description`

Coercion and rejection rules are documented in `docs/` (Week 1 policy: reject missing required columns, coerce amount, normalise text fields, require non-empty customer_id).

---

## Local setup (Week 1)

### Prerequisites
- Python 3.11 installed
- Virtualenv available
- Optional: Docker and Docker Compose

### 1) Create a virtualenv and install deps
- Create `.venv`
- Install requirements
- Activate the environment

(Use whatever workflow you already follow for this repo.)

### 2) Environment variables
This project uses environment variables for configuration. For Week 1, you can load them into your shell before running Django:

- Copy `.env.example` to `.env`
- Edit values as needed

One reliable way to load `.env` into the current shell session:
- `set -a; source .env; set +a`

Note: If you run Django locally (non-Docker) and you are using Postgres, your DB host typically needs to be `localhost` rather than `db`. If your `.env` points at `db`, that is intended for Docker networking.

---

## Step 1: Run tests (local)

Run the test suite:
- `pytest -q`

Run coverage:
- `coverage run -m pytest -q`
- `coverage report`

CI enforces a minimum coverage threshold. If CI fails, check the coverage report for the missing lines and add targeted tests for those branches.

---

## Step 2: Run the app locally (dev server)

Load env vars and start the server:
- `set -a; source .env; set +a; python manage.py runserver`

Open:
- http://127.0.0.1:8000/

Week 1 expected behaviour:
- Upload `data/sample_transactions.csv`
- Click the validate and score action
- See KPIs and a scored table (category, confidence, risk, flag)
- If a CSV is missing required columns, see a readable error message

---

## Step 3: Run via Docker

### Quick checks
List service names:
- `docker compose config --services`

### A) Start only Postgres (useful when running Django on host)
- `docker compose up -d db`

If your local Django process connects to Postgres, set host to `localhost` for local runs (either in `.env` or via exported env vars).

### B) Start the full stack in containers
If you have a web service defined in `docker-compose.yml`:
- `docker compose up -d --build`

Rebuild is important after dependency or code changes inside the image.

View logs:
- `docker compose logs -f`

Stop:
- `docker compose down`

---

## Common Week 1 gotchas

- Git ignore patterns can accidentally exclude code (for example, patterns intended for datasets). If a module is missing in CI, verify with:
  - `git check-ignore -v path/to/file.py`

- Contract mismatch bugs are normal early on: a model can run correctly while the UI fails because expected diagnostic keys or output columns drifted. Week 1 fixes this by making contracts explicit and testing them.

---

## Roadmap

### Week 1
- Deterministic baselines
- Unified scoring contract
- Schema-validated ingestion
- Dashboard wiring
- CI hardening and one-command UX improvements
- README v1

### Future weeks (planned)
- Persistence (Postgres models, audit logs)
- Monitoring and metrics (scoring drift, ingestion failures)
- Background jobs (Celery/Redis) where appropriate
- Deployment workflow

---

## Maintainer
Adrian Adewunmi

## Repository
https://github.com/AAdewunmi/Neo-Bank-Fraud-Detection-Project
```
