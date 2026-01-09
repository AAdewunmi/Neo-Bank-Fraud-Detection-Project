[![CI](https://github.com/AAdewunmi/Neo-Bank-Fraud-Detection-Project/actions/workflows/ci.yml/badge.svg)](https://github.com/AAdewunmi/Neo-Bank-Fraud-Detection-Project/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-80%25%20min-brightgreen)](https://github.com/AAdewunmi/Neo-Bank-Fraud-Detection-Project/actions)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![Django](https://img.shields.io/badge/django-5.x-darkgreen)](https://www.djangoproject.com/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED)](https://www.docker.com/)
[![Docker Compose](https://img.shields.io/badge/docker%20compose-supported-2496ED)](https://docs.docker.com/compose/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

# Neo-bank Fraud and Transaction Categorisation System 

## LedgerGuard

LedgerGuard is a production-minded fraud detection and transaction categorisation system for a neo-bank setting, delivering an end-to-end ML pipeline that auto-labels merchant transactions and assigns a fraud risk score, surfaced through an Ops Dashboard for ingestion, scoring, review, and feedback, alongside a read-only Customer Dashboard for safe end-user visibility.

## Highlights

- CSV ingestion with validation and schema coercion
- Baseline category prediction (TF-IDF + Logistic Regression)
- Baseline fraud risk scoring (Isolation Forest)
- Unified scoring pipeline with category, confidence, fraud risk, and flagged state
- Ops Dashboard for uploads, KPIs, filtering, and review
- Analyst feedback loop with stable row IDs and exportable edits
- Rules overlay to override model category with audit tags
- Customer Dashboard with minimal exposure and customer flag capture
- Model performance page powered by `model_registry.json`
- Docker-ready deployment with healthcheck endpoint

## Quick Start

1. Install dependencies and set env vars:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

2. Demo users are not pre-seeded (no fixture or migration ships with accounts). If you want to use demo credentials, create them manually:

Ops admin:
- username: admin
- email: admin@neobank.com
- password: admin

Customer user:
- username: customer1
- password: pass1234

```bash
python manage.py migrate
python manage.py createsuperuser
```

Create a customer user via Django admin or your normal signup flow. Avoid a username that collides with a `customer_id` in your CSVs to prevent confusion when filtering or selecting a customer dashboard.

3. If you want to create your own admin + customer users, follow the same steps but pick your own credentials.

4. Start the server and upload demo data:

```bash
python manage.py runserver
```

Use demo data: `data/sample_transactions.csv`

## Screens and Docs

- Screenshots and notes: `docs/screenshots`
- Model registry: `model_registry.json`
- Rules configuration: `rules/category_overrides.json`

## Architecture Overview

- `dashboard/` Ops workflows, ingestion services, scoring orchestration, UI
- `customer_site/` Customer views and templates
- `ml/` Training and inference for baseline models
- `rules/` Category override definitions
- `artefacts/` Model artefacts and metrics
- `data/` Sample datasets
- `tests/` Unit and integration-style tests

## Data Contract

Expected CSV columns:

- `timestamp`
- `amount`
- `customer_id`
- `merchant`
- `description`

Contract guarantees:

- Required columns must be present
- Amounts are coerced to numeric
- Text fields are normalized
- `customer_id` must be non-empty

## Local Development (Python)

Prerequisites:

- Python 3.11
- PostgreSQL

Setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Update `.env` with your local database values, then run:

```bash
python manage.py migrate
python manage.py runserver
```

Ops Dashboard: http://127.0.0.1:8000/ops/

Customer Dashboard: http://127.0.0.1:8000/customer/

## Local Development (Docker)

```bash
docker compose up --build
```

The Docker setup includes Postgres and runs Django on port 8000.

## Environment Variables

Core settings (see `.env.example`):

- `DEBUG`
- `SECRET_KEY`
- `ALLOWED_HOSTS`
- `DATABASE_URL`

PostgreSQL settings (when not using `DATABASE_URL`):

- `POSTGRES_DB`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_HOST`
- `POSTGRES_PORT`

Optional integrations (present in `.env.example`):

- `REDIS_URL`
- `CELERY_BROKER_URL`
- `CELERY_RESULT_BACKEND`
- `EMAIL_HOST`
- `EMAIL_PORT`

## Ops Dashboard Workflow

1. Upload a CSV file.
2. Set a fraud threshold.
3. Review KPIs and scored results.
4. Filter rows by flags, customer, merchant, category, or fraud risk.
5. Edit categories and export feedback.

Category provenance is explicit:

- `model` for model output
- `rule` for rule-based override
- `edit` for analyst override

Precedence: edit > rule > model.

## Customer Dashboard

The customer site is a read-only surface that exposes a limited transaction view and allows customers to flag a transaction with an optional note. It deliberately hides any fraud scores, thresholds, or model output.

## Rules Overlay

Rules are stored in `rules/category_overrides.json` and applied using case-insensitive substring matching against merchant and description. This keeps overrides auditable and fast to change.

## Healthcheck

- `GET /health/`
- Returns `{ "ok": true }`

## Testing

Run the test suite:

```bash
pytest -q
```

Coverage:

```bash
coverage run -m pytest -q
coverage report
```

CI enforces an 80% minimum coverage threshold.

## Deployment

The project ships with a production Dockerfile and a gunicorn entrypoint. For a container platform:

- Set `DEBUG=False`
- Configure `ALLOWED_HOSTS`
- Provide `DATABASE_URL`
- Run database migrations

Example gunicorn start command:

```bash
gunicorn neobank_site.wsgi:application --bind 0.0.0.0:$PORT
```

## Render Deployment

Render is supported out of the box via the existing `render.yaml`.

High-level flow:

1. Create a new Render service from this repo.
2. Add a managed PostgreSQL database and set `DATABASE_URL`.
3. Set `DEBUG=False` and configure `ALLOWED_HOSTS`.
4. Trigger a deploy and run migrations if needed.

## License

MIT. See `LICENSE`.

## Maintainer

Adrian Adewunmi

## Repository

https://github.com/AAdewunmi/Neo-Bank-Fraud-Detection-Project
