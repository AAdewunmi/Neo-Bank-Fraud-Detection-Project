[![CI](https://github.com/AAdewunmi/Neo-Bank-Fraud-Detection-Project/actions/workflows/ci.yml/badge.svg)](https://github.com/AAdewunmi/Neo-Bank-Fraud-Detection-Project/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-80%25%20min-brightgreen)](https://github.com/AAdewunmi/Neo-Bank-Fraud-Detection-Project/actions)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![Django](https://img.shields.io/badge/django-5.x-darkgreen)](https://www.djangoproject.com/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED)](https://www.docker.com/)
[![Docker Compose](https://img.shields.io/badge/docker%20compose-supported-2496ED)](https://docs.docker.com/compose/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)


You’re right to call that out. A README is a **single full artefact**, not snippets, so the correct move here is to **regenerate the entire file cleanly**, incorporating the changes end to end. Below is a **fully regenerated README** with:

* More neutral, factual deployment wording
* A short, concrete Railway deploy appendix
* A one-line roadmap note about public deployment
* No meta commentary, no justification language, no portfolio signalling

You can replace your current README with this verbatim.

---

```markdown
# LedgerGuard (Neo-Bank Fraud Detection Project)

LedgerGuard is a production-minded fraud detection and transaction categorisation system built around a neo-bank use case. By Week 4, the system closes the loop from model output to operational action: analyst edits are captured as structured feedback, business rules can override model categories with audit tags, a read-only customer experience is introduced, and the application is packaged for deployment with a healthcheck.

Status: Work in progress (Week 4). This README reflects the current end-to-end system and will continue to evolve as persistence, monitoring, and automated retraining are extended.

---

## What exists in Week 4

### Capabilities
- Transaction categorisation baseline using TF IDF with Logistic Regression
- Fraud risk baseline using Isolation Forest
- Unified scorer interface returning category, confidence, fraud risk, and flagged status
- CSV ingestion with schema validation and coercion rules
- Django Ops Dashboard to upload CSVs, view KPIs, filter rows, and inspect scored results
- Feedback loop for analyst category edits, merged in-session and exportable as `feedback_edits.csv`
- Stable row identifiers (`row_id`) to keep feedback durable across filtering, ordering, and truncation
- Rules overlay for category overrides with audit tagging (`category_source = model | rule | edit`)
- Preview mode for large uploads with capped rows while preserving full-run totals
- Performance page that reads `model_registry.json` and surfaces model metadata and metrics
- Read-only customer dashboard with limited data exposure and customer flag capture
- Comprehensive test suite with coverage enforcement in CI

### Design goals (Week 4)
- Deterministic, reproducible behaviour where possible
- Explicit contracts between ingestion, scoring, rules, persistence, and UI
- Fast failure with informative error states for invalid uploads
- Operational feedback treated as first-class data
- Clear auditability of category provenance

---

## Tech stack (Week 4)
- Python 3.11
- Django 5.x
- PostgreSQL
- pandas, scikit-learn
- pytest, pytest-django, coverage
- Docker and Docker Compose
- gunicorn for production serving

---

## Repository layout (high level)
- `dashboard/` Ops workflows, ingestion services, scoring orchestration, and UI rendering
- `customer/` Read-only customer-facing views and templates
- `ml/` Training and inference code for baseline models
- `rules/` Category override definitions
- `tests/` Unit and integration-style tests
- `docs/` Contracts, notes, and screenshots
- `artefacts/` Model artefacts, metrics, and insight assets
- `data/` Local sample datasets

---

## Data contract

Expected CSV columns:
- `timestamp`
- `amount`
- `customer_id`
- `merchant`
- `description`

Policy:
- Required columns must be present
- Amounts are coerced to numeric
- Text fields are normalised
- `customer_id` must be non-empty

Detailed coercion and rejection rules are documented in `docs/`.

---

## Ops Dashboard workflows

### Upload and score
- Upload a CSV
- Select a fraud threshold
- View KPIs and a scored preview table

Large uploads use preview mode, which:
- Caps rows rendered in the UI
- Preserves total transaction and flagged counts
- Prioritises flagged rows in the preview

### Filtering
Supported filters include:
- flagged only
- customer_id
- merchant contains
- category contains
- minimum fraud risk

### Category provenance
Each row carries a source tag:
- `model` for model output
- `rule` for rule-based override
- `edit` for analyst override

Precedence is explicit:
- edit overrides rule
- rule overrides model

### Analyst edits and feedback export
Categories can be edited inline. Edits are keyed by stable `row_id` and merged across submits.

Exported feedback includes:
- `row_id`
- transaction identity fields
- `predicted_category`
- `new_category`
- `edited_at`

---

## Customer site

The customer site is a separate, read-only surface designed to expose a minimal, privacy-safe view of transaction data.

Scope:
- Display recent transactions and aggregated spend by category
- Allow customers to flag a transaction as not recognised with an optional note

Constraints:
- No fraud scores, thresholds, model outputs, or Ops controls
- Explicit field allow-listing and row filtering
- Customer actions stored as a separate feedback stream

Customer flags are persisted independently and exportable as `customer_flags.csv` for downstream review.


---

## Rules overlay

Rules are defined in:
- `rules/category_overrides.json`

Rule structure:
- case-insensitive substring match
- applied to merchant and description
- mapped to a category label

Rules are designed to be simple, auditable, and fast to change, with explicit precedence against model output.

---

## Local setup

### Prerequisites
- Python 3.11
- Virtualenv
- PostgreSQL
- Optional Docker and Docker Compose

### Environment configuration
- Copy `.env.example` to `.env`
- Set `DATABASE_URL` to a running PostgreSQL instance
- Load environment variables into the shell before running Django

---

## Running tests

Run the test suite:
- `pytest -q`

Run coverage:
- `coverage run -m pytest -q`
- `coverage report`

CI enforces minimum coverage thresholds.

---

## Running locally

Start the development server:
- `python manage.py runserver`

Expected behaviour:
- CSV upload and scoring
- Filterable results
- Inline edits that persist on refresh
- Exportable feedback files
- Clear error states for invalid inputs

---

## Healthcheck

A lightweight health endpoint is available:
- `GET /health/`
- Returns `{"ok": true}`

This endpoint is suitable for container orchestration and platform probes.

---

## Running via Docker (production parity)

Build and run the application using Docker:
- Image includes gunicorn and static asset collection
- Configuration is environment-driven
- Same image can be used locally or on PaaS platforms

---

## Deployment status

LedgerGuard is packaged as a deployable application with:
- Gunicorn-based entrypoint
- Dockerfile suitable for PaaS builds
- Environment-based configuration
- Healthcheck endpoint
- PostgreSQL-only configuration

Deployment workflows are verified locally and in CI. A public deployment endpoint is not currently exposed.

---

## Railway deployment (appendix)

LedgerGuard can be deployed to Railway using the existing Dockerfile.

High-level steps:
- Create a new Railway project
- Add a PostgreSQL service
- Set environment variables:
  - `DEBUG=False`
  - `ALLOWED_HOSTS=<railway-hostname>`
  - `DATABASE_URL` from Railway Postgres
- Set start command:
  - `gunicorn neobank_site.wsgi:application --bind 0.0.0.0:$PORT`
- Deploy from the repository

The same configuration applies to other container-based platforms.

---

## Common gotchas
- Git ignore rules can unintentionally exclude code
- Contract drift between scoring output and UI expectations
- Preview truncation affecting debugging of large uploads

--- 

## Maintainer
Adrian Adewunmi

## Repository
https://github.com/AAdewunmi/Neo-Bank-Fraud-Detection-Project
```


