# Dockerfile
# syntax=docker/dockerfile:1

FROM python:3.11-slim

# Environment defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/cache/huggingface \
    LEDGERGUARD_MAX_EMBED_ROWS=2000

WORKDIR /app

# System deps:
# - build-essential: compiler toolchain (covers gcc and friends)
# - libpq-dev: Postgres headers for psycopg2 builds
# - libgomp1: OpenMP runtime (often needed by lightgbm)
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
       build-essential \
       libpq-dev \
       libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt /app/requirements.txt
COPY requirements/ /app/requirements/

RUN python -m pip install --upgrade pip \
  && python -m pip install --no-cache-dir -r /app/requirements.txt

# Optional ML dependencies for Week 3 labs
ARG INSTALL_ML=0
ARG PRELOAD_ENCODER=0

RUN if [ "$INSTALL_ML" = "1" ]; then \
      echo "[LedgerGuard] Installing ML dependencies..." && \
      python -m pip install --no-cache-dir -r /app/requirements/ml.txt; \
    fi

# Copy the rest of the source tree
COPY . /app

# Preload encoder into HF cache if both ML and preload flags are enabled
RUN mkdir -p /app/cache/huggingface && \
    if [ "$INSTALL_ML" = "1" ] && [ "$PRELOAD_ENCODER" = "1" ]; then \
      echo "[LedgerGuard] Preloading sentence-transformer encoder into cache..." && \
      python -m ml.scripts.preload_encoder; \
    fi

# Collect static assets for production deploys.
ARG COLLECTSTATIC=0
RUN if [ "$COLLECTSTATIC" = "1" ]; then \
      python manage.py collectstatic --noinput; \
    fi

EXPOSE 8000

# docker-compose.yml can override this CMD
CMD ["gunicorn", "neobank_site.wsgi:application", "--bind", "0.0.0.0:8000"]
