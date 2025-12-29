# Dockerfile
# syntax=docker/dockerfile:1


FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

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

# Install dependencies first for better Docker layer caching.
# Keep requirements.txt as the default entry point (it can include -r requirements/base.txt).
COPY requirements.txt /app/requirements.txt
COPY requirements/ /app/requirements/

RUN python -m pip install --upgrade pip \
  && python -m pip install --no-cache-dir -r /app/requirements.txt

# Optional ML dependencies for Week 3 labs
ARG INSTALL_ML=0
RUN if [ "$INSTALL_ML" = "1" ]; then python -m pip install --no-cache-dir -r /app/requirements/ml.txt; fi

# Copy the rest of the application code.
COPY . /app

EXPOSE 8000

# docker-compose.yml can override this with a command that also runs migrations.
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
