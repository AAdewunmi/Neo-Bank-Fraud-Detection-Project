# syntax=docker/dockerfile:1

FROM python:3.11-slim

# Keep Python logs unbuffered and avoid writing .pyc files inside containers.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# PostgreSQL client headers are required for psycopg2 builds in some environments.
# If you're using psycopg2-binary, this is still harmless and keeps builds reliable.
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
       gcc \
       libpq-dev \
  && rm -rf /var/lib/apt/lists/*

# Install dependencies first for better Docker layer caching.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip \
  && pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application code.
COPY . /app

EXPOSE 8000

# docker-compose.yml overrides this with a command that also runs migrations.
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
