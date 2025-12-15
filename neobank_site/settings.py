"""
Django settings for neobank_site.

Week 1: minimal configuration to support routing + tests + templates,
using PostgreSQL via Docker.

Assumptions:
- In Docker Compose, the Postgres service is named "db" (so host is "db").
- Locally, you may use a .env file for convenience; in Docker, env vars are injected.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env if present (safe for local dev). In Docker, env vars can be injected directly.
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = "django-insecure-)d0ta*xwng@mu%7m*di6bkrjj_cm!klxqy$ldsskfs9$^t^ugx"


def _parse_bool(value: str, default: bool = False) -> bool:
    """
    Parse a boolean-like environment value.

    Args:
        value: Raw string from environment.
        default: Fallback if value is empty.

    Returns:
        Parsed boolean.
    """
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_allowed_hosts(value: str) -> List[str]:
    """
    Parse comma-separated ALLOWED_HOSTS.

    Args:
        value: Comma-separated hostnames/IPs.

    Returns:
        List of allowed hosts, excluding empties.
    """
    if not value:
        return ["localhost", "127.0.0.1"]
    return [h.strip() for h in value.split(",") if h.strip()]


def _db_from_database_url(database_url: str) -> Optional[dict]:
    """
    Convert a DATABASE_URL string into a Django DATABASES 'default' dict.

    Supported:
      - postgres://user:pass@host:port/dbname
      - postgresql://user:pass@host:port/dbname

    Args:
        database_url: Database URL.

    Returns:
        Django DB config dict if parseable; otherwise None.
    """
    if not database_url:
        return None

    parsed = urlparse(database_url)
    if parsed.scheme not in {"postgres", "postgresql"}:
        return None

    # urlparse includes leading "/" in path. Example: "/neobank"
    db_name = parsed.path.lstrip("/")
    if not db_name:
        return None

    return {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": db_name,
        "USER": parsed.username or "",
        "PASSWORD": parsed.password or "",
        "HOST": parsed.hostname or "db",
        "PORT": str(parsed.port or 5432),
        "CONN_MAX_AGE": 60,  # small but useful; reduces connection churn
    }


SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-change-me")
DEBUG = _parse_bool(os.getenv("DEBUG", "True"), default=True)
ALLOWED_HOSTS: List[str] = _parse_allowed_hosts(os.getenv("ALLOWED_HOSTS", ""))

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "dashboard",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "neobank_site.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "neobank_site.wsgi.application"

# Prefer DATABASE_URL if provided (best for Docker/CI), otherwise fall back to discrete
# POSTGRES_* vars.
_database_url = os.getenv("DATABASE_URL", "").strip()
_db_from_url = _db_from_database_url(_database_url)

if _db_from_url is not None:
    DATABASES = {"default": _db_from_url}
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": os.getenv("POSTGRES_DB", "neobank"),
            "USER": os.getenv("POSTGRES_USER", "neobank"),
            "PASSWORD": os.getenv("POSTGRES_PASSWORD", "neobank_password"),
            "HOST": os.getenv("POSTGRES_HOST", "db"),
            "PORT": os.getenv("POSTGRES_PORT", "5432"),
            "CONN_MAX_AGE": 60,
        }
    }

LANGUAGE_CODE = "en-gb"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
