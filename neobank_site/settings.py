"""
Django settings for neobank_site.

Week 1: minimal configuration to support routing + tests + templates,
using PostgreSQL via Docker.

Key behavior:
- Inside Docker Compose: POSTGRES_HOST defaults to "db"
- Outside Docker (local dev): POSTGRES_HOST defaults to "localhost"

This avoids the common issue where running manage.py locally fails because
"db" only resolves inside the Docker network.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env if present (local dev convenience). In Docker, env vars are injected.
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)


def _parse_bool(value: str | None, default: bool = False) -> bool:
    """
    Parse a boolean-like environment value.

    Args:
        value: Raw string from environment.
        default: Fallback if value is None.

    Returns:
        Parsed boolean.
    """
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_allowed_hosts(value: str | None) -> List[str]:
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


def _db_from_database_url(database_url: str, running_in_docker: bool) -> Optional[dict]:
    """
    Convert a DATABASE_URL string into a Django DATABASES 'default' dict.

    Supported:
      - postgres://user:pass@host:port/dbname
      - postgresql://user:pass@host:port/dbname

    Note:
      If running locally and the URL host is "db", we rewrite it to "localhost"
      to prevent the "could not translate host name db" failure.

    Args:
        database_url: Database URL.
        running_in_docker: True if running inside Docker.

    Returns:
        Django DB config dict if parseable; otherwise None.
    """
    if not database_url:
        return None

    parsed = urlparse(database_url)
    if parsed.scheme not in {"postgres", "postgresql"}:
        return None

    db_name = parsed.path.lstrip("/")
    if not db_name:
        return None

    host = parsed.hostname or ""
    if not running_in_docker and host == "db":
        host = "localhost"

    return {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": db_name,
        "USER": parsed.username or "",
        "PASSWORD": parsed.password or "",
        "HOST": host,
        "PORT": str(parsed.port or 5432),
        "CONN_MAX_AGE": 60,
    }


SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-change-me")
DEBUG = _parse_bool(os.getenv("DEBUG"), default=True)
ALLOWED_HOSTS: List[str] = _parse_allowed_hosts(os.getenv("ALLOWED_HOSTS"))

# Set this in docker-compose for clarity
RUNNING_IN_DOCKER = _parse_bool(os.getenv("RUNNING_IN_DOCKER"), default=False)

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

# Prefer DATABASE_URL (Docker/CI friendly); fall back to POSTGRES_* vars.
_database_url = os.getenv("DATABASE_URL", "").strip()
_db_from_url = _db_from_database_url(_database_url, running_in_docker=RUNNING_IN_DOCKER)

_default_host = "db" if RUNNING_IN_DOCKER else "localhost"

if _db_from_url is not None:
    DATABASES = {"default": _db_from_url}
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": os.getenv("POSTGRES_DB", "neobank"),
            "USER": os.getenv("POSTGRES_USER", "neobank"),
            "PASSWORD": os.getenv("POSTGRES_PASSWORD", "neobank_password"),
            "HOST": os.getenv("POSTGRES_HOST", _default_host),
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
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}
