# neobank_site/settings.py
"""
Django settings for Neo-Bank Fraud Detection Project.

Week 2 intent:
- Customer site at /
- Ops dashboard at /ops/ (internal)
- Auth pages at /accounts/ (login/logout)

Database intent:
- Dev and CI run on Postgres.
- DATABASE_URL is the preferred config for all environments.

Important:
- Users live in the active database.
- Use Postgres for dev, CI, and production.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict
from urllib.parse import unquote, urlparse

BASE_DIR = Path(__file__).resolve().parent.parent


# CHANGE: secrets and debug come from env first, safe fallback for local dev.
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-insecure-key-change-me")
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"

ALLOWED_HOSTS = os.environ.get(
    "DJANGO_ALLOWED_HOSTS",
    "localhost,127.0.0.1,ops.localhost,customer.localhost",
).split(",")

OPS_HOST = os.environ.get("OPS_HOST", "ops.localhost")
CUSTOMER_HOST = os.environ.get("CUSTOMER_HOST", "customer.localhost")
HOST_ROUTING_ENABLED = os.environ.get("HOST_ROUTING_ENABLED", "1") == "1"


INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "dashboard",
    "customer_site",
]

MIDDLEWARE = [
    "neobank_site.middleware.HostScopedCookieMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "neobank_site.middleware.HostRoutingMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "neobank_site.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    }
]

WSGI_APPLICATION = "neobank_site.wsgi.application"


def _database_from_url(database_url: str) -> Dict[str, Any]:
    """
    Parse DATABASE_URL into a Django DATABASES['default'] dict.

    Supports:
    - postgres://user:pass@host:port/dbname
    - postgresql://user:pass@host:port/dbname
    """
    parsed = urlparse(database_url)

    scheme = parsed.scheme.lower()
    if scheme in {"postgres", "postgresql"}:
        return {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": unquote(parsed.path.lstrip("/")),
            "USER": unquote(parsed.username or ""),
            "PASSWORD": unquote(parsed.password or ""),
            "HOST": parsed.hostname or "",
            "PORT": str(parsed.port or ""),
        }

    raise ValueError(f"Unsupported DATABASE_URL scheme: {parsed.scheme}")


# CHANGE: clean switching logic
# Priority order:
# 1) DATABASE_URL
# 2) Postgres env vars (local or Docker)
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

if DATABASE_URL:
    DATABASES = {"default": _database_from_url(DATABASE_URL)}
elif os.environ.get("POSTGRES_HOST") or os.environ.get("POSTGRES_DB"):
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": os.environ.get("POSTGRES_DB", "neobank"),
            "USER": os.environ.get("POSTGRES_USER", "neobank"),
            "PASSWORD": os.environ.get("POSTGRES_PASSWORD", "neobank"),
            "HOST": os.environ.get("POSTGRES_HOST", "localhost"),
            "PORT": os.environ.get("POSTGRES_PORT", "5432"),
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": "neobank",
            "USER": "neobank",
            "PASSWORD": "neobank",
            "HOST": "localhost",
            "PORT": "5432",
        }
    }


AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "UserAttributeSimilarityValidator"
        )
    },
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {
        "NAME": (
            "django.contrib.auth.password_validation.CommonPasswordValidator"
        )
    },
    {
        "NAME": (
            "django.contrib.auth.password_validation.NumericPasswordValidator"
        )
    },
]

LANGUAGE_CODE = "en-gb"
TIME_ZONE = "Europe/London"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"

# Static files configuration
# - Adds optional static directories without failing when they do not exist.
# - Optionally exposes the runtime-generated artefacts/ directory
# under /static/artefacts/ for local dev.
STATIC_ROOT = BASE_DIR / "staticfiles"

STATICFILES_DIRS = []

_static_dir = BASE_DIR / "static"
if _static_dir.exists():
    STATICFILES_DIRS.append(_static_dir)

_project_static_dir = BASE_DIR / "neobank_site" / "static"
if _project_static_dir.exists():
    STATICFILES_DIRS.append(_project_static_dir)

_artefacts_dir = BASE_DIR / "artefacts"
if _artefacts_dir.exists():
    STATICFILES_DIRS.append(("artefacts", _artefacts_dir))

# Default primary key field type
# https://docs.djangoproject.com/en/5.0/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# CHANGE: auth and ops access toggles
LOGIN_URL = os.environ.get("DJANGO_LOGIN_URL", "/accounts/login/")
LOGIN_REDIRECT_URL = os.environ.get("DJANGO_LOGIN_REDIRECT_URL", "/ops/")
LOGOUT_REDIRECT_URL = os.environ.get("DJANGO_LOGOUT_REDIRECT_URL", "/")

DASHBOARD_REQUIRE_LOGIN = os.environ.get("DASHBOARD_REQUIRE_LOGIN", "1") == "1"
DASHBOARD_REQUIRE_STAFF = os.environ.get("DASHBOARD_REQUIRE_STAFF", "1") == "1"

# Production toggles (Week 4).
# Keeps local development friendly while making deploy settings explicit.
if "DEBUG" in os.environ:
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"

if "ALLOWED_HOSTS" in os.environ:
    ALLOWED_HOSTS = [
        host.strip()
        for host in os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
        if host.strip()
    ]

if "STATIC_URL" in os.environ:
    STATIC_URL = os.getenv("STATIC_URL", "static/")

if "STATIC_ROOT" in os.environ:
    STATIC_ROOT = Path(os.getenv("STATIC_ROOT", str(BASE_DIR / "staticfiles")))
