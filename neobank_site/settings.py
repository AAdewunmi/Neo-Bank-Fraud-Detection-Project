# neobank_site/settings.py
"""
Django settings for Neo-Bank Fraud Detection Project.

Week 2 intent:
- Customer site at /
- Ops dashboard at /ops/ (internal)
- Auth pages at /accounts/ (login/logout)

Database intent:
- Dev runs on Postgres (Docker or local Postgres).
- CI runs on SQLite to keep GitHub Actions lightweight.
- Local commands without Postgres env vars fall back to SQLite, so pytest can
  run easily.

Important:
- Users live in the active database.
- Use Postgres for your real dev login user, SQLite is for CI and test runs.
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
    "localhost,127.0.0.1",
).split(",")


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
    - sqlite:///absolute/path/to/db.sqlite3
    - sqlite:///:memory:
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

    if scheme == "sqlite":
        # sqlite:////absolute/path.db or sqlite:///:memory:
        path = (parsed.netloc + parsed.path).strip()
        if path in {":memory:", "/:memory:"}:
            name = ":memory:"
        else:
            # urlparse gives leading slash for absolute paths.
            name = path
        return {"ENGINE": "django.db.backends.sqlite3", "NAME": name}

    raise ValueError(f"Unsupported DATABASE_URL scheme: {parsed.scheme}")


def _default_sqlite() -> Dict[str, Any]:
    """Local-friendly SQLite default."""
    return {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }


# CHANGE: clean switching logic
# Priority order:
# 1) DATABASE_URL (Docker dev uses this)
# 2) CI or GitHub Actions ->
# SQLite (keeps CI light)
# 3) Postgres env vars present -> Postgres (local dev)
# 4) Fallback -> SQLite (makes local pytest easy)
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
IS_CI = (
    os.environ.get("CI", "").lower() == "true"
    or os.environ.get("GITHUB_ACTIONS", "").lower() == "true"
)

if DATABASE_URL:
    DATABASES = {"default": _database_from_url(DATABASE_URL)}
elif IS_CI:
    DATABASES = {"default": _default_sqlite()}
elif os.environ.get("POSTGRES_HOST"):
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
    DATABASES = {"default": _default_sqlite()}


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

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# CHANGE: auth and ops access toggles
LOGIN_URL = os.environ.get("DJANGO_LOGIN_URL", "/accounts/login/")
LOGIN_REDIRECT_URL = os.environ.get("DJANGO_LOGIN_REDIRECT_URL", "/ops/")
LOGOUT_REDIRECT_URL = os.environ.get("DJANGO_LOGOUT_REDIRECT_URL", "/")

DASHBOARD_REQUIRE_LOGIN = os.environ.get("DASHBOARD_REQUIRE_LOGIN", "1") == "1"
DASHBOARD_REQUIRE_STAFF = os.environ.get("DASHBOARD_REQUIRE_STAFF", "1") == "1"
