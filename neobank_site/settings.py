"""
Django settings for Neo-Bank Fraud Detection Project.

This settings module supports two database modes:
1) SQLite for CI and lightweight local runs.
2) Postgres for Docker-based development.

Database selection is driven by environment variables so CI and development can
use different backends without code changes.
"""

import os
from pathlib import Path
from urllib.parse import parse_qs, urlparse

BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-insecure-secret-key")

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

RUNNING_IN_DOCKER = os.getenv("RUNNING_IN_DOCKER", "0") == "1"


def _sqlite_database_config(base_dir: Path) -> dict:
    """
    Build a SQLite database config.

    Args:
        base_dir: Project base directory used to place db.sqlite3.

    Returns:
        A Django DATABASES['default'] config dict for SQLite.
    """
    return {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": base_dir / "db.sqlite3",
    }


def _postgres_database_config_from_env() -> dict:
    """
    Build a Postgres database config from discrete environment variables.

    Expected variables:
        POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT

    Returns:
        A Django DATABASES['default'] config dict for Postgres.
    """
    # CHANGED: Do not default POSTGRES_HOST to localhost.
    # Absence of POSTGRES_HOST means "no Postgres configured" and we fall back to SQLite.
    db_name = os.getenv("POSTGRES_DB", "neobank")
    db_user = os.getenv("POSTGRES_USER", "neobank")
    db_password = os.getenv("POSTGRES_PASSWORD", "neobank_password")
    db_host = os.getenv("POSTGRES_HOST", "")
    db_port = os.getenv("POSTGRES_PORT", "5432")

    return {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": db_name,
        "USER": db_user,
        "PASSWORD": db_password,
        "HOST": db_host,
        "PORT": db_port,
        "CONN_MAX_AGE": int(os.getenv("DB_CONN_MAX_AGE", "60")),
    }


def _postgres_database_config_from_url(database_url: str) -> dict:
    """
    Build a Postgres database config from DATABASE_URL.

    Supports postgres and postgresql schemes, for example:
        postgres://user:pass@host:5432/dbname

    Args:
        database_url: The DATABASE_URL string.

    Returns:
        A Django DATABASES['default'] config dict for Postgres.

    Raises:
        ValueError: If the URL scheme is unsupported.
    """
    parsed = urlparse(database_url)
    scheme = (parsed.scheme or "").lower()

    if scheme not in {"postgres", "postgresql"}:
        raise ValueError(f"Unsupported DATABASE_URL scheme: {scheme}")

    options = parse_qs(parsed.query or "")
    # Keep options minimal and explicit. Extend later if you add sslmode or similar.
    django_options = {}
    if "sslmode" in options and options["sslmode"]:
        django_options["sslmode"] = options["sslmode"][0]

    return {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": (parsed.path or "").lstrip("/"),
        "USER": parsed.username or "",
        "PASSWORD": parsed.password or "",
        "HOST": parsed.hostname or "",
        "PORT": str(parsed.port or "5432"),
        "CONN_MAX_AGE": int(os.getenv("DB_CONN_MAX_AGE", "60")),
        "OPTIONS": django_options,
    }


def _select_database(base_dir: Path) -> dict:
    """
    Select the active database backend.

    Priority order:
    1) DJANGO_DB explicit override (sqlite or postgres)
    2) CI signals (GITHUB_ACTIONS or CI) default to SQLite
    3) DATABASE_URL for Postgres or SQLite
    4) Docker dev signal or explicit POSTGRES_HOST uses Postgres
    5) Fallback to SQLite

    Returns:
        A Django DATABASES['default'] config dict.
    """
    dj_db = (os.getenv("DJANGO_DB") or "").strip().lower()
    database_url = (os.getenv("DATABASE_URL") or "").strip()

    # CHANGED: Explicit override via DJANGO_DB.
    if dj_db in {"sqlite", "sqlite3"}:
        return _sqlite_database_config(base_dir)
    if dj_db in {"postgres", "postgresql"}:
        if database_url:
            return _postgres_database_config_from_url(database_url)
        return _postgres_database_config_from_env()

    # CHANGED: CI defaults to SQLite unless you explicitly set DJANGO_DB or DATABASE_URL.
    is_ci = (
        (os.getenv("GITHUB_ACTIONS") or "").lower() == "true"
        or (os.getenv("CI") or "").lower() == "true"
    )
    if is_ci:
        return _sqlite_database_config(base_dir)

    # DATABASE_URL support.
    if database_url:
        parsed = urlparse(database_url)
        scheme = (parsed.scheme or "").lower()
        if scheme in {"sqlite", "sqlite3"}:
            # Support sqlite:///absolute/path style URLs if you later want them.
            name = (parsed.path or "").lstrip("/") or str(base_dir / "db.sqlite3")
            return {"ENGINE": "django.db.backends.sqlite3", "NAME": f"/{name}" if database_url.startswith("sqlite:////") else name}
        if scheme in {"postgres", "postgresql"}:
            return _postgres_database_config_from_url(database_url)

    # CHANGED: Use POSTGRES_HOST as the explicit signal for a local Postgres setup.
    if RUNNING_IN_DOCKER or os.getenv("POSTGRES_HOST"):
        return _postgres_database_config_from_env()

    return _sqlite_database_config(base_dir)


DATABASES = {"default": _select_database(BASE_DIR)}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]
