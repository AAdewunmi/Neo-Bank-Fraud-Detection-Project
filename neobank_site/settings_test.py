"""
Test settings for neobank_site.

Goal:
- Tests must not require external services (no local Postgres, no Docker dependency).
- Use SQLite for pytest runs so CI is deterministic.

Usage:
- pytest.ini points DJANGO_SETTINGS_MODULE at this module.
"""
from __future__ import annotations

# from .settings import *  # noqa: F403
from .settings import BASE_DIR  # noqa: F401


DEBUG = False

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        # File-based SQLite is more reliable than :memory: across Django's connections.
        "NAME": str(BASE_DIR / "test.sqlite3"),
    }
}

PASSWORD_HASHERS = [
    "django.contrib.auth.hashers.MD5PasswordHasher",
]

EMAIL_BACKEND = "django.core.mail.backends.locmem.EmailBackend"

CACHES = {
    "default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"},
}
