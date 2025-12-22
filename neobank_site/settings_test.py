"""
Test settings for neobank_site.

Goal:
- Tests must not require external services (no local Postgres, no Docker dependency).
- Reuse the full base settings (INSTALLED_APPS, MIDDLEWARE, ROOT_URLCONF, templates, etc.).
- Override DATABASES to SQLite for deterministic pytest runs.

pytest.ini points DJANGO_SETTINGS_MODULE at this module.
"""
from __future__ import annotations

from pathlib import Path

from . import settings as base


# Copy every UPPERCASE setting from the base settings module.
for _name in dir(base):
    if _name.isupper():
        globals()[_name] = getattr(base, _name)


# Ensure BASE_DIR exists even if base settings differ slightly.
BASE_DIR = globals().get("BASE_DIR")
if BASE_DIR is None:
    BASE_DIR = Path(__file__).resolve().parent.parent
    globals()["BASE_DIR"] = BASE_DIR


DEBUG = False

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
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
