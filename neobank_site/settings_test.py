"""
Test settings for neobank_site.

Goal:
- Reuse the full base settings (INSTALLED_APPS, MIDDLEWARE, ROOT_URLCONF, templates, etc.).
- Keep test-friendly overrides (hashers, email backend) without changing DB engine.

pytest.ini points DJANGO_SETTINGS_MODULE at this module.
"""
from __future__ import annotations

from . import settings as base


# Copy every UPPERCASE setting from the base settings module.
for _name in dir(base):
    if _name.isupper():
        globals()[_name] = getattr(base, _name)


DEBUG = False
HOST_ROUTING_ENABLED = False
WHITENOISE_ENABLED = False
ALLOWED_HOSTS = [
    "localhost",
    "127.0.0.1",
    "ops.localhost",
    "customer.localhost",
]

PASSWORD_HASHERS = [
    "django.contrib.auth.hashers.MD5PasswordHasher",
]

EMAIL_BACKEND = "django.core.mail.backends.locmem.EmailBackend"

CACHES = {
    "default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"},
}
