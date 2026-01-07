"""Models for durable Ops data."""

from __future__ import annotations

from django.db import models


class OpsCategoryEdit(models.Model):
    row_id = models.CharField(max_length=64, unique=True)
    timestamp = models.CharField(max_length=64)
    customer_id = models.CharField(max_length=64)
    amount = models.CharField(max_length=32)
    merchant = models.CharField(max_length=128)
    description = models.TextField(blank=True)
    predicted_category = models.CharField(max_length=128, blank=True)
    new_category = models.CharField(max_length=128)
    edited_at = models.DateTimeField()

    def __str__(self) -> str:
        return f"{self.row_id}:{self.new_category}"
