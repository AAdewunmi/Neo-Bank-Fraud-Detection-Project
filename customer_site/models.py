"""Models for customer-facing durable data."""

from __future__ import annotations

from django.db import models


class CustomerTransaction(models.Model):
    row_id = models.CharField(max_length=64, unique=True)
    customer_id = models.CharField(max_length=64, db_index=True)
    timestamp = models.CharField(max_length=64)
    amount = models.CharField(max_length=32)
    merchant = models.CharField(max_length=128)
    description = models.TextField(blank=True)
    category = models.CharField(max_length=128, blank=True)
    predicted_category = models.CharField(max_length=128, blank=True)
    category_source = models.CharField(max_length=16, default="model")
    fraud_risk = models.FloatField(null=True, blank=True)
    flagged = models.BooleanField(default=False)
    scored_at = models.DateTimeField(db_index=True)

    class Meta:
        indexes = [
            models.Index(fields=["customer_id", "scored_at"], name="customer_scored_idx"),
        ]

    def __str__(self) -> str:
        return f"{self.customer_id}:{self.row_id}"


class CustomerFlag(models.Model):
    row_id = models.CharField(max_length=64, unique=True)
    customer_id = models.CharField(max_length=64, db_index=True)
    timestamp = models.CharField(max_length=64)
    amount = models.CharField(max_length=32)
    merchant = models.CharField(max_length=128)
    description = models.TextField(blank=True)
    reason = models.CharField(max_length=200, blank=True)
    flagged_at = models.DateTimeField()

    class Meta:
        indexes = [
            models.Index(fields=["customer_id", "flagged_at"], name="customer_flag_idx"),
        ]

    def __str__(self) -> str:
        return f"{self.customer_id}:{self.row_id}"
