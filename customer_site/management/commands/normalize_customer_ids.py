"""Normalize stored customer_id values for customer transactions."""

from __future__ import annotations

from django.core.management.base import BaseCommand

from customer_site.models import CustomerTransaction
from customer_site.services import normalize_customer_id


class Command(BaseCommand):
    help = "Normalize CustomerTransaction.customer_id values in place."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--batch-size",
            type=int,
            default=500,
            help="Number of rows to update per batch.",
        )

    def handle(self, *args, **options) -> None:
        batch_size = max(1, int(options["batch_size"]))
        queryset = CustomerTransaction.objects.all().only("id", "customer_id")
        total = queryset.count()
        updated = 0
        buffer = []

        for row in queryset.iterator(chunk_size=batch_size):
            normalized = normalize_customer_id(row.customer_id)
            if normalized != row.customer_id:
                row.customer_id = normalized
                buffer.append(row)

            if len(buffer) >= batch_size:
                CustomerTransaction.objects.bulk_update(buffer, ["customer_id"])
                updated += len(buffer)
                buffer = []

        if buffer:
            CustomerTransaction.objects.bulk_update(buffer, ["customer_id"])
            updated += len(buffer)

        self.stdout.write(f"Normalized customer_id for {updated} of {total} rows.")
