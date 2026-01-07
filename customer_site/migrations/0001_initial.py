# Squashed initial migration for customer_site.

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="CustomerTransaction",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("row_id", models.CharField(max_length=64, unique=True)),
                ("customer_id", models.CharField(db_index=True, max_length=64)),
                ("timestamp", models.CharField(max_length=64)),
                ("amount", models.CharField(max_length=32)),
                ("merchant", models.CharField(max_length=128)),
                ("description", models.TextField(blank=True)),
                ("category", models.CharField(blank=True, max_length=128)),
                ("predicted_category", models.CharField(blank=True, max_length=128)),
                ("category_source", models.CharField(default="model", max_length=16)),
                ("fraud_risk", models.FloatField(blank=True, null=True)),
                ("flagged", models.BooleanField(default=False)),
                ("scored_at", models.DateTimeField(db_index=True)),
            ],
            options={
                "indexes": [
                    models.Index(
                        fields=["customer_id", "scored_at"],
                        name="customer_scored_idx",
                    )
                ],
            },
        ),
        migrations.CreateModel(
            name="CustomerFlag",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("row_id", models.CharField(max_length=64, unique=True)),
                ("customer_id", models.CharField(db_index=True, max_length=64)),
                ("timestamp", models.CharField(max_length=64)),
                ("amount", models.CharField(max_length=32)),
                ("merchant", models.CharField(max_length=128)),
                ("description", models.TextField(blank=True)),
                ("reason", models.CharField(blank=True, max_length=200)),
                ("flagged_at", models.DateTimeField()),
            ],
            options={
                "indexes": [
                    models.Index(
                        fields=["customer_id", "flagged_at"],
                        name="customer_flag_idx",
                    )
                ],
            },
        ),
    ]
