# Generated manually for CustomerFlag.

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("customer_site", "0002_add_scoring_fields"),
    ]

    operations = [
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
                    models.Index(fields=["customer_id", "flagged_at"], name="customer_flag_idx")
                ],
            },
        ),
    ]
