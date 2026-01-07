# Generated manually for OpsCategoryEdit.

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="OpsCategoryEdit",
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
                ("timestamp", models.CharField(max_length=64)),
                ("customer_id", models.CharField(max_length=64)),
                ("amount", models.CharField(max_length=32)),
                ("merchant", models.CharField(max_length=128)),
                ("description", models.TextField(blank=True)),
                ("predicted_category", models.CharField(blank=True, max_length=128)),
                ("new_category", models.CharField(max_length=128)),
                ("edited_at", models.DateTimeField()),
            ],
        ),
    ]
