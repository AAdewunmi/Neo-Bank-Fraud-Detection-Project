# Generated manually for CustomerTransaction scoring fields.

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("customer_site", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="customertransaction",
            name="fraud_risk",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="customertransaction",
            name="flagged",
            field=models.BooleanField(default=False),
        ),
    ]
