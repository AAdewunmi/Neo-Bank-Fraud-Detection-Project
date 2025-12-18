"""
Forms for the dashboard app.

Week 1 intent:
- accept a CSV upload
- accept a fraud threshold value used by the scoring layer
"""
from __future__ import annotations

from django import forms


class UploadForm(forms.Form):
    """
    Upload a CSV file and provide a fraud threshold.

    Fields:
        csv_file: The input CSV file.
        threshold: Float in [0.0, 1.0] used to flag risky transactions.
    """

    csv_file = forms.FileField()
    threshold = forms.FloatField(min_value=0.0, max_value=1.0, initial=0.65)
