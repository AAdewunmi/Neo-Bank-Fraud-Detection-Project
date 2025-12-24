"""
Dashboard forms.

UploadForm:
- Upload a CSV and select fraud threshold.

FilterForm:
- Filter the last scored run stored in session (GET-based).
"""
from __future__ import annotations

from django import forms


class UploadForm(forms.Form):
    """
    Upload and scoring controls.

    Fields:
        csv_file: CSV upload input.
        threshold: Fraud threshold (0..1).
    """

    csv_file = forms.FileField(
        required=True,
        widget=forms.ClearableFileInput(attrs={"class": "form-control", "accept": ".csv"}),
    )
    threshold = forms.FloatField(
        required=True,
        min_value=0.0,
        max_value=1.0,
        initial=0.7,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "0.01"}),
        help_text="0 to 1. Higher reduces false positives.",
    )


class FilterForm(forms.Form):
    """
    Filters applied to session-backed scored rows (no re-score).

    Fields:
        flagged_only: show only flagged rows.
        customer_id: exact match.
        merchant: substring match.
        category: substring match.
        min_fraud_risk: minimum fraud risk inclusive.
    """

    flagged_only = forms.BooleanField(required=False)
    customer_id = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={"class": "form-control", "placeholder": "Exact customer id"}),
    )
    merchant = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={"class": "form-control", "placeholder": "Contains text"}),
    )
    category = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={"class": "form-control", "placeholder": "Contains text"}),
    )
    min_fraud_risk = forms.FloatField(
        required=False,
        min_value=0.0,
        max_value=1.0,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "0.01",
                                        "placeholder": "0.00"}),
    )
