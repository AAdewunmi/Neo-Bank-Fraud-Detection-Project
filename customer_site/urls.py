"""Customer site URL configuration."""

from django.urls import path
from customer_site import views

app_name = "customer"

urlpatterns = [
    path("login/", views.customer_login, name="login"),
    path("logout/", views.customer_logout, name="logout"),
    path("", views.dashboard, name="dashboard"),
    path("flag/", views.flag_transaction, name="flag"),
    path("export-flags/", views.export_flags, name="export_flags"),
]
