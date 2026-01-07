"""Customer site URL configuration."""

from django.urls import path
from customer_site import views

app_name = "customer"

urlpatterns = [
    path("", views.home, name="home"),
    path("flag/", views.flag_transaction, name="flag"),
]
