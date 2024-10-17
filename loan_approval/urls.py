from django.urls import path
from .views import loan_approval_view

urlpatterns = [
    path('', loan_approval_view, name='loan_form'),
]

