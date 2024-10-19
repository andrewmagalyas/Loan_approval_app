from django.urls import path
from . import views

urlpatterns = [
    path('', views.loan_approval_view, name='loan_approval'),
]

