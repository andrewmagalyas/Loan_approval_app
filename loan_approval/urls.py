from django.urls import path
from . import views

urlpatterns = [
    path('', views.loan_approval_view, name='loan_approval'),
    path('register/', views.register, name='register'),
    path('login/', views.user_login, name='login'),
    path('home/', views.home, name='home'),
]

