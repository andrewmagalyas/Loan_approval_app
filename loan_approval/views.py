import joblib
import pandas as pd
from django import forms
from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from .forms import LoanApprovalForm

#@login_required
def home(request):
    """
    Відображає домашню сторінку для зареєстрованих користувачів.

    Parameters:
    request (HttpRequest): Об'єкт запиту.

    Returns:
    HttpResponse: Відповідь з рендерингом шаблону 'home.html'.
    """
    return render(request, 'home.html')


class UserRegistrationForm(forms.ModelForm):
    """
    Форма для реєстрації нового користувача, включаючи поле для пароля.

    Attributes:
    password (CharField): Поле для введення пароля.
    Meta: Вказує модель та поля, що використовуються у формі.
    """
    password = forms.CharField(widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ['username', 'password']


def register(request):
    """
    Обробляє реєстрацію користувача. Якщо форма валідна, користувач реєструється і аутентифікується.

    Parameters:
    request (HttpRequest): Об'єкт запиту.

    Returns:
    HttpResponse: Рендеринг форми реєстрації або редирект на сторінку схвалення позики.
    """
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            messages.success(request, 'Registration successful.')
            return redirect('loan_approval')  # Редирект на сторінку схвалення позики або головну
    else:
        form = UserRegistrationForm()
    return render(request, 'registration/register.html', {'form': form})


def user_login(request):
    """
    Обробляє вхід користувача. Якщо облікові дані валідні, користувач входить у систему.

    Parameters:
    request (HttpRequest): Об'єкт запиту.

    Returns:
    HttpResponse: Рендеринг сторінки входу або редирект на сторінку схвалення позики.
    """
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('loan_approval')  # Редирект на сторінку схвалення позики після входу
        else:
            return render(request, 'registration/login.html', {'error': 'Invalid credentials'})

    return render(request, 'registration/login.html')


# Завантаження моделі
model = joblib.load('data_analysis/loan_approval_model.pkl')


def loan_approval_view(request):
    """
    Відображає форму для схвалення позики і обробляє прогноз моделі на основі введених даних.

    Parameters:
    request (HttpRequest): Об'єкт запиту.

    Returns:
    HttpResponse: Рендеринг сторінки 'loan_approval.html' з результатом схвалення або не схвалення.
    """
    result = None
    if request.method == 'POST':
        form = LoanApprovalForm(request.POST)
        if form.is_valid():
            # Отримання даних з форми
            data = [
                form.cleaned_data['int_rate'],
                form.cleaned_data['installment'],
                form.cleaned_data['log_annual_inc'],
                form.cleaned_data['dti'],
                form.cleaned_data['fico'],
                form.cleaned_data['days_with_cr_line'],
                form.cleaned_data['revol_bal'],
                form.cleaned_data['revol_util'],
                form.cleaned_data['inq_last_6mths'],
                form.cleaned_data['delinq_2yrs'],
                form.cleaned_data['pub_rec'],
                form.cleaned_data['purpose']
            ]

            # Прогноз за допомогою моделі
            data_df = pd.DataFrame([data], columns=[
                'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico',
                'days.with.cr.line', 'revol.bal', 'revol.util',
                'inq.last.6mths', 'delinq.2yrs', 'pub.rec', 'purpose'
            ])

            prediction = model.predict(data_df)[0]
            result = 'Схвалено' if prediction == 0 else 'Не схвалено'
    else:
        form = LoanApprovalForm()

    return render(request, 'loan_approval/loan_approval.html', {'form': form, 'result': result})
