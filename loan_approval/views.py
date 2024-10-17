import os

import joblib
from django.conf import settings

from .forms import LoanApplicationForm

# Визначення шляху до моделі
model_path = os.path.join(settings.BASE_DIR, 'loan_approval', 'ml_models', 'loan_approval_model.pkl')

# Завантаження моделі
model = joblib.load(model_path)


def loan_approval_view(request):
    if request.method == 'POST':
        form = LoanApplicationForm(request.POST)
        if form.is_valid():
            application = form.save(commit=False)
            # Підготовка даних для передбачення
            input_data = [
                application.age,
                application.income,
                application.loan_amount,
                application.gender,
                application.employment_type
            ]
            result = model.predict([input_data])[0]
            decision = 'Approved' if result == 1 else 'Rejected'
            return render(request, 'loans/result.html', {'decision': decision})
    else:
        form = LoanApplicationForm()
    return render(request, 'loans/loan_form.html', {'form': form})
from django.shortcuts import render


