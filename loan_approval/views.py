from django.shortcuts import render
from .forms import LoanApprovalForm
import joblib
import numpy as np

# Завантаження моделі
model = joblib.load('data_analysis/loan_approval_model.pkl')


def loan_approval_view(request):
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
            ]
            purpose = form.cleaned_data['purpose']

            # Виконуємо кодування цільової змінної purpose (враховуючи OneHotEncoding)
            purpose_mapping = {
                'debt_consolidation': 0,
                'credit_card': 1,
                'home_improvement': 2,
                'major_purchase': 3,
                'small_business': 4,
                'other': 5
            }
            purpose_encoded = [0] * len(purpose_mapping)
            purpose_encoded[purpose_mapping[purpose]] = 1
            data.extend(purpose_encoded)

            # Прогноз за допомогою моделі
            prediction = model.predict([data])[0]
            result = 'Approved' if prediction == 0 else 'Not Approved'
    else:
        form = LoanApprovalForm()

    return render(request, 'loan_approval/loan_approval.html', {'form': form, 'result': result})
