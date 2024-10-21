import joblib
import pandas as pd
from django.shortcuts import render

from .forms import LoanApprovalForm

# Завантаження моделі
model = joblib.load('data_analysis/loan_approval_model.pkl')

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
                form.cleaned_data['purpose']
            ]

            # Прогноз за допомогою моделі
            # Оскільки дані подаються як список, перетворимо їх на DataFrame
            data_df = pd.DataFrame([data], columns=[
                'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico',
                'days.with.cr.line', 'revol.bal', 'revol.util',
                'inq.last.6mths', 'delinq.2yrs', 'pub.rec', 'purpose'
            ])

            prediction = model.predict(data_df)[0]
            result = 'Approved' if prediction == 0 else 'Not Approved'
    else:
        form = LoanApprovalForm()

    return render(request, 'loan_approval/loan_approval.html', {'form': form, 'result': result})

