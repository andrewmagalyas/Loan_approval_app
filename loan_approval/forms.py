from django import forms

class LoanApprovalForm(forms.Form):
    int_rate = forms.FloatField(
        label='Процентна ставка за позикою',
        widget=forms.TextInput(attrs={'placeholder': 'Наприклад, 5.5 для 5.5%'}),
    )
    installment = forms.FloatField(
        label='Місячний платіж <span style="color: red;">*</span>',
        widget=forms.TextInput(attrs={'placeholder': 'Наприклад, 250.75'}),
    )
    log_annual_inc = forms.FloatField(
        label='Логарифм річного доходу <span style="color: red;">*</span>',
        widget=forms.TextInput(attrs={'placeholder': 'Наприклад, 10.5'}),
    )
    dti = forms.FloatField(
        label='Відношення боргу до доходу <span style="color: red;">*</span>',
        widget=forms.TextInput(attrs={'placeholder': 'Наприклад, 30.5'}),
    )
    fico = forms.IntegerField(
        label='Кредитний рейтинг позичальника <span style="color: red;">*</span>',
        widget=forms.TextInput(attrs={'placeholder': 'Цілісне число, наприклад, 720'}),
    )
    days_with_cr_line = forms.FloatField(
        label='Дні з кредитною лінією',
        widget=forms.TextInput(attrs={'placeholder': 'Наприклад, 300'}),
    )
    revol_bal = forms.FloatField(
        label='Обертовий баланс',
        widget=forms.TextInput(attrs={'placeholder': 'Наприклад, 500.00'}),
    )
    revol_util = forms.FloatField(
        label='Використання обертового кредиту',
        widget=forms.TextInput(attrs={'placeholder': 'Наприклад, 50.0 для 50%'}),
    )
    inq_last_6mths = forms.IntegerField(
        label='Запити за останні 6 місяців <span style="color: red;">*</span>',
        widget=forms.TextInput(attrs={'placeholder': 'Ціле число, наприклад, 2'}),
    )
    delinq_2yrs = forms.IntegerField(
        label='Прострочки за останні 2 роки <span style="color: red;">*</span>',
        widget=forms.TextInput(attrs={'placeholder': 'Ціле число, наприклад, 1'}),
    )
    pub_rec = forms.IntegerField(
        label='Публічні записи',
        widget=forms.TextInput(attrs={'placeholder': 'Ціле число, наприклад, 0'}),
    )
    purpose = forms.ChoiceField(
        choices=[
            ('debt_consolidation', 'Debt Consolidation'),
            ('credit_card', 'Credit Card'),
            ('home_improvement', 'Home Improvement'),
            ('major_purchase', 'Major Purchase'),
            ('small_business', 'Small Business'),
            ('all_other', 'Other')
        ],
        label='Мета позики <span style="color: red;">*</span>',
    )
