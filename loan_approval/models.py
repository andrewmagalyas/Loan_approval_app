from django.db import models

class LoanApplication(models.Model):
    age = models.IntegerField()
    income = models.FloatField()
    loan_amount = models.FloatField()
    gender = models.CharField(max_length=10)
    employment_type = models.CharField(max_length=20)

    def __str__(self):
        return f"Application by {self.gender}, Age: {self.age}"

