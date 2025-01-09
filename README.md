## Loan Approval Web Application
This project is a Django-based web application that predicts loan approval outcomes using a machine learning model trained on financial data. It includes user authentication and a secure interface for submitting loan applications, with real-time predictions based on user inputs.

### Features
- User Authentication: Secure registration and login for users.
- Loan Prediction: Machine learning-based loan approval predictions using a trained Random Forest model.
- Data Analysis: Preprocessing, feature engineering, and optimization of the prediction model for accuracy.
- Visualization: Insights through feature importance analysis and data correlation visualizations.
 
### Usage
1. Register or log in to your account.
2. avigate to the loan application form.
3. Fill in the required financial details and submit the form.
4. View the loan approval prediction result.

### Machine Learning Model
The loan prediction model is built using a Random Forest Classifier, optimized via GridSearchCV. Features such as interest rates, credit scores, and income were used for predictions. The model achieves high accuracy on test data and provides interpretable results.
