import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
from data_analysis.visualization import (plot_correlation_matrix,
                           plot_feature_importance,
                           plot_numerical_distributions,
                           plot_numerical_vs_target)

# Завантаження даних
df = pd.read_csv('loan_data.csv')

# Аналіз даних
print("Перевірка колонок:")
print(df.columns)
print("\nПерші 5 записів:")
print(df.head())
print("\nІнформація про дані:")
print(df.info())
print("\nОпис даних:")
print(df.describe())

# Перевірка на пропущені значення
print("\nПеревірка на пропущені значення:")
print(df.isnull().sum())

# Розподіл цільової змінної
print("\nРозподіл цільової змінної 'not.fully.paid':")
print(df['not.fully.paid'].value_counts())

# Розділення на ознаки та цільову змінну
X = df.drop('not.fully.paid', axis=1)
y = df['not.fully.paid']

# Розділення на тренувальні та тестові дані
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Визначення числових та категоріальних ознак
numerical_features = ['int.rate', 'installment', 'log.annual.inc', 'dti', 'fico',
                      'days.with.cr.line', 'revol.bal', 'revol.util',
                      'inq.last.6mths', 'delinq.2yrs', 'pub.rec']
categorical_features = ['purpose']

# Налаштування Pipeline для обробки даних та моделі
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)

# Створення класифікаційного Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Налаштування GridSearchCV для підбору параметрів
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, 20]
}

# Налаштування GridSearchCV з 5-кратною перехресною валідацією
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Виведення найкращих параметрів і точності моделі
print(f"\nНайкращі параметри: {grid_search.best_params_}")
print(f"\nТочність на тестових даних: {grid_search.score(X_test, y_test)}")

# Збереження найкращої моделі
joblib.dump(grid_search.best_estimator_, 'loan_approval_model.pkl')
print("\nМодель збережено у 'loan_approval_model.pkl'")

# Отримання закодованих даних для аналізу кореляцій
X_preprocessed = preprocessor.fit_transform(X)
df_preprocessed = pd.DataFrame(X_preprocessed,
                               columns=numerical_features +
                                       list(preprocessor.named_transformers_['cat'].
                                            get_feature_names_out(categorical_features)))

# Додавання цільової змінної для кореляційного аналізу
df_preprocessed['not.fully.paid'] = y.values


# Важливість ознак
feature_importances = grid_search.best_estimator_.named_steps['classifier'].feature_importances_
feature_names = numerical_features + list(preprocessor.named_transformers_['cat'].
                                          get_feature_names_out(categorical_features))

# Створення DataFrame для важливості ознак
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Візуалізація важливості ознак
plot_correlation_matrix(df_preprocessed)
plot_feature_importance(importance_df)
plot_numerical_distributions(df, numerical_features)
plot_numerical_vs_target(df, numerical_features)
