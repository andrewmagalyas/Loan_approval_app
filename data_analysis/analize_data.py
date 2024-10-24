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
def analyze_data(df):
    """
    Виконує базовий аналіз даних, включаючи перевірку колонок,
    перші записи, інформацію про типи даних і статистичні показники.

    Parameters:
    df (pandas.DataFrame): Вхідний набір даних для аналізу.

    Returns:
    None
    """
    print("Перевірка колонок:")
    print(df.columns)
    print("\nПерші 5 записів:")
    print(df.head())
    print("\nІнформація про дані:")
    print(df.info())
    print("\nОпис даних:")
    print(df.describe())
    print("\nПеревірка на пропущені значення:")
    print(df.isnull().sum())
    print("\nРозподіл цільової змінної 'not.fully.paid':")
    print(df['not.fully.paid'].value_counts())


# Виклик функції для аналізу даних
analyze_data(df)


# Розділення на ознаки та цільову змінну
def split_data(df, target_column):
    """
    Розділяє дані на ознаки та цільову змінну.

    Parameters:
    df (pandas.DataFrame): Вхідний набір даних.
    target_column (str): Назва цільової колонки.

    Returns:
    X (pandas.DataFrame): Ознаки.
    y (pandas.Series): Цільова змінна.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y


X, y = split_data(df, 'not.fully.paid')

# Розділення на тренувальні та тестові дані
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Визначення числових та категоріальних ознак
numerical_features = ['int.rate', 'installment', 'log.annual.inc', 'dti', 'fico',
                      'days.with.cr.line', 'revol.bal', 'revol.util',
                      'inq.last.6mths', 'delinq.2yrs', 'pub.rec']
categorical_features = ['purpose']


# Налаштування Pipeline для обробки даних та моделі
def build_pipeline(numerical_features, categorical_features):
    """
    Створює pipeline для обробки числових і категоріальних ознак
    та класифікації за допомогою випадкового лісу.

    Parameters:
    numerical_features (list): Список числових ознак.
    categorical_features (list): Список категоріальних ознак.

    Returns:
    pipeline (sklearn.pipeline.Pipeline): Готовий pipeline для моделювання.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    return pipeline


pipeline = build_pipeline(numerical_features, categorical_features)


# Налаштування GridSearchCV для підбору параметрів
def tune_model(pipeline, param_grid, X_train, y_train):
    """
    Налаштовує модель за допомогою GridSearchCV.

    Parameters:
    pipeline (sklearn.pipeline.Pipeline): Pipeline для налаштування.
    param_grid (dict): Сітка параметрів для GridSearchCV.
    X_train (pandas.DataFrame): Тренувальні ознаки.
    y_train (pandas.Series): Тренувальна цільова змінна.

    Returns:
    grid_search (sklearn.model_selection.GridSearchCV): Налаштована модель.
    """
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search


param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, 20]
}

grid_search = tune_model(pipeline, param_grid, X_train, y_train)

# Виведення найкращих параметрів і точності моделі
print(f"\nНайкращі параметри: {grid_search.best_params_}")
print(f"\nТочність на тестових даних: {grid_search.score(X_test, y_test)}")


# Збереження найкращої моделі
def save_model(model, filename):
    """
    Зберігає модель у файл.

    Parameters:
    model (sklearn.base.BaseEstimator): Модель для збереження.
    filename (str): Назва файлу для збереження.

    Returns:
    None
    """
    joblib.dump(model, filename)
    print(f"\nМодель збережено у '{filename}'")


save_model(grid_search.best_estimator_, 'loan_approval_model.pkl')


# Обробка даних для аналізу кореляцій
def preprocess_for_analysis(pipeline, X, numerical_features, categorical_features, y):
    """
    Обробляє дані для аналізу кореляцій, використовуючи підготовлений ColumnTransformer.

    Parameters:
    pipeline (sklearn.pipeline.Pipeline): Pipeline, що містить підготовлений preprocessor.
    X (pandas.DataFrame): Вхідні ознаки.
    numerical_features (list): Числові ознаки.
    categorical_features (list): Категоріальні ознаки.
    y (pandas.Series): Цільова змінна.

    Returns:
    df_preprocessed (pandas.DataFrame): Оброблені дані з додаванням цільової змінної.
    """
    preprocessor = pipeline.named_steps['preprocessor']  # Отримання preprocessor з pipeline
    X_preprocessed = preprocessor.fit_transform(X)
    df_preprocessed = pd.DataFrame(X_preprocessed,
                                   columns=numerical_features +
                                           list(preprocessor.named_transformers_['cat'].
                                                get_feature_names_out(categorical_features)))
    df_preprocessed['not.fully.paid'] = y.values
    return df_preprocessed

# Виклик функції
df_preprocessed = preprocess_for_analysis(grid_search.best_estimator_, X, numerical_features, categorical_features, y)



# Важливість ознак
def compute_feature_importance(model, numerical_features, categorical_features):
    """
    Обчислює важливість ознак для моделі.

    Parameters:
    model (sklearn.ensemble.RandomForestClassifier): Модель для аналізу.
    numerical_features (list): Числові ознаки.
    categorical_features (list): Категоріальні ознаки.

    Returns:
    importance_df (pandas.DataFrame): Важливість ознак у вигляді DataFrame.
    """
    feature_importances = model.named_steps['classifier'].feature_importances_
    feature_names = numerical_features + list(model.named_steps['preprocessor'].
                                              named_transformers_['cat'].
                                              get_feature_names_out(categorical_features))
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    return importance_df.sort_values(by='Importance', ascending=False)


importance_df = compute_feature_importance(grid_search.best_estimator_, numerical_features, categorical_features)

# Візуалізація важливості ознак
plot_correlation_matrix(df_preprocessed)
plot_feature_importance(importance_df)
plot_numerical_distributions(df, numerical_features)
plot_numerical_vs_target(df, numerical_features)
