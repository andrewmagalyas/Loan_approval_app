import pandas as pd
import logging
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    logging.info("Перевірка колонок:")
    logging.info(df.columns)
    logging.info("\nПерші 5 записів:")
    logging.info(df.head())
    logging.info("\nІнформація про дані:")
    logging.info(df.info())
    logging.info("\nОпис даних:")
    logging.info(df.describe())
    logging.info("\nПеревірка на пропущені значення:")
    logging.info(df.isnull().sum())
    logging.info("\nРозподіл цільової змінної 'not.fully.paid':")
    logging.info(df['not.fully.paid'].value_counts())


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
    logging.info(f"\nМодель збережено у '{filename}'")


# Обробка даних для аналізу кореляцій
def preprocess_for_analysis(pipeline, X_train, X_test, numerical_features, categorical_features, y_train):
    """
    Обробляє тренувальні дані для аналізу кореляцій, використовуючи підготовлений ColumnTransformer.
    """
    preprocessor = pipeline.named_steps['preprocessor']  # Отримання preprocessor з pipeline

    # Використовуємо fit_transform для тренувальних даних
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    # Transform для тестових даних, щоб уникнути витоку
    X_test_preprocessed = preprocessor.transform(X_test)

    # Створюємо DataFrame з обробленими тренувальними даними та ознаками
    df_train_preprocessed = pd.DataFrame(X_train_preprocessed,
                                         columns=numerical_features +
                                                 list(preprocessor.named_transformers_['cat'].get_feature_names_out(
                                                     categorical_features)))
    df_train_preprocessed['not.fully.paid'] = y_train.values

    return df_train_preprocessed, X_test_preprocessed


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


if __name__ == '__main__':
    # Завантаження даних
    df = pd.read_csv('loan_data.csv')

    # Виклик функції для аналізу даних
    analyze_data(df)

    # Розділення на ознаки та цільову змінну
    X, y = split_data(df, 'not.fully.paid')

    # Розділення на тренувальні та тестові дані
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Визначення числових та категоріальних ознак
    numerical_features = ['int.rate', 'installment', 'log.annual.inc', 'dti', 'fico',
                          'days.with.cr.line', 'revol.bal', 'revol.util',
                          'inq.last.6mths', 'delinq.2yrs', 'pub.rec']
    categorical_features = ['purpose']

    # Створення pipeline
    pipeline = build_pipeline(numerical_features, categorical_features)

    # Налаштування GridSearchCV для підбору параметрів
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [5, 10, 20]
    }

    grid_search = tune_model(pipeline, param_grid, X_train, y_train)

    # Виведення найкращих параметрів і точності моделі
    logging.info(f"\nНайкращі параметри: {grid_search.best_params_}")
    logging.info(f"\nТочність на тестових даних: {grid_search.score(X_test, y_test)}")

    # Збереження моделі
    save_model(grid_search.best_estimator_, 'loan_approval_model.pkl')

    # Підготовка даних для аналізу кореляцій
    df_train_preprocessed, X_test_preprocessed = preprocess_for_analysis(
        grid_search.best_estimator_, X_train, X_test, numerical_features, categorical_features, y_train
    )

    # Обчислення важливості ознак
    importance_df = compute_feature_importance(grid_search.best_estimator_, numerical_features, categorical_features)

    # Візуалізація важливості ознак
    plot_correlation_matrix(df_train_preprocessed)
    plot_feature_importance(importance_df)
    plot_numerical_distributions(df, numerical_features)
    plot_numerical_vs_target(df, numerical_features)
