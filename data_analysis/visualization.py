import os
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('visualizations', exist_ok=True)

def plot_correlation_matrix(df_preprocessed):
    plt.figure(figsize=(10, 6))
    corr_matrix = df_preprocessed.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Матриця кореляцій')
    plt.tight_layout()
    plt.savefig('visualizations/correlation_matrix.png')  # Збереження у файл
    plt.close()  # Закриваємо, щоб звільнити пам'ять

def plot_feature_importance(importance_df):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Важливість ознак')
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png')  # Збереження у файл
    plt.close()

def plot_numerical_distributions(df, numerical_features):
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(numerical_features, 1):
        plt.subplot(3, 4, i)
        sns.histplot(df[col], kde=True)
        plt.title(f'Розподіл {col}')
        plt.tight_layout()
    plt.savefig('visualizations/numerical_distributions.png')  # Збереження у файл
    plt.close()

def plot_numerical_vs_target(df, numerical_features):
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(numerical_features, 1):
        plt.subplot(3, 4, i)
        sns.boxplot(x=df['not.fully.paid'], y=df[col])
        plt.title(f'{col} vs not.fully.paid')
        plt.tight_layout()
    plt.savefig('visualizations/numerical_vs_target.png')  # Збереження у файл
    plt.close()
