"""
Model training and evaluation utilities for the Heart Disease Prediction project.
"""

from typing import Dict, List, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def get_model(model_name: str, **kwargs):
    """
    Factory function to get a model instance by name.
    
    Args:
        model_name (str): Name of the model ('logistic', 'random_forest', 'svm', 'knn')
        **kwargs: Additional parameters to pass to the model constructor
        
    Returns:
        sklearn model instance
    """
    models = {
        'logistic': LogisticRegression,
        'random_forest': RandomForestClassifier,
        'svm': SVC,
        'knn': KNeighborsClassifier
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not supported. Choose from {list(models.keys())}")
    
    default_params = {
        'logistic': {'max_iter': 1000, 'random_state': 42},
        'random_forest': {'n_estimators': 100, 'random_state': 42},
        'svm': {'random_state': 42},
        'knn': {'n_neighbors': 5}
    }
    
    params = {**default_params.get(model_name, {}), **kwargs}
    return models[model_name](**params)


def train_model(model, X_train, y_train):
    """
    Train a machine learning model.
    
    Args:
        model: sklearn model instance
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained model
    """
    print(f"Training {model.__class__.__name__}...")
    model.fit(X_train, y_train)
    print(f"{model.__class__.__name__} trained successfully.")
    return model


def evaluate_model(model, X_test, y_test, model_name: str = None) -> Dict:
    """
    Evaluate a trained model and return metrics.
    
    Args:
        model: Trained sklearn model
        X_test: Test features
        y_test: Test labels
        model_name (str): Name of the model for display purposes
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    if model_name is None:
        model_name = model.__class__.__name__
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred
    }
    
    print(f"\n{'='*60}")
    print(f"{model_name} Performance")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nClassification Report:\n{report}")
    
    return metrics


def train_and_evaluate_models(
    models_config: Dict[str, Dict],
    X_train,
    y_train,
    X_test,
    y_test
) -> Dict[str, Dict]:
    """
    Train and evaluate multiple models.
    
    Args:
        models_config (dict): Dictionary with model names as keys and config dicts as values
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Dictionary containing metrics for each model
    """
    results = {}
    
    for model_name, config in models_config.items():
        model = get_model(model_name, **config)
        trained_model = train_model(model, X_train, y_train)
        metrics = evaluate_model(trained_model, X_test, y_test, model_name)
        metrics['model'] = trained_model
        results[model_name] = metrics
    
    return results


def get_best_model(results: Dict[str, Dict], metric: str = 'f1_score') -> Tuple[str, Dict]:
    """
    Find the best performing model based on a specific metric.
    
    Args:
        results (dict): Results dictionary from train_and_evaluate_models
        metric (str): Metric to use for comparison
        
    Returns:
        tuple: (model_name, model_results) of the best performing model
    """
    best_model_name = max(results.keys(), key=lambda k: results[k][metric])
    print(f"\nBest model based on {metric}: {best_model_name}")
    print(f"{metric.capitalize()}: {results[best_model_name][metric]:.4f}")
    
    return best_model_name, results[best_model_name]


def get_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract feature importances from tree-based models.
    
    Args:
        model: Trained sklearn model (must have feature_importances_ attribute)
        feature_names (list): List of feature names
        
    Returns:
        pd.DataFrame: DataFrame with features and their importances, sorted
    """
    if not hasattr(model, 'feature_importances_'):
        raise AttributeError(f"{model.__class__.__name__} does not have feature importances")
    
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    return feature_importance_df
