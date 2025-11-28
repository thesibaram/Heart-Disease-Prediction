"""
Visualization utilities for the Heart Disease Prediction project.
"""

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_plot_style():
    """Set the default plotting style for the project."""
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10


def plot_confusion_matrix(
    confusion_matrix,
    model_name: str,
    class_labels: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Plot and optionally save a confusion matrix heatmap.
    
    Args:
        confusion_matrix: Confusion matrix array
        model_name (str): Name of the model for the title
        class_labels (list): Labels for the classes
        save_path (str): Path to save the plot (optional)
    """
    if class_labels is None:
        class_labels = ['No Disease', 'Disease', 'Severity 2', 'Severity 3', 'Severity 4']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_feature_importance(
    feature_importance_df: pd.DataFrame,
    top_n: int = 10,
    save_path: Optional[str] = None
):
    """
    Plot and optionally save feature importance from tree-based models.
    
    Args:
        feature_importance_df (pd.DataFrame): DataFrame with 'Feature' and 'Importance' columns
        top_n (int): Number of top features to display
        save_path (str): Path to save the plot (optional)
    """
    top_features = feature_importance_df.head(top_n)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x='Importance',
        y='Feature',
        data=top_features,
        palette='rocket',
        hue='Feature',
        legend=False
    )
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to: {save_path}")
    
    plt.show()


def plot_target_distribution(
    y: pd.Series,
    save_path: Optional[str] = None
):
    """
    Plot the distribution of the target variable.
    
    Args:
        y (pd.Series): Target variable
        save_path (str): Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 6))
    counts = y.value_counts().sort_index()
    
    ax = sns.barplot(x=counts.index, y=counts.values, palette='viridis', hue=counts.index, legend=False)
    plt.xlabel('Heart Disease Severity', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.title('Distribution of Heart Disease Severity', fontsize=14, fontweight='bold')
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Target distribution plot saved to: {save_path}")
    
    plt.show()


def plot_model_comparison(
    results: dict,
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
    save_path: Optional[str] = None
):
    """
    Create a bar plot comparing different models across multiple metrics.
    
    Args:
        results (dict): Dictionary containing model results
        metrics (list): List of metrics to compare
        save_path (str): Path to save the plot (optional)
    """
    comparison_data = []
    
    for model_name, model_results in results.items():
        for metric in metrics:
            if metric in model_results:
                comparison_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Metric': metric.replace('_', ' ').title(),
                    'Score': model_results[metric]
                })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    plt.figure(figsize=(14, 6))
    sns.barplot(
        data=comparison_df,
        x='Metric',
        y='Score',
        hue='Model',
        palette='Set2'
    )
    plt.xlabel('Metric', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to: {save_path}")
    
    plt.show()


def plot_correlation_matrix(
    df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plot correlation matrix for numerical features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        save_path (str): Path to save the plot (optional)
    """
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    plt.figure(figsize=(12, 10))
    correlation_matrix = numeric_df.corr()
    
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={'label': 'Correlation'}
    )
    plt.title('Correlation Matrix of Numerical Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation matrix saved to: {save_path}")
    
    plt.show()
