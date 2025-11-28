"""
Main training script for the Heart Disease Prediction project.

This script loads data, preprocesses it, trains multiple models,
evaluates them, and saves results.
"""

import argparse
import os
from pathlib import Path

import joblib
import pandas as pd

from src.data_loader import get_data_info, load_data
from src.models import (get_best_model, get_feature_importance,
                        train_and_evaluate_models)
from src.preprocessing import build_preprocessing_pipeline, split_data, split_features_target
from src.utils import ensure_directory, save_json, seed_everything
from src.visualization import (plot_confusion_matrix, plot_feature_importance,
                                plot_model_comparison, plot_target_distribution,
                                set_plot_style)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train heart disease prediction models')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to the dataset CSV file (if None, downloads from Kaggle)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for testing (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility (default: 42)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Directory to save outputs (default: outputs)')
    parser.add_argument('--save-models', action='store_true',
                        help='Save trained models to disk')
    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()
    
    print("="*80)
    print("Heart Disease Prediction - Training Pipeline")
    print("="*80)
    
    seed_everything(args.random_state)
    set_plot_style()
    
    output_dir = Path(args.output_dir)
    plots_dir = ensure_directory(output_dir / 'plots')
    models_dir = ensure_directory(output_dir / 'models')
    reports_dir = ensure_directory('reports')
    
    print("\n[1/6] Loading data...")
    df = load_data(file_path=args.data_path, from_kaggle=(args.data_path is None))
    
    print("\n[2/6] Exploring data...")
    data_info = get_data_info(df)
    
    print("\n[3/6] Preprocessing data...")
    X, y = split_features_target(df, target_column='num', drop_columns=['id', 'dataset'])
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    plot_target_distribution(y, save_path=plots_dir / 'target_distribution.png')
    
    preprocessor = build_preprocessing_pipeline(X)
    
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    print("\nFitting preprocessing pipeline...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    feature_names = preprocessor.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    
    print(f"Processed features shape: {X_train_df.shape}")
    
    print("\n[4/6] Training models...")
    models_config = {
        'logistic': {},
        'random_forest': {'n_estimators': 100, 'random_state': args.random_state},
        'svm': {'random_state': args.random_state, 'probability': True},
        'knn': {'n_neighbors': 5}
    }
    
    results = train_and_evaluate_models(
        models_config,
        X_train_df,
        y_train,
        X_test_df,
        y_test
    )
    
    print("\n[5/6] Analyzing results...")
    best_model_name, best_model_results = get_best_model(results, metric='f1_score')
    
    plot_model_comparison(results, save_path=plots_dir / 'model_comparison.png')
    
    plot_confusion_matrix(
        best_model_results['confusion_matrix'],
        best_model_name.replace('_', ' ').title(),
        save_path=plots_dir / f'confusion_matrix_{best_model_name}.png'
    )
    
    if 'random_forest' in results:
        rf_model = results['random_forest']['model']
        feature_importance_df = get_feature_importance(rf_model, feature_names)
        print("\nTop 10 Important Features (Random Forest):")
        print(feature_importance_df.head(10))
        plot_feature_importance(
            feature_importance_df,
            top_n=10,
            save_path=plots_dir / 'feature_importance.png'
        )
    
    print("\n[6/6] Saving results...")
    metrics_summary = {}
    for model_name, model_results in results.items():
        metrics_summary[model_name] = {
            'accuracy': float(model_results['accuracy']),
            'precision': float(model_results['precision']),
            'recall': float(model_results['recall']),
            'f1_score': float(model_results['f1_score'])
        }
    
    save_json(metrics_summary, reports_dir / 'metrics_summary.json')
    
    if args.save_models:
        print("\nSaving trained models...")
        for model_name, model_results in results.items():
            model_path = models_dir / f'{model_name}_model.pkl'
            joblib.dump(model_results['model'], model_path)
            print(f"  {model_name} saved to {model_path}")
        
        preprocessor_path = models_dir / 'preprocessor.pkl'
        joblib.dump(preprocessor, preprocessor_path)
        print(f"  Preprocessor saved to {preprocessor_path}")
    
    print("\n" + "="*80)
    print("Training pipeline completed successfully!")
    print(f"Best model: {best_model_name}")
    print(f"F1-Score: {best_model_results['f1_score']:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
