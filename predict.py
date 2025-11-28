"""
Inference script for the Heart Disease Prediction project.

This script loads a trained model and makes predictions on new data.
"""

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make predictions using trained model')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained model file (.pkl)')
    parser.add_argument('--preprocessor-path', type=str, required=True,
                        help='Path to the preprocessor file (.pkl)')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to CSV file with data to predict')
    parser.add_argument('--interactive', action='store_true',
                        help='Enter interactive mode to input a single patient data')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save predictions CSV (optional)')
    return parser.parse_args()


def load_model_and_preprocessor(model_path: str, preprocessor_path: str):
    """Load the trained model and preprocessor."""
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    print(f"Loading preprocessor from: {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor


def interactive_input():
    """Get patient data from user input interactively."""
    print("\n" + "="*80)
    print("Enter patient data (press Enter to skip optional fields):")
    print("="*80)
    
    data = {}
    
    data['age'] = int(input("Age: "))
    data['sex'] = input("Sex (Male/Female): ")
    data['cp'] = input("Chest Pain Type (typical angina/atypical angina/non-anginal/asymptomatic): ")
    data['trestbps'] = float(input("Resting Blood Pressure (mm Hg): "))
    data['chol'] = float(input("Serum Cholesterol (mg/dl): "))
    data['fbs'] = input("Fasting Blood Sugar > 120 mg/dl (True/False): ").lower() == 'true'
    data['restecg'] = input("Resting ECG (normal/lv hypertrophy/st-t abnormality): ")
    data['thalch'] = float(input("Maximum Heart Rate Achieved: "))
    data['exang'] = input("Exercise Induced Angina (True/False): ").lower() == 'true'
    data['oldpeak'] = float(input("ST Depression Induced by Exercise: "))
    data['slope'] = input("Slope of Peak Exercise ST Segment (upsloping/flat/downsloping): ")
    data['ca'] = float(input("Number of Major Vessels (0-3): "))
    data['thal'] = input("Thalassemia (normal/fixed defect/reversable defect): ")
    
    df = pd.DataFrame([data])
    return df


def predict(model, preprocessor, data: pd.DataFrame):
    """Make predictions on the input data."""
    data_processed = preprocessor.transform(data)
    predictions = model.predict(data_processed)
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(data_processed)
        return predictions, probabilities
    
    return predictions, None


def interpret_prediction(prediction: int):
    """Convert prediction to human-readable format."""
    interpretations = {
        0: "No heart disease",
        1: "Heart disease - Severity Level 1",
        2: "Heart disease - Severity Level 2",
        3: "Heart disease - Severity Level 3",
        4: "Heart disease - Severity Level 4"
    }
    return interpretations.get(prediction, f"Unknown (prediction: {prediction})")


def main():
    """Main inference pipeline."""
    args = parse_args()
    
    print("="*80)
    print("Heart Disease Prediction - Inference")
    print("="*80)
    
    model, preprocessor = load_model_and_preprocessor(args.model_path, args.preprocessor_path)
    
    if args.interactive:
        data = interactive_input()
        predictions, probabilities = predict(model, preprocessor, data)
        
        print("\n" + "="*80)
        print("Prediction Result:")
        print("="*80)
        print(f"Prediction: {interpret_prediction(predictions[0])}")
        
        if probabilities is not None:
            print("\nPrediction Probabilities:")
            for i, prob in enumerate(probabilities[0]):
                print(f"  Class {i}: {prob:.4f} ({prob*100:.2f}%)")
        
    elif args.data_path:
        print(f"\nLoading data from: {args.data_path}")
        data = pd.read_csv(args.data_path)
        
        if 'id' in data.columns:
            data = data.drop(columns=['id'])
        if 'dataset' in data.columns:
            data = data.drop(columns=['dataset'])
        if 'num' in data.columns:
            data = data.drop(columns=['num'])
        
        print(f"Data shape: {data.shape}")
        
        predictions, probabilities = predict(model, preprocessor, data)
        
        results_df = data.copy()
        results_df['prediction'] = predictions
        results_df['prediction_label'] = [interpret_prediction(p) for p in predictions]
        
        if probabilities is not None:
            for i in range(probabilities.shape[1]):
                results_df[f'probability_class_{i}'] = probabilities[:, i]
        
        print("\nPredictions:")
        print(results_df[['prediction', 'prediction_label']].head(10))
        
        if args.output:
            results_df.to_csv(args.output, index=False)
            print(f"\nPredictions saved to: {args.output}")
        
    else:
        print("\nError: Either --data-path or --interactive must be specified")
        return
    
    print("\n" + "="*80)
    print("Inference completed successfully!")
    print("="*80)


if __name__ == '__main__':
    main()
