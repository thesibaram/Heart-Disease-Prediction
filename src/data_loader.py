"""
Data loading utilities for the Heart Disease Prediction project.
"""

import os
import shutil
from pathlib import Path

import pandas as pd
import kagglehub


def download_heart_disease_data(dataset_name="redwankarimsony/heart-disease-data", 
                                 force_download=False):
    """
    Download the heart disease dataset from Kaggle using kagglehub.
    
    Args:
        dataset_name (str): Kaggle dataset identifier
        force_download (bool): Whether to force re-download if already cached
        
    Returns:
        str: Path to the downloaded dataset directory
    """
    print(f"Downloading dataset: {dataset_name}...")
    path = kagglehub.dataset_download(dataset_name)
    print(f"Dataset downloaded successfully to: {path}")
    return path


def load_data(file_path=None, from_kaggle=True):
    """
    Load the heart disease dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file. If None, downloads from Kaggle
        from_kaggle (bool): Whether to download from Kaggle if file_path is not specified
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if file_path is None and from_kaggle:
        path = download_heart_disease_data()
        kaggle_file = Path(path) / 'heart_disease_uci.csv'
        local_raw_path = Path('data') / 'raw' / 'heart_disease_uci.csv'
        local_raw_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(kaggle_file, local_raw_path)
        print(f"Dataset copied to: {local_raw_path}")
        file_path = local_raw_path
    elif file_path is None:
        file_path = Path('data') / 'raw' / 'heart_disease_uci.csv'
    else:
        file_path = Path(file_path)
    
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


def get_data_info(df):
    """
    Display basic information about the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Dictionary containing dataset statistics
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.to_dict(),
        'target_distribution': df['num'].value_counts().to_dict() if 'num' in df.columns else None
    }
    
    print("Dataset Information:")
    print(f"Shape: {info['shape']}")
    print(f"\nColumns: {info['columns']}")
    print(f"\nMissing Values:\n{pd.Series(info['missing_values'])}")
    if info['target_distribution']:
        print(f"\nTarget Distribution:\n{pd.Series(info['target_distribution']).sort_index()}")
    
    return info
