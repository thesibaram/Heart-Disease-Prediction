"""Data preprocessing utilities for the Heart Disease Prediction project."""

from typing import Dict, List, Tuple

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_features_target(
    df: pd.DataFrame,
    target_column: str = "num",
    drop_columns: List[str] | None = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and target while dropping unwanted columns."""
    if drop_columns is None:
        drop_columns = ["id", "dataset"]

    X = df.drop(columns=[col for col in drop_columns if col in df.columns] + [target_column])
    y = df[target_column]
    return X, y


def build_preprocessing_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing pipeline using ColumnTransformer."""
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'bool']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets with stratification."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
