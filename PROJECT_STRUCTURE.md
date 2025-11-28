# Heart Disease Prediction - Project Structure

This document provides a comprehensive overview of the project's file organization and architecture.

## Directory Structure

```
heart-disease-prediction/
│
├── .git/                               # Git version control
├── .venv/                              # Python virtual environment
├── .gitignore                          # Git ignore rules
│
├── README.md                           # Main project documentation
├── CONTRIBUTING.md                     # Contributing guidelines
├── LICENSE                             # MIT License
├── USAGE_GUIDE.md                      # Detailed usage instructions
├── requirements.txt                    # Python dependencies
│
├── main.py                             # Main training script
├── predict.py                          # Inference script
│
├── data/                               # Data directory
│   ├── raw/                            # Raw datasets (after download)
│   ├── processed/                      # Preprocessed data (optional)
│   └── README.md                       # Data documentation
│
├── src/                                # Source code package
│   ├── __init__.py                     # Package initializer
│   ├── data_loader.py                  # Data loading utilities
│   ├── preprocessing.py                # Data preprocessing pipeline
│   ├── models.py                       # Model training and evaluation
│   ├── visualization.py                # Plotting functions
│   └── utils.py                        # Helper utilities
│
├── outputs/                            # Training outputs
│   ├── plots/                          # Visualizations (PNG files)
│   │   ├── confusion_matrix_*.png
│   │   ├── feature_importance.png
│   │   ├── model_comparison.png
│   │   └── target_distribution.png
│   ├── models/                         # Trained models (PKL files)
│   │   ├── logistic_model.pkl
│   │   ├── random_forest_model.pkl
│   │   ├── svm_model.pkl
│   │   ├── knn_model.pkl
│   │   └── preprocessor.pkl
│   └── README.md                       # Outputs documentation
│
├── reports/                            # Generated reports
│   └── metrics_summary.json            # Model performance metrics
│
└── notebooks/                          # Jupyter notebooks
    └── Day_4_Heart_disease_prediction.ipynb  # Original analysis notebook
```

## Key Files Description

### Root Level Files

| File | Purpose |
|------|---------|
| `main.py` | Main training script that orchestrates the entire ML pipeline |
| `predict.py` | Inference script for making predictions on new data |
| `requirements.txt` | Lists all Python package dependencies with versions |
| `README.md` | Main project documentation and usage instructions |
| `USAGE_GUIDE.md` | Detailed guide on how to use the project |
| `CONTRIBUTING.md` | Guidelines for contributing to the project |
| `LICENSE` | MIT License for the project |
| `.gitignore` | Specifies files/directories to ignore in version control |

### Source Code (`src/`)

| Module | Purpose | Key Functions |
|--------|---------|--------------|
| `data_loader.py` | Data loading and downloading | `load_data()`, `download_heart_disease_data()`, `get_data_info()` |
| `preprocessing.py` | Data preprocessing pipeline | `split_features_target()`, `build_preprocessing_pipeline()`, `split_data()` |
| `models.py` | Model training and evaluation | `get_model()`, `train_model()`, `evaluate_model()`, `train_and_evaluate_models()` |
| `visualization.py` | Data and results visualization | `plot_confusion_matrix()`, `plot_feature_importance()`, `plot_model_comparison()` |
| `utils.py` | Helper utilities | `ensure_directory()`, `save_json()`, `seed_everything()` |

### Data Flow

```
1. Raw Data (Kaggle/CSV)
   ↓
2. data_loader.py → Load and inspect data
   ↓
3. preprocessing.py → Clean, split, and transform
   ↓
4. models.py → Train and evaluate models
   ↓
5. visualization.py → Create plots
   ↓
6. outputs/ → Save models, plots, and metrics
```

## Module Dependencies

```
main.py
  ├── src.data_loader
  ├── src.preprocessing
  ├── src.models
  ├── src.visualization
  └── src.utils

predict.py
  └── joblib (for loading saved models)

src.models
  └── scikit-learn

src.preprocessing
  └── scikit-learn

src.visualization
  ├── matplotlib
  └── seaborn

src.data_loader
  ├── pandas
  └── kagglehub
```

## Output Files

### After Training (`python main.py --save-models`)

1. **Visualizations** (`outputs/plots/`)
   - `target_distribution.png`: Shows class distribution in dataset
   - `model_comparison.png`: Bar chart comparing all models
   - `confusion_matrix_logistic.png`: Confusion matrix for best model
   - `feature_importance.png`: Top 10 important features from Random Forest

2. **Models** (`outputs/models/`)
   - `{model_name}_model.pkl`: Trained scikit-learn model objects
   - `preprocessor.pkl`: Fitted preprocessing pipeline

3. **Reports** (`reports/`)
   - `metrics_summary.json`: Performance metrics for all models

### After Prediction (`python predict.py ...`)

- `predictions.csv`: Contains original data plus predictions (if `--output` specified)

## Configuration

### Model Configuration (in `main.py`)

```python
models_config = {
    'logistic': {},
    'random_forest': {'n_estimators': 100, 'random_state': 42},
    'svm': {'random_state': 42, 'probability': True},
    'knn': {'n_neighbors': 5}
}
```

### Preprocessing Pipeline (in `src/preprocessing.py`)

```
Numerical features:
  → Median imputation
  → StandardScaler

Categorical features:
  → Mode imputation
  → OneHotEncoder
```

## File Sizes (Approximate)

- **Source code**: ~30 KB total
- **Trained models**: ~5-10 MB (all models combined)
- **Visualizations**: ~500 KB (all plots combined)
- **Documentation**: ~50 KB
- **Requirements**: ~1 KB
- **Dataset**: ~100 KB (CSV)

## Development Workflow

1. **Setup**: Install dependencies from `requirements.txt`
2. **Training**: Run `main.py` to train models
3. **Evaluation**: Review outputs in `outputs/` and `reports/`
4. **Inference**: Use `predict.py` with saved models
5. **Exploration**: Use Jupyter notebook for interactive analysis

## Adding New Features

### Adding a New Model

1. Add model configuration to `models_config` in `main.py`
2. Ensure model is supported in `src/models.py` `get_model()` function
3. Run training pipeline

### Adding New Preprocessing Steps

1. Modify `build_preprocessing_pipeline()` in `src/preprocessing.py`
2. Update documentation
3. Retrain models

### Adding New Visualizations

1. Add function to `src/visualization.py`
2. Call from `main.py` or use standalone
3. Save to `outputs/plots/`

## Testing Strategy

- **Manual testing**: Run `main.py` and `predict.py` with various parameters
- **Visual inspection**: Check plots in `outputs/plots/`
- **Metric validation**: Review `reports/metrics_summary.json`
- **Prediction validation**: Use `predict.py` with known data

## Best Practices

1. **Always save the preprocessor** alongside models
2. **Use consistent random seeds** for reproducibility
3. **Document changes** in commit messages
4. **Keep data separate** from code (don't commit large files)
5. **Version your models** if making significant changes

---

This structure follows industry-standard ML project organization and is designed for:
- Easy navigation
- Clear separation of concerns
- Scalability
- Maintainability
- Collaboration
