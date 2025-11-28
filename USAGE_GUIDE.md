# Heart Disease Prediction - Usage Guide

This guide provides detailed instructions on how to use this project for training, evaluation, and prediction.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training Models](#training-models)
3. [Making Predictions](#making-predictions)
4. [Understanding Outputs](#understanding-outputs)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

```bash
python main.py --save-models
```

This will:
- Download the dataset from Kaggle
- Preprocess the data
- Train 4 different models
- Save visualizations and metrics
- Save trained models to disk

### 3. Make Predictions

```bash
python predict.py \
    --model-path outputs/models/logistic_model.pkl \
    --preprocessor-path outputs/models/preprocessor.pkl \
    --interactive
```

## Training Models

### Basic Training

The simplest way to train models:

```bash
python main.py
```

This uses default parameters and doesn't save models to disk.

### Save Trained Models

To save models for later inference:

```bash
python main.py --save-models
```

Models will be saved in `outputs/models/` as `.pkl` files.

### Custom Data Path

If you've already downloaded the dataset:

```bash
python main.py --data-path data/raw/heart_disease_uci.csv --save-models
```

### Adjust Train/Test Split

Change the test set size (default is 0.2 = 20%):

```bash
python main.py --test-size 0.25 --save-models
```

### Custom Output Directory

Save outputs to a different location:

```bash
python main.py --output-dir my_experiments/run1 --save-models
```

### Complete Example

```bash
python main.py \
    --data-path data/raw/heart_disease_uci.csv \
    --test-size 0.2 \
    --random-state 42 \
    --output-dir outputs \
    --save-models
```

## Making Predictions

### Interactive Mode (Single Patient)

Enter data for one patient interactively:

```bash
python predict.py \
    --model-path outputs/models/logistic_model.pkl \
    --preprocessor-path outputs/models/preprocessor.pkl \
    --interactive
```

You'll be prompted to enter values like:

```
Age: 55
Sex (Male/Female): Male
Chest Pain Type (typical angina/atypical angina/non-anginal/asymptomatic): asymptomatic
Resting Blood Pressure (mm Hg): 145
Serum Cholesterol (mg/dl): 250
...
```

### Batch Prediction (Multiple Patients)

Predict on a CSV file with multiple patient records:

```bash
python predict.py \
    --model-path outputs/models/logistic_model.pkl \
    --preprocessor-path outputs/models/preprocessor.pkl \
    --data-path data/new_patients.csv \
    --output predictions.csv
```

**CSV Format:**
Your input CSV should have these columns (without the 'id', 'dataset', and 'num' columns):
```
age,sex,cp,trestbps,chol,fbs,restecg,thalch,exang,oldpeak,slope,ca,thal
63,Male,typical angina,145,233,True,lv hypertrophy,150,False,2.3,downsloping,0.0,fixed defect
67,Female,asymptomatic,160,286,False,normal,108,True,1.5,flat,3.0,normal
```

### Using Different Models

You can use any of the trained models:

**Logistic Regression:**
```bash
python predict.py \
    --model-path outputs/models/logistic_model.pkl \
    --preprocessor-path outputs/models/preprocessor.pkl \
    --data-path data/new_patients.csv
```

**Random Forest:**
```bash
python predict.py \
    --model-path outputs/models/random_forest_model.pkl \
    --preprocessor-path outputs/models/preprocessor.pkl \
    --data-path data/new_patients.csv
```

**SVM:**
```bash
python predict.py \
    --model-path outputs/models/svm_model.pkl \
    --preprocessor-path outputs/models/preprocessor.pkl \
    --data-path data/new_patients.csv
```

**KNN:**
```bash
python predict.py \
    --model-path outputs/models/knn_model.pkl \
    --preprocessor-path outputs/models/preprocessor.pkl \
    --data-path data/new_patients.csv
```

## Understanding Outputs

### Training Outputs

After running `main.py`, you'll find:

#### 1. Console Output
- Dataset statistics
- Training progress for each model
- Performance metrics (accuracy, precision, recall, F1-score)
- Classification reports

#### 2. Visualizations (`outputs/plots/`)
- `target_distribution.png`: Distribution of heart disease severity in the dataset
- `model_comparison.png`: Bar chart comparing all models
- `confusion_matrix_[model].png`: Confusion matrix for the best model
- `feature_importance.png`: Top 10 important features (from Random Forest)

#### 3. Model Files (`outputs/models/`)
- `logistic_model.pkl`: Trained Logistic Regression model
- `random_forest_model.pkl`: Trained Random Forest model
- `svm_model.pkl`: Trained SVM model
- `knn_model.pkl`: Trained KNN model
- `preprocessor.pkl`: Data preprocessing pipeline

#### 4. Metrics Report (`reports/metrics_summary.json`)
JSON file containing performance metrics for all models:

```json
{
  "logistic": {
    "accuracy": 0.5707,
    "precision": 0.5318,
    "recall": 0.5707,
    "f1_score": 0.5468
  },
  ...
}
```

### Prediction Outputs

#### Interactive Mode
- Console output showing prediction and probabilities

#### Batch Mode
- Console output with sample predictions
- CSV file (if `--output` specified) with:
  - Original features
  - `prediction`: Numeric prediction (0-4)
  - `prediction_label`: Human-readable label
  - `probability_class_N`: Probability for each class (if model supports it)

### Interpreting Predictions

| Prediction | Meaning |
|------------|---------|
| 0 | No heart disease |
| 1 | Heart disease - Severity Level 1 |
| 2 | Heart disease - Severity Level 2 |
| 3 | Heart disease - Severity Level 3 |
| 4 | Heart disease - Severity Level 4 |

## Advanced Usage

### Using the Jupyter Notebook

For exploratory analysis and visualization:

```bash
jupyter notebook notebooks/Day_4_Heart_disease_prediction.ipynb
```

### Programmatic Usage

You can also import and use the modules in your own Python scripts:

```python
from src.data_loader import load_data
from src.preprocessing import split_features_target, split_data
from src.models import get_model, train_model, evaluate_model
from src.visualization import plot_confusion_matrix

# Load data
df = load_data(from_kaggle=True)

# Preprocess
X, y = split_features_target(df)
X_train, X_test, y_train, y_test = split_data(X, y)

# Train a model
model = get_model('logistic')
model = train_model(model, X_train, y_train)

# Evaluate
metrics = evaluate_model(model, X_test, y_test)

# Visualize
plot_confusion_matrix(metrics['confusion_matrix'], 'Logistic Regression')
```

### Customizing Models

Modify `main.py` to add custom model parameters:

```python
models_config = {
    'logistic': {'max_iter': 2000, 'C': 0.5},
    'random_forest': {'n_estimators': 200, 'max_depth': 10},
    'svm': {'kernel': 'rbf', 'C': 1.0},
    'knn': {'n_neighbors': 7, 'weights': 'distance'}
}
```

## Troubleshooting

### Issue: Kaggle dataset download fails

**Solution:**
1. Set up Kaggle API credentials (see Installation guide)
2. Or manually download the dataset and use `--data-path`

### Issue: Missing dependencies

**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

### Issue: Model file not found

**Solution:**
Make sure you've trained and saved models first:
```bash
python main.py --save-models
```

### Issue: Memory errors during training

**Solution:**
- Reduce the dataset size
- Use a machine with more RAM
- Train models one at a time instead of all together

### Issue: Import errors

**Solution:**
Make sure you're running Python from the project root:
```bash
cd /path/to/heart-disease-prediction
python main.py
```

### Issue: Plots not showing

**Solution:**
- For headless environments, plots are saved to disk automatically
- Use `plt.show()` if running in Jupyter
- Check `outputs/plots/` for saved images

## Getting Help

If you encounter issues not covered here:

1. Check the main [README.md](README.md)
2. Review the code documentation in `src/` modules
3. Open an issue on GitHub with:
   - Your Python version
   - Error messages
   - Steps to reproduce

## Tips for Best Results

1. **Always use the same preprocessor** for training and prediction
2. **Save your models** after training to avoid retraining
3. **Check data quality** - ensure input CSV matches expected format
4. **Use the best model** based on F1-score for predictions
5. **Monitor class imbalance** - some severity levels have few samples
6. **Experiment with parameters** - tune hyperparameters for better performance

---

Happy predicting! 🫀
