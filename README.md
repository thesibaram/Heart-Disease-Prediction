# 🫀 Heart Disease Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A professional machine learning project that predicts the presence and severity of heart disease using clinical and medical attributes. This project demonstrates end-to-end ML pipeline development, from data preprocessing to model deployment.

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Tools and Libraries](#-tools-and-libraries)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Training Models](#training-models)
  - [Making Predictions](#making-predictions)
  - [Using Jupyter Notebook](#using-jupyter-notebook)
- [Evaluation Metrics](#-evaluation-metrics)
- [Results](#-results)
- [Future Improvements](#-future-improvements)
- [References](#-references)
- [License](#-license)

## 🎯 Problem Statement

Heart disease is one of the leading causes of death worldwide. Early detection and accurate prediction can significantly improve patient outcomes. This project aims to:

- **Predict** the presence of heart disease in patients based on medical attributes
- **Classify** the severity of heart disease into 5 categories (0 = no disease, 1-4 = increasing severity)
- **Compare** multiple machine learning algorithms to identify the best performing model
- **Provide** an easy-to-use inference pipeline for clinical decision support

## 📊 Dataset

The project uses the **UCI Heart Disease Dataset** available on Kaggle, which combines data from four medical institutions:

- Cleveland Clinic Foundation
- Hungarian Institute of Cardiology, Budapest
- V.A. Medical Center, Long Beach, CA
- University Hospital, Zurich, Switzerland

### Dataset Features

| Feature | Description | Type |
|---------|-------------|------|
| `age` | Age in years | Numerical |
| `sex` | Sex (Male/Female) | Categorical |
| `cp` | Chest pain type (typical angina, atypical angina, non-anginal, asymptomatic) | Categorical |
| `trestbps` | Resting blood pressure (mm Hg) | Numerical |
| `chol` | Serum cholesterol (mg/dl) | Numerical |
| `fbs` | Fasting blood sugar > 120 mg/dl | Boolean |
| `restecg` | Resting electrocardiographic results | Categorical |
| `thalch` | Maximum heart rate achieved | Numerical |
| `exang` | Exercise induced angina | Boolean |
| `oldpeak` | ST depression induced by exercise relative to rest | Numerical |
| `slope` | Slope of the peak exercise ST segment | Categorical |
| `ca` | Number of major vessels colored by fluoroscopy (0-3) | Numerical |
| `thal` | Thalassemia (normal, fixed defect, reversable defect) | Categorical |
| `num` | Target: Heart disease severity (0-4) | Categorical |

**Dataset Statistics:**
- Total samples: 920
- Features: 13 input features + 1 target
- Missing values: Handled via imputation

**Data Source:** [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)

## 🧠 Model Architecture

This project implements and compares **four supervised classification algorithms**:

### 1. Logistic Regression
- Linear model for multi-class classification
- Baseline model with interpretable coefficients
- Fast training and prediction

### 2. Random Forest Classifier
- Ensemble of 100 decision trees
- Provides feature importance analysis
- Robust to overfitting through bagging

### 3. Support Vector Machine (SVM)
- Kernel-based classification
- Effective in high-dimensional spaces
- Provides competitive performance with nonlinear decision boundaries

### 4. K-Nearest Neighbors (KNN)
- Instance-based learning (k=5)
- Non-parametric approach
- Good for capturing local patterns

### Preprocessing Pipeline

```
Raw Data → Drop ID/Dataset columns → Train/Test Split (80/20)
    ↓
Numerical Features → Median Imputation → StandardScaler
    ↓
Categorical Features → Mode Imputation → OneHotEncoder
    ↓
Combined Features → Model Training → Evaluation
```

## 🛠 Tools and Libraries

- **Python 3.12**: Core programming language
- **pandas 2.3.3**: Data manipulation and analysis
- **NumPy 2.3.5**: Numerical computing
- **scikit-learn 1.7.2**: Machine learning algorithms and preprocessing
- **matplotlib 3.10.7**: Data visualization
- **seaborn 0.13.2**: Statistical visualizations
- **kagglehub 0.3.13**: Dataset downloading from Kaggle
- **joblib 1.5.2**: Model serialization

## 📁 Project Structure

```
heart-disease-prediction/
│
├── data/                           # Data directory
│   ├── raw/                        # Raw datasets (downloaded)
│   ├── processed/                  # Preprocessed datasets
│   └── README.md                   # Data documentation
│
├── src/                            # Source code package
│   ├── __init__.py
│   ├── data_loader.py              # Data loading utilities
│   ├── preprocessing.py            # Data preprocessing pipeline
│   ├── models.py                   # Model training and evaluation
│   ├── visualization.py            # Plotting functions
│   └── utils.py                    # Helper utilities
│
├── outputs/                        # Output directory
│   ├── plots/                      # Generated visualizations
│   ├── models/                     # Saved model files (.pkl)
│   └── README.md                   # Outputs documentation
│
├── reports/                        # Generated reports
│   └── metrics_summary.json        # Model performance metrics
│
├── notebooks/                      # Jupyter notebooks
│   └── Day_4_Heart_disease_prediction.ipynb  # Original analysis notebook
│
├── main.py                         # Main training script
├── predict.py                      # Inference script
├── requirements.txt                # Python dependencies
├── USAGE_GUIDE.md                  # Detailed usage instructions
├── CONTRIBUTING.md                 # Contribution guidelines
├── LICENSE                         # Project license (MIT)
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Virtual environment tool

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Kaggle API (Optional)

To download data automatically from Kaggle:

1. Create a Kaggle account at [kaggle.com](https://www.kaggle.com/)
2. Go to Account → API → Create New API Token
3. Place the downloaded `kaggle.json` file in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)

## 💻 Usage

For an in-depth walkthrough, see the [Usage Guide](USAGE_GUIDE.md).

### Training Models

Train all models and save outputs:

```bash
python main.py --save-models
```

**Options:**

- `--data-path`: Path to local CSV file (if not using Kaggle download)
- `--test-size`: Test set proportion (default: 0.2)
- `--random-state`: Random seed for reproducibility (default: 42)
- `--output-dir`: Directory for outputs (default: outputs)
- `--save-models`: Flag to save trained models

**Example with custom parameters:**

```bash
python main.py --data-path data/raw/heart_disease_uci.csv --test-size 0.25 --save-models
```

**Output:**
- Model performance metrics printed to console
- Visualizations saved to `outputs/plots/`
- Trained models saved to `outputs/models/` (if `--save-models` flag used)
- Metrics JSON saved to `reports/metrics_summary.json`

### Making Predictions

#### Batch Prediction

Predict on a CSV file:

```bash
python predict.py \
    --model-path outputs/models/logistic_model.pkl \
    --preprocessor-path outputs/models/preprocessor.pkl \
    --data-path data/new_patients.csv \
    --output predictions.csv
```

#### Interactive Prediction

Enter patient data manually:

```bash
python predict.py \
    --model-path outputs/models/logistic_model.pkl \
    --preprocessor-path outputs/models/preprocessor.pkl \
    --interactive
```

You'll be prompted to enter values for each feature:

```
Enter patient data:
Age: 63
Sex (Male/Female): Male
Chest Pain Type: typical angina
...
```

### Using Jupyter Notebook

Explore the data and models interactively:

```bash
jupyter notebook notebooks/Day_4_Heart_disease_prediction.ipynb
```

Or use the cleaned notebook version for a step-by-step walkthrough.

## 📈 Evaluation Metrics

Models are evaluated using the following metrics:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Ratio of true positive predictions to all positive predictions
- **Recall (Sensitivity)**: Ratio of true positives to all actual positives
- **F1-Score**: Harmonic mean of precision and recall (primary metric)
- **Confusion Matrix**: Detailed breakdown of predictions vs actual values

## 🏆 Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** | **0.5707** | **0.5318** | **0.5707** | **0.5468** |
| Random Forest | 0.5652 | 0.5085 | 0.5652 | 0.5343 |
| Support Vector Machine | 0.5761 | 0.5002 | 0.5761 | 0.5352 |
| K-Nearest Neighbors | 0.5652 | 0.5005 | 0.5652 | 0.5300 |

**Best Model:** Logistic Regression with an F1-Score of **0.5468**

### Key Findings

1. **Logistic Regression performs best** overall, offering strong baseline performance and interpretability.
2. **SVM and Random Forest** are close contenders, indicating that non-linear decision boundaries add marginal gains.
3. **Top predictive features** include the number of major vessels (`ca`), ST depression (`oldpeak`), and chest pain type (`cp`).
4. Model performance suggests room for improvement through:
   - Feature engineering
   - Hyperparameter tuning
   - Handling class imbalance
   - Ensemble methods

### Sample Visualizations

**Confusion Matrix (Logistic Regression):**

![Confusion Matrix](outputs/plots/confusion_matrix_logistic.png)

**Feature Importance (Random Forest):**

![Feature Importance](outputs/plots/feature_importance.png)

**Model Comparison:**

![Model Comparison](outputs/plots/model_comparison.png)

## 🚧 Future Improvements

### Model Enhancements
- [ ] Implement cross-validation for robust evaluation
- [ ] Hyperparameter tuning using GridSearchCV/RandomizedSearchCV
- [ ] Try advanced models: XGBoost, LightGBM, Neural Networks
- [ ] Implement ensemble methods (stacking, voting)
- [ ] Address class imbalance using SMOTE or class weights

### Feature Engineering
- [ ] Create interaction features
- [ ] Apply polynomial features
- [ ] Domain-specific feature engineering (e.g., BMI if height/weight available)
- [ ] Feature selection using statistical tests

### Deployment
- [ ] Build REST API using Flask/FastAPI
- [ ] Create web interface for predictions
- [ ] Containerize with Docker
- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] Implement model monitoring and retraining pipeline

### Additional Features
- [ ] SHAP/LIME for model explainability
- [ ] Generate PDF reports for predictions
- [ ] Multi-language support
- [ ] Integration with electronic health records (EHR)

## 📚 References

1. **Dataset:**
   - [UCI Machine Learning Repository - Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
   - [Kaggle - Heart Disease Data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)

2. **Research Papers:**
   - Detrano, R., et al. "International application of a new probability algorithm for the diagnosis of coronary artery disease." The American Journal of Cardiology (1989).
   - Alizadehsani, R., et al. "Machine learning-based coronary artery disease diagnosis: A comprehensive review." Computers in Biology and Medicine (2019).

3. **Documentation:**
   - [scikit-learn Documentation](https://scikit-learn.org/)
   - [pandas Documentation](https://pandas.pydata.org/)
   - [matplotlib Documentation](https://matplotlib.org/)

4. **Tutorials:**
   - [Kaggle Learn - Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)
   - [scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/index.html)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact [your-email@example.com](mailto:your-email@example.com).

---

**⭐ If you find this project helpful, please consider giving it a star!**

---

*Built with ❤️ for advancing healthcare through machine learning*
