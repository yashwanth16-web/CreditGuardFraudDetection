# CreditGuard — Credit Card Fraud Detection Using Machine Learning

## Project Overview

CreditGuard is an original machine learning project that identifies potentially fraudulent credit card transactions. It compares six supervised and unsupervised machine learning models and provides an interactive Streamlit dashboard for model evaluation and transaction scoring.

The project emphasizes imbalanced classification, chronological data splitting, validation-based threshold selection, and evaluation on a reserved test set.

## Objectives

- Detect potentially fraudulent transactions.
- Compare six machine learning approaches using fraud-focused metrics.
- Study the precision–recall trade-off in highly imbalanced data.
- Select decision thresholds using validation data.
- Present model results through an interactive dashboard.
- Support batch transaction scoring from CSV files.

## Machine Learning Models

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost
5. Linear Support Vector Machine (SVM)
6. Isolation Forest

The first five models use labeled transaction data. Isolation Forest is an anomaly-detection model whose anomaly scores are oriented so higher scores indicate stronger fraud evidence.

## Technology Stack

- Python 3.11
- pandas and NumPy
- scikit-learn
- XGBoost
- Matplotlib
- Plotly
- Streamlit
- joblib

## Dataset

The project uses the public Credit Card Fraud Detection dataset available from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains anonymized transaction features, transaction time, transaction amount, and a binary target column named `Class`.

- `Class = 0`: legitimate transaction
- `Class = 1`: fraudulent transaction

The dataset is externally sourced. The preprocessing, model training, evaluation workflow, and dashboard implementation were developed for this project.

## Project Structure

```text
CreditCardFraudDetection/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   └── processed/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
├── models/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── linear_svm.joblib
│   └── isolation_forest.joblib
├── reports/
│   ├── figures/
│   ├── threshold_comparison.csv
│   ├── precision_recall_metrics.csv
│   └── final_test_results.csv
└── src/
    ├── check_environment.py
    ├── validate_dataset.py
    ├── explore_data.py
    ├── prepare_data.py
    ├── preprocessing.py
    ├── train_logistic_regression.py
    ├── train_decision_tree.py
    ├── train_random_forest.py
    ├── train_xgboost.py
    ├── train_svm.py
    ├── train_isolation_forest.py
    ├── compare_models.py
    ├── tune_thresholds.py
    ├── evaluate_final_model.py
    ├── evaluate_curves.py
    └── visualize_comparison.py
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd CreditCardFraudDetection
```

Replace `YOUR_REPOSITORY_URL` with the URL of your repository.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the dataset

Download the dataset from the Kaggle link above and place `creditcard.csv` inside `data/raw/`.

### 5. Launch the dashboard

```powershell
streamlit run app.py
```

## Evaluation Strategy

The project uses chronological training, validation, and test partitions.

- Training data is used to fit the models and preprocessing steps.
- Validation data is used for model comparison and threshold selection.
- Test data is reserved for the final evaluation of the selected XGBoost model.

Metrics include:

- Average precision (area under the precision–recall curve)
- ROC-AUC
- Fraud precision
- Fraud recall
- Fraud F1-score
- Confusion matrix
- False positives and false negatives

Accuracy is not treated as the primary metric because fraud cases are extremely rare relative to legitimate transactions.

## Current Results

The reserved test evaluation for XGBoost produced the following results:

| Metric | Result |
|---|---:|
| Average precision | 0.770869 |
| ROC-AUC | 0.984931 |
| Fraud precision | 86.67% |
| Fraud recall | 75.00% |
| Fraud F1-score | 0.804124 |
| True positives | 39 |
| False positives | 6 |
| True negatives | 42,663 |
| False negatives | 13 |

These results apply to the current dataset split and selected threshold. They do not guarantee performance on future transactions.

## Dashboard Features

- Executive overview of test-set metrics
- Confusion matrix visualization
- Model comparison leaderboard
- Precision–recall curve comparison
- CSV transaction scoring
- Downloadable scored transaction results
- Configurable fraud decision threshold
- Project information and limitations

## Limitations

- The dataset contains anonymized features and may not represent current fraud patterns.
- The model's performance can change with transaction behavior and data drift.
- A decision threshold involves trade-offs between missed fraud and false alerts.
- Validation scores may be optimistic when thresholds are tuned on the same validation partition used for reporting.
- CSV scoring requires the same feature names and compatible feature representations used during training.

## Future Improvements

- Add automated model tests and data validation tests.
- Add model explainability with SHAP.
- Evaluate calibration and threshold stability.
- Add monitoring for data drift.
- Introduce reproducible experiment tracking.
- Improve dashboard accessibility and reporting.
- Develop a deployment workflow with model versioning.

## Disclaimer

CreditGuard is an educational and research prototype. It is not a production banking system and must not be used as the sole basis for approving, declining, or blocking financial transactions.
