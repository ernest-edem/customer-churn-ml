# Customer Churn Analysis and Prediction

[![Tests](https://github.com/ernest-edem/customer-churn-ml/actions/workflows/tests.yml/badge.svg)](https://github.com/ernest-edem/customer-churn-ml/actions/workflows/tests.yml)

![Customer Churn Analysis and Prediction](docs/images/customer-churn-github-preview.png)

A configuration-driven machine learning system for customer churn analysis and prediction, developed as part of the **Saiket Systems Machine Learning Internship**.

The project implements a complete machine learning workflow covering data preparation, data splitting, preprocessing, feature selection, model selection, model training, evaluation, cross-validation, persistence, reporting, and automated testing.

## 1. Project Overview

Customer churn occurs when customers discontinue their relationship with a company or service provider. Predicting potential churn can help organizations identify customers who may leave and support data-driven retention strategies.

This project develops a reusable machine learning system that predicts customer churn while keeping the machine learning framework separate from the specific customer churn problem.

The system is **configuration-driven**, meaning important settings such as the dataset path, target column, train-test split, preprocessing strategies, feature-selection method, model, model parameters, and evaluation metrics are controlled through configuration rather than hardcoded throughout the application.

### Core Principle

> **Separate the ML framework from the ML problem.**

The implementation follows the principle:

> **As simple as possible, as structured as necessary.**

---

## 2. Internship Requirements

The project implements all six major tasks specified for the internship:

1. Data Preparation
2. Data Splitting
3. Feature Selection
4. Model Selection
5. Model Training
6. Model Evaluation

Additional supporting capabilities were implemented to make the workflow reproducible and maintainable:

* Configuration management
* Dataset validation
* Dataset profiling
* Automated preprocessing
* Cross-validation
* Model persistence
* Metrics reporting
* Structured logging
* Automated testing

---

## 3. Objectives

The main objectives are to:

* Prepare the customer churn dataset for machine learning.
* Separate training and testing data correctly.
* Automatically preprocess numerical and categorical features.
* Apply configurable feature selection.
* Compare multiple classification algorithms.
* Validate model performance using cross-validation.
* Train the selected model.
* Evaluate model performance using appropriate classification metrics.
* Persist the trained model for later use.
* Generate machine-readable evaluation reports.
* Maintain a modular and reusable ML architecture.
* Validate the system through automated tests.

---

## 4. Dataset

The project uses a customer churn dataset containing information about customer demographics, services, contracts, billing, and churn status.

### Original Dataset

* Rows: **7,043**
* Columns: **21**
* Target: `Churn`
* Identifier: `customerID`
* Numerical and categorical features are included.

The `TotalCharges` column was originally represented as a string and was converted to a numeric data type during dataset preparation.

The `customerID` identifier was removed because it does not provide meaningful predictive information.

After preprocessing:

* Rows: **7,032**
* Columns: **20**
* Missing values: **0**
* Duplicate rows: **0**
* Target classes:

  * `No`: 5,163
  * `Yes`: 1,869

Eleven records with invalid or blank `TotalCharges` values were removed during dataset preparation.

---

## 5. System Architecture

The system follows a modular machine learning architecture:

```text
                    Configuration
                         │
                         ▼
                  Configuration Loader
                         │
                         ▼
                    Data Loading
                         │
                         ▼
                   Data Validation
                         │
                         ▼
                    Data Splitting
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        Training Data          Testing Data
              │
              ▼
        Preprocessing
              │
              ▼
       Feature Selection
              │
              ▼
         Model Factory
              │
              ▼
        Model Training
              │
              ▼
        Model Evaluation ◄──────── Testing Data
              │
        ┌─────┴──────────┐
        ▼                ▼
   Model Store      Metrics Reports
```

The main training workflow is orchestrated by:

```text
src/ml_system/pipeline.py
```

Model benchmarking and cross-validation are implemented in:

```text
src/ml_system/benchmark.py
```

---

## 6. Project Structure

```text
customer-churn-ml/

│
├── config/
│   ├── config.yaml
│   └── logging.yaml
│
├── data/
│   └── processed/
│       └── customer_churn_processed.csv
│
├── models/
│
├── reports/
│   ├── figures/
│   └── metrics/
│       ├── model_benchmark.csv
│       ├── cross_validation.csv
│       └── production_metrics.json
│
├── src/
│   └── ml_system/
│       ├── config/
│       │   ├── loader.py
│       │   └── schemas.py
│       │
│       ├── data/
│       │   ├── loader.py
│       │   ├── validator.py
│       │   ├── profiler.py
│       │   ├── splitter.py
│       │   └── dataset/
│       │       ├── Churn_Dataset.csv
│       │       └── prepare_dataset.py
│       │
│       ├── preprocessing/
│       │   └── pipeline.py
│       │
│       ├── features/
│       │   └── selector.py
│       │
│       ├── models/
│       │   └── factory.py
│       │
│       ├── training/
│       │   └── trainer.py
│       │
│       ├── evaluation/
│       │   └── evaluator.py
│       │
│       ├── persistence/
│       │   └── model_store.py
│       │
│       ├── reporting/
│       │   ├── __init__.py
│       │   └── report_generator.py
│       │
│       ├── benchmark.py
│       └── pipeline.py
│
├── tests/
│   ├── test_benchmark.py
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_data_profiler.py
│   ├── test_data_splitter.py
│   ├── test_data_validation.py
│   ├── test_edge_cases.py
│   ├── test_evaluation.py
│   ├── test_feature_selection.py
│   ├── test_logging.py
│   ├── test_model_factory.py
│   ├── test_persistence.py
│   ├── test_pipeline.py
│   ├── test_preprocessing.py
│   ├── test_reporting.py
│   └── test_training.py
│
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 7. Data Preparation

Dataset preparation is implemented in:

```text
src/ml_system/data/dataset/prepare_dataset.py
```

The preparation process:

1. Loads the raw CSV dataset.
2. Removes the `customerID` identifier.
3. Converts `TotalCharges` to numeric values.
4. Handles invalid `TotalCharges` values.
5. Removes records where `TotalCharges` cannot be converted.
6. Saves the processed dataset to:

```text
data/processed/customer_churn_processed.csv
```

The preparation step produces a clean dataset suitable for the ML pipeline.

---

## 8. Data Validation and Profiling

Dataset validation is implemented in:

```text
src/ml_system/data/validator.py
```

The validator checks:

* Dataset type
* Empty datasets
* Presence of columns
* Target column existence
* Valid target observations

Dataset profiling is implemented in:

```text
src/ml_system/data/profiler.py
```

The profiler provides information about:

* Dataset dimensions
* Column names
* Data types
* Missing values
* Duplicate rows
* Numerical columns
* Categorical columns
* Target distribution

---

## 9. Data Splitting

Data splitting is implemented in:

```text
src/ml_system/data/splitter.py
```

The default configuration uses:

```yaml
split:
  test_size: 0.20
  random_state: 42
  stratify: true
```

Therefore:

* **80%** of the data is used for training.
* **20%** is used for testing.
* `random_state=42` provides reproducibility.
* Stratification preserves the target-class distribution between training and testing data.

The test set remains isolated from model fitting and cross-validation.

---

## 10. Preprocessing

Preprocessing is implemented in:

```text
src/ml_system/preprocessing/pipeline.py
```

The system dynamically identifies numerical and categorical features.

### Numerical Features

The configured numerical preprocessing uses:

* Median imputation
* Standard scaling

### Categorical Features

The configured categorical preprocessing uses:

* Most-frequent-value imputation
* One-hot encoding
* `handle_unknown="ignore"`

The preprocessing is implemented using scikit-learn's:

* `Pipeline`
* `ColumnTransformer`
* `SimpleImputer`
* `StandardScaler`
* `OneHotEncoder`

This ensures that preprocessing operations are learned from the training data as part of the machine learning pipeline.

---

## 11. Feature Selection

Feature selection is implemented in:

```text
src/ml_system/features/selector.py
```

The current configuration uses:

```yaml
features:
  selection:
    enabled: true
    method: mutual_information
    top_k: 10
```

The system uses **mutual information** to select the top 10 transformed features.

Feature selection can be disabled or modified through configuration without changing the core training workflow.

---

## 12. Model Selection

Four classification models were implemented and benchmarked:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

The model factory is implemented in:

```text
src/ml_system/models/factory.py
```

Supported models are registered centrally and created from configuration.

---

## 13. Model Benchmark

The four models were evaluated using the same train-test split and evaluation criteria.

### Holdout Benchmark Results

| Model               | Accuracy | Precision | Recall |     F1 |    ROC-AUC |
| ------------------- | -------: | --------: | -----: | -----: | ---------: |
| Gradient Boosting   |   78.75% |    77.81% | 78.75% | 78.11% | **83.44%** |
| Logistic Regression |   78.82% |    78.06% | 78.82% | 78.33% |     83.39% |
| Random Forest       |   77.33% |    76.24% | 77.33% | 76.59% |     79.23% |
| Decision Tree       |   71.29% |    71.63% | 71.29% | 71.45% |     64.20% |

The benchmark demonstrates that Logistic Regression achieved slightly higher accuracy, precision, recall, and F1-score than Gradient Boosting on the holdout set.

However, **Gradient Boosting achieved the highest ROC-AUC**, at 83.44%.

Because ROC-AUC provides an important measure of binary classification discrimination across classification thresholds, Gradient Boosting was selected as the production model.

The benchmark results are persisted to:

```text
reports/metrics/model_benchmark.csv
```

---

## 14. Cross-Validation

To provide a more robust estimate of model performance, the training portion of the dataset is also evaluated using **5-fold stratified cross-validation**.

The test set remains isolated and is not included in cross-validation.

Cross-validation is implemented in:

```text
src/ml_system/benchmark.py
```

The evaluation uses:

```text
StratifiedKFold
n_splits = 5
shuffle = true
random_state = 42
```

Preprocessing and feature selection remain inside the scikit-learn pipeline during each fold, preventing information leakage between training and validation folds.

### 5-Fold Cross-Validation Results

| Model               | Accuracy Mean | Accuracy Std | Precision Mean | Precision Std | Recall Mean | Recall Std |    F1 Mean | F1 Std | ROC-AUC Mean | ROC-AUC Std |
| ------------------- | ------------: | -----------: | -------------: | ------------: | ----------: | ---------: | ---------: | -----: | -----------: | ----------: |
| Gradient Boosting   |    **79.96%** |        0.61% |     **79.00%** |         0.72% |  **79.96%** |      0.61% | **79.16%** |  0.70% |   **84.48%** |       0.61% |
| Logistic Regression |        79.66% |        0.60% |         78.89% |         0.67% |      79.66% |      0.60% |     79.12% |  0.65% |       84.30% |       0.52% |
| Random Forest       |        77.81% |        0.61% |         76.73% |         0.60% |      77.81% |      0.61% |     77.05% |  0.58% |       81.28% |       0.50% |
| Decision Tree       |        73.55% |        1.12% |         73.68% |         0.94% |      73.55% |      1.12% |     73.61% |  1.02% |       66.83% |       1.28% |

Gradient Boosting achieved the highest mean ROC-AUC:

```text
0.844769 ± 0.006141
```

Logistic Regression was very close:

```text
0.843022 ± 0.005229
```

The relatively small standard deviations indicate consistent performance across the five folds.

The cross-validation results provide additional evidence supporting Gradient Boosting as the selected production model.

The results are persisted to:

```text
reports/metrics/cross_validation.csv
```

Cross-validation is used as an evaluation and model-selection aid. It does not replace the isolated holdout test evaluation.

---

## 15. Final Model

The production configuration uses:

```yaml
model:
  name: gradient_boosting
  parameters:
    n_estimators: 100
    learning_rate: 0.1
    max_depth: 3
    random_state: 42
```

The final model is:

```text
GradientBoostingClassifier
```

Gradient Boosting was retained as the production model because it achieved the highest ROC-AUC on both the holdout benchmark and 5-fold cross-validation.

---

## 16. Final Model Performance

The final Gradient Boosting model achieved the following holdout performance:

| Metric    |     Result |
| --------- | ---------: |
| Accuracy  | **78.75%** |
| Precision | **77.81%** |
| Recall    | **78.75%** |
| F1-score  | **78.11%** |
| ROC-AUC   | **83.44%** |

### Cross-Validation Performance

The corresponding 5-fold cross-validation results were:

| Metric    |       Mean | Standard Deviation |
| --------- | ---------: | -----------------: |
| Accuracy  | **79.96%** |              0.61% |
| Precision | **79.00%** |              0.72% |
| Recall    | **79.96%** |              0.61% |
| F1-score  | **79.16%** |              0.70% |
| ROC-AUC   | **84.48%** |              0.61% |

The cross-validation results are consistent with the holdout evaluation and provide additional evidence that the model's performance is reasonably stable across different training and validation folds.

### Confusion Matrix

```text
                Predicted

                 No      Yes

Actual No       914     119

       Yes      180     194
```

The model correctly identified:

* 914 non-churn customers
* 194 churn customers

It incorrectly classified:

* 119 non-churn customers as churn
* 180 churn customers as non-churn

The class-level results show that identifying churn customers remains more difficult than identifying customers who do not churn.

---

## 17. Model Training

Model training is implemented in:

```text
src/ml_system/training/trainer.py
```

The training workflow combines:

```text
Preprocessing
      ↓
Feature Selection
      ↓
Model
```

into a single scikit-learn `Pipeline`.

This keeps the transformations and trained estimator together and allows the complete pipeline to be persisted.

---

## 18. Model Evaluation

Evaluation is implemented in:

```text
src/ml_system/evaluation/evaluator.py
```

The system supports:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrix
* Classification report

Metrics are configured through:

```yaml
evaluation:
  metrics:
    - accuracy
    - precision
    - recall
    - f1
    - roc_auc
```

---

## 19. Model Persistence

Model persistence is implemented in:

```text
src/ml_system/persistence/model_store.py
```

The trained pipeline can be saved and loaded using `joblib`.

The configured model location is:

```text
models/model.joblib
```

Generated model artifacts are excluded from version control through `.gitignore`.

---

## 20. Reporting

Reporting functionality is implemented in:

```text
src/ml_system/reporting/report_generator.py
```

The reporting module supports:

* JSON evaluation reports
* CSV model benchmark reports
* CSV cross-validation reports

Generated reports include:

```text
reports/metrics/production_metrics.json
reports/metrics/model_benchmark.csv
reports/metrics/cross_validation.csv
```

The reporting functions are also covered by dedicated automated tests.

---

## 21. Configuration

The system is controlled through:

```text
config/config.yaml
```

Important configurable components include:

* Dataset path
* Target column
* Task type
* Test size
* Random state
* Stratification
* Missing-value strategy
* Categorical encoding
* Numerical scaling
* Feature-selection method
* Number of selected features
* Model
* Model parameters
* Evaluation metrics
* Model persistence path

For example:

```yaml
data:
  path: data/processed/customer_churn_processed.csv
  target_column: Churn

task:
  type: classification
```

This design minimizes hardcoded problem-specific assumptions.

---

## 22. Logging

Structured application logging is configured through:

```text
config/logging.yaml
```

The system uses Python's `logging` framework with:

* Console logging
* Rotating file logging
* Configurable log levels
* Structured log formatting

Application logs are excluded from version control.

---

## 23. Testing

The project includes automated tests covering the major system components.

The test suite covers:

* Configuration loading
* Logging
* Dataset loading
* Dataset validation
* Dataset profiling
* Dataset splitting
* Preprocessing
* Feature selection
* Model factory
* Model training
* Evaluation
* Persistence
* Pipeline integration
* Benchmarking
* Cross-validation
* Reporting
* Edge cases

The current test suite contains:

```text
125 tests
```

All tests passed successfully:

```text
125 passed in 292.44s (0:04:52)
```

Tests can be executed with:

```powershell
pytest -q
```

---

## 24. Installation

### Requirements

* Python 3.11+
* pip
* Git

### Clone the repository

```powershell
git clone https://github.com/ernest-edem/customer-churn-ml.git
cd customer-churn-ml
```

### Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
```

### Activate the environment

```powershell
.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Install the project

```powershell
python -m pip install -e .
```

---

## 25. Prepare the Dataset

Run:

```powershell
python src/ml_system/data/dataset/prepare_dataset.py
```

This creates:

```text
data/processed/customer_churn_processed.csv
```

---

## 26. Run the Complete Training Pipeline

The complete workflow can be executed through the project's training pipeline.

The workflow performs:

```text
Load configuration
        ↓
Load dataset
        ↓
Validate dataset
        ↓
Split dataset
        ↓
Build preprocessing pipeline
        ↓
Select features
        ↓
Build model
        ↓
Train model
        ↓
Evaluate model
        ↓
Save model
        ↓
Save metrics
```

---

## 27. Run the Model Benchmark and Cross-Validation

To compare the four supported models and run 5-fold stratified cross-validation:

```powershell
python -m ml_system.benchmark
```

The benchmark evaluates:

```text
Logistic Regression
Decision Tree
Random Forest
Gradient Boosting
```

The command produces:

```text
Model Benchmark Results
5-Fold Stratified Cross-Validation Results
```

and persists the results to:

```text
reports/metrics/model_benchmark.csv
reports/metrics/cross_validation.csv
```

---

## 28. Reproducibility

Reproducibility is supported through configuration-controlled random states.

The current project uses:

```text
random_state = 42
```

for data splitting, cross-validation, and applicable models.

The preprocessing, feature selection, model training, and evaluation steps are assembled into consistent scikit-learn pipelines.

Cross-validation uses:

```text
StratifiedKFold
n_splits = 5
shuffle = true
random_state = 42
```

This provides a reproducible evaluation procedure while preserving the class distribution across folds.

---

## 29. Limitations

The current system has several limitations.

### Class imbalance

The churn dataset contains substantially more non-churn customers than churn customers. As a result, the model performs better on the `No` class than on the `Yes` class.

### Churn recall

The final model achieved approximately **52% recall for the churn class** on the holdout evaluation. This means a significant proportion of actual churn customers are still classified as non-churn.

### Binary classification

The current evaluation workflow is designed for binary classification, particularly for ROC-AUC evaluation.

### Dataset scope

The model's performance depends on the characteristics and quality of the available dataset. Results may not generalize to different customer populations without additional validation.

### No external validation

The current evaluation uses the available dataset and an isolated holdout test set. External validation on an independent customer population has not been performed.

---

## 30. Future Improvements

Potential future improvements include:

* Hyperparameter optimization
* Class-imbalance handling
* Threshold optimization
* Additional model evaluation
* ROC and Precision-Recall curve generation
* Explainability using SHAP or similar methods
* Model monitoring
* Prediction API
* Deployment of the trained model
* External validation on an independent dataset

These improvements are intentionally outside the current internship implementation to keep the system focused on the required objectives.

---

## 31. Conclusion

This project implements a complete and reusable machine learning workflow for customer churn analysis and prediction.

All six internship tasks were completed:

```text
✓ Data Preparation

✓ Data Splitting

✓ Feature Selection

✓ Model Selection

✓ Model Training

✓ Model Evaluation
```

Four classification models were benchmarked, with Gradient Boosting selected as the production model based primarily on ROC-AUC performance.

The final holdout evaluation achieved:

```text
Accuracy:  78.75%
F1-score:  78.11%
ROC-AUC:   83.44%
```

The 5-fold stratified cross-validation evaluation achieved:

```text
Accuracy:  79.96% ± 0.61%
F1-score:  79.16% ± 0.70%
ROC-AUC:   84.48% ± 0.61%
```

Gradient Boosting achieved the highest ROC-AUC in both the holdout benchmark and cross-validation, while Logistic Regression remained a close alternative.

The project also demonstrates practical software engineering principles through modular architecture, configuration-driven execution, automated testing, structured logging, model persistence, reproducible evaluation, and machine-readable reporting.

---

## 32. Technologies Used

* Python 3.11
* Pandas
* NumPy
* Scikit-learn
* PyYAML
* Joblib
* Pytest
* Git
* GitHub

---

## 33. Project Status

**Status: Completed**

The implementation satisfies the defined Machine Learning Project Contract v1.0 and the six core internship tasks.

**Test status:**

```text
125 passed
```

**Production model:**

```text
GradientBoostingClassifier
```

**Holdout ROC-AUC:**

```text
83.44%
```

**5-Fold Cross-Validation ROC-AUC:**

```text
84.48% ± 0.61%
```

**Generated evaluation reports:**

```text
reports/metrics/production_metrics.json
reports/metrics/model_benchmark.csv
reports/metrics/cross_validation.csv
```

---
## About the Author

**Ernest Edem Dzisah** is a Computer Science and Engineering student focused on Software Engineering and Artificial Intelligence and Machine Learning (AI/ML).

His technical interests include machine learning, data science, Python development, and building practical software systems that combine data, automation, and intelligent decision-making.

This project was developed as part of his Saiket Systems Machine Learning Internship and demonstrates practical experience with:

* Python and scikit-learn
* Data preprocessing and feature engineering
* Supervised machine learning
* Feature selection
* Model benchmarking and evaluation
* Cross-validation
* Configuration-driven software design
* Automated testing with pytest
* Continuous integration with GitHub Actions

* **Email:** `ernestedem.d@gmail.com`
* **GitHub:** [ernest-edem](https://github.com/ernest-edem)
* **LinkedIn:** [LinkedIn](https://www.linkedin.com/in/ernest-edem-dzisah)

---

## License

This project was developed for educational and internship purposes as part of the Saiket Systems Machine Learning Internship.
