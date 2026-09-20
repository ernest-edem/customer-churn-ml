# Customer Churn Analysis and Prediction

[![Tests](https://github.com/ernest-edem/customer-churn-ml/actions/workflows/tests.yml/badge.svg)](https://github.com/ernest-edem/customer-churn-ml/actions/workflows/tests.yml)

![Customer Churn Analysis and Prediction](docs/images/customer-churn-github-preview.png)

A configuration driven machine learning system for customer churn analysis and prediction, developed as part of the **Saiket Systems Machine Learning Internship**.

The project combines a reusable machine learning workflow with a **FastAPI prediction API** and a **React web application**. It covers data preparation, validation, preprocessing, feature selection, model benchmarking, training, evaluation, persistence, reporting, automated testing, and deployment.

## Live Application

**Frontend:**
https://customer-churn-ml-1-j47f.onrender.com

**Backend API:**
https://customer-churn-ml-api-juuf.onrender.com

**API Documentation:**
https://customer-churn-ml-api-juuf.onrender.com/docs

The production application has been tested successfully from the deployed frontend through the prediction API.

## Project Overview

Customer churn occurs when customers discontinue their relationship with a company or service provider. Predicting potential churn can help organizations identify customers who may require additional attention.

The main design principle of this project is:

> **Separate the ML framework from the ML problem.**

The system is configuration driven. Important settings such as the dataset, target column, preprocessing, feature selection, model, model parameters, evaluation metrics, and persistence path are controlled through configuration.

The project was developed with another guiding principle:

> **As simple as possible, as structured as necessary.**

## Application

The current application uses:

* React
* Vite
* Tailwind CSS
* FastAPI
* Pydantic
* scikit learn
* Recharts
* Lucide React

### Application flow

```text
Landing Page
     |
     v
Customer Assessment
     |
     v
POST /predict
     |
     v
FastAPI Backend
     |
     v
Persisted ML Pipeline
     |
     v
Prediction Results
     |
     v
Model Performance
```

The frontend provides:

* Customer assessment form
* Step by step form navigation
* Client side validation
* Loading and error states
* Churn prediction
* Churn probability visualization
* Model performance information
* Responsive layouts
* Accessibility focused form and navigation

## Machine Learning Workflow

The ML workflow is implemented as a modular framework:

```text
Configuration
     |
     v
Data Loading
     |
     v
Data Validation
     |
     v
Data Splitting
     |
     v
Preprocessing
     |
     v
Feature Selection
     |
     v
Model Factory
     |
     v
Model Training
     |
     v
Model Evaluation
     |
     v
Model Persistence
     |
     v
Metrics Reporting
```

The main workflow is implemented in:

```text
src/ml_system/pipeline.py
```

Model benchmarking and cross validation are implemented in:

```text
src/ml_system/benchmark.py
```

## Dataset

The project uses a customer churn dataset containing customer demographics, services, contracts, billing information, and churn status.

### Original dataset

| Property   |        Value |
| ---------- | -----------: |
| Rows       |        7,043 |
| Columns    |           21 |
| Target     |      `Churn` |
| Identifier | `customerID` |

During preparation:

* `customerID` was removed.
* `TotalCharges` was converted to numeric.
* 11 records containing invalid or blank `TotalCharges` values were removed.

### Processed dataset

| Property         | Value |
| ---------------- | ----: |
| Rows             | 7,032 |
| Columns          |    20 |
| Missing values   |     0 |
| Duplicate rows   |     0 |
| Non churn (`No`) | 5,163 |
| Churn (`Yes`)    | 1,869 |

Dataset preparation is implemented in:

```text
src/ml_system/data/dataset/prepare_dataset.py
```

The processed dataset is saved to:

```text
data/processed/customer_churn_processed.csv
```

## Data Preparation and Validation

The system validates the dataset before training.

Validation covers:

* Dataset type
* Empty datasets
* Required columns
* Target column
* Valid target observations

Dataset profiling provides:

* Dataset dimensions
* Column names
* Data types
* Missing values
* Duplicate rows
* Numerical columns
* Categorical columns
* Target distribution

Relevant modules:

```text
src/ml_system/data/loader.py
src/ml_system/data/validator.py
src/ml_system/data/profiler.py
```

## Data Splitting

The project uses an 80/20 stratified train test split.

```yaml
split:
  test_size: 0.20
  random_state: 42
  stratify: true
```

The test set remains isolated from model fitting and cross validation.

## Preprocessing

Numerical features use:

* Median imputation
* Standard scaling

Categorical features use:

* Most frequent value imputation
* One hot encoding
* `handle_unknown="ignore"`

Preprocessing is implemented with scikit learn:

* `Pipeline`
* `ColumnTransformer`
* `SimpleImputer`
* `StandardScaler`
* `OneHotEncoder`

The preprocessing pipeline is fitted using training data and persisted together with the model.

Implementation:

```text
src/ml_system/preprocessing/pipeline.py
```

## Feature Selection

The current configuration uses mutual information to select the top 10 transformed features.

```yaml
features:
  selection:
    enabled: true
    method: mutual_information
    top_k: 10
```

Implementation:

```text
src/ml_system/features/selector.py
```

## Model Selection

Four classification models were benchmarked:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

The model factory is implemented in:

```text
src/ml_system/models/factory.py
```

### Holdout benchmark

| Model               | Accuracy | Precision | Recall |     F1 |    ROC AUC |
| ------------------- | -------: | --------: | -----: | -----: | ---------: |
| Gradient Boosting   |   78.75% |    77.81% | 78.75% | 78.11% | **83.44%** |
| Logistic Regression |   78.82% |    78.06% | 78.82% | 78.33% |     83.39% |
| Random Forest       |   77.33% |    76.24% | 77.33% | 76.59% |     79.23% |
| Decision Tree       |   71.29% |    71.63% | 71.29% | 71.45% |     64.20% |

ROC AUC was the primary criterion used for selecting the production model. Gradient Boosting recorded the highest ROC AUC on the documented holdout benchmark.

The benchmark results are saved to:

```text
reports/metrics/model_benchmark.csv
```

## Cross Validation

The training portion of the dataset is evaluated using 5 fold stratified cross validation.

```text
StratifiedKFold
n_splits = 5
shuffle = true
random_state = 42
```

The test set is not included in cross validation.

### Cross validation results

| Model               |         Accuracy |        Precision |           Recall |               F1 |              ROC AUC |
| ------------------- | ---------------: | ---------------: | ---------------: | ---------------: | -------------------: |
| Gradient Boosting   | 79.96% +/- 0.61% | 79.00% +/- 0.72% | 79.96% +/- 0.61% | 79.16% +/- 0.70% | **84.48% +/- 0.61%** |
| Logistic Regression | 79.66% +/- 0.60% | 78.89% +/- 0.67% | 79.66% +/- 0.60% | 79.12% +/- 0.65% |     84.30% +/- 0.52% |
| Random Forest       | 77.81% +/- 0.61% | 76.73% +/- 0.60% | 77.81% +/- 0.61% | 77.05% +/- 0.58% |     81.28% +/- 0.50% |
| Decision Tree       | 73.55% +/- 1.12% | 73.68% +/- 0.94% | 73.55% +/- 1.12% | 73.61% +/- 1.02% |     66.83% +/- 1.28% |


Gradient Boosting:

```text
ROC AUC: 0.844769 +/- 0.006141
```

Logistic Regression:

```text
ROC AUC: 0.843022 +/- 0.005229
```


Cross validation results are saved to:

```text
reports/metrics/cross_validation.csv
```

## Production Model

The persisted production model is a `GradientBoostingClassifier`.

Configuration:

```yaml
model:
  name: gradient_boosting
  parameters:
    n_estimators: 100
    learning_rate: 0.1
    max_depth: 3
    random_state: 42
```

The complete preprocessing, feature selection, and model pipeline is persisted as:

```text
models/model.joblib
```

The model artifact is excluded from normal Git version control. CI and Docker deployment download the verified `v1.0.0` release artifact and validate its SHA-256 checksum before use.

## Final Model Performance

### Holdout evaluation

| Metric    |     Result |
| --------- | ---------: |
| Accuracy  | **78.75%** |
| Precision | **77.81%** |
| Recall    | **78.75%** |
| F1 score  | **78.11%** |
| ROC AUC   | **83.44%** |

### 5 fold cross validation

| Metric    |       Mean | Standard Deviation |
| --------- | ---------: | -----------------: |
| Accuracy  | **79.96%** |              0.61% |
| Precision | **79.00%** |              0.72% |
| Recall    | **79.96%** |              0.61% |
| F1 score  | **79.16%** |              0.70% |
| ROC AUC   | **84.48%** |              0.61% |

### Holdout confusion matrix

```text
                 Predicted
                 No      Yes

Actual No        914     119
Actual Yes       180     194
```

The model correctly identified 194 of the 374 churn customers in the holdout set.

This corresponds to approximately 52% recall for the churn class, which is an important limitation of the current model.

## Prediction API

The FastAPI application is implemented in:

```text
backend/app/main.py
```

### Health check

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

### Prediction

```http
POST /predict
```

The endpoint validates the customer assessment using Pydantic schemas and returns the prediction and model derived churn probability.

Example:

```json
{
  "prediction": "Yes",
  "churn_probability": 0.7354953070025138
}
```

The prediction is generated from the persisted machine learning pipeline. The probability is model derived and is not hardcoded.
### API documentation

FastAPI provides interactive Swagger documentation at:

```text
http://127.0.0.1:8000/docs
```

The production API documentation is available at:

```text
https://customer-churn-ml-api-juuf.onrender.com/docs
```

## React Frontend

The frontend is located in:

```text
frontend/
```

Technologies include:

* React 19
* Vite 8
* Tailwind CSS 4
* React Router
* Recharts
* Lucide React
* ESLint

Routes:

```text
/             Landing page
/assessment   Customer assessment
/results      Prediction results
/model        Model performance
```

The prediction service is:

```text
frontend/src/services/predictionService.js
```

### API configuration

Local development uses:

```text
http://127.0.0.1:8000
```

The production API is configured through:

```text
VITE_API_URL
```

Production configuration points to:

```text
https://customer-churn-ml-api-juuf.onrender.com
```

## Project Structure

```text
customer-churn-ml/
|
├── backend/
|   └── app/
|       ├── main.py
|       ├── schemas.py
|       └── services/
|           └── prediction.py
|
├── config/
|   ├── config.yaml
|   └── logging.yaml
|
├── data/
|   └── processed/
|       └── customer_churn_processed.csv
|
├── docs/
|   └── images/
|       └── customer-churn-github-preview.png
|
├── frontend/
|   ├── public/
|   |   └── favicon.svg
|   └── src/
|       ├── components/
|       |   └── CustomerAssessmentForm.jsx
|       ├── pages/
|       |   ├── AssessmentPage.jsx
|       |   ├── LandingPage.jsx
|       |   ├── ModelPage.jsx
|       |   └── ResultsPage.jsx
|       ├── services/
|       |   └── predictionService.js
|       ├── App.jsx
|       ├── index.css
|       └── main.jsx
|
├── models/
|   └── model.joblib
|
├── reports/
|   └── metrics/
|       ├── model_benchmark.csv
|       ├── cross_validation.csv
|       └── production_metrics.json
|
├── src/
|   └── ml_system/
|       ├── config/
|       ├── data/
|       ├── preprocessing/
|       ├── features/
|       ├── models/
|       ├── training/
|       ├── evaluation/
|       ├── persistence/
|       ├── reporting/
|       ├── benchmark.py
|       └── pipeline.py
|
├── tests/
|
├── .github/
|   └── workflows/
|       ├── tests.yml
|       └── docker.yml
|
├── Dockerfile
├── .dockerignore
├── .gitignore
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
```


The obsolete Streamlit application has been removed.

## Testing

The backend and ML system have a comprehensive automated test suite covering:

* Configuration
* Logging
* Dataset loading
* Dataset validation
* Dataset profiling
* Data splitting
* Preprocessing
* Feature selection
* Model factory
* Training
* Evaluation
* Persistence
* Pipeline integration
* Benchmarking
* Cross validation
* Reporting
* API endpoints
* Edge cases

Latest verified result:

```text
129 passed, 1 warning
```

The warning is an existing Starlette and AnyIO deprecation warning and did not cause a test failure.

### Backend tests

```powershell
python -m pytest -q
```

### Frontend lint

```powershell
cd frontend
npm run lint
```

### Frontend production build

```powershell
npm run build
```

The latest frontend lint completed without errors or warnings, and the production build completed successfully.

## Installation

### Requirements

* Python 3.11+
* Node.js
* npm
* Git

### Clone the repository

```powershell
git clone https://github.com/ernest-edem/customer-churn-ml.git
cd customer-churn-ml
```

### Create the Python environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

Install frontend dependencies:

```powershell
cd frontend
npm install
cd ..
```

## Run Locally

### Prepare the dataset

```powershell
python src/ml_system/data/dataset/prepare_dataset.py
```

### Run the ML pipeline

The main training workflow is:

Load configuration
      |
      v
Load dataset
      |
      v
Validate dataset
      |
      v
Split dataset
      |
      v
Build preprocessing pipeline
      |
      v
Select features
      |
      v
Build model
      |
      v
Train model
      |
      v
Evaluate model
      |
      v
Persist model
      |
      v
Save reports

### Run the benchmark

```powershell
python -m ml_system.benchmark
```

### Start FastAPI

From the project root:

```powershell
python -m uvicorn backend.app.main:app --reload
```

The local API is available at:

```text
http://127.0.0.1:8000
```

### Start React

Open another PowerShell terminal:

```powershell
cd frontend
npm run dev
```

Vite normally serves the frontend at:

```text
http://localhost:5173
```

## Docker

The project includes a production Docker configuration.

The Docker image:

* Uses Python 3.11
* Installs the ML and API dependencies
* Downloads the versioned production model artifact
* Verifies the model SHA-256 checksum
* Installs the project
* Runs the FastAPI application with Uvicorn

The Dockerfile is:

```text
Dockerfile
```

Build locally:

```powershell
docker build -t customer-churn-ml:local .
```

Run locally:

```powershell
docker run --rm -p 8000:8000 customer-churn-ml:local
```

## Continuous Integration

GitHub Actions provides automated validation.

### Python tests

```text
.github/workflows/tests.yml
```

The workflow:

1. Checks out the repository.
2. Sets up Python 3.11.
3. Downloads the versioned model artifact.
4. Verifies the model artifact.
5. Installs dependencies.
6. Installs the project.
7. Runs the test suite.

### Docker build

```text
.github/workflows/docker.yml
```

The Docker workflow verifies that the production Docker image can be built successfully.

Both workflows run from the `master` branch workflow configuration.

## Deployment

The application is deployed as two services:

```text
React Static Site
       |
       v
FastAPI API
       |
       v
Persisted ML Pipeline
```

### Frontend

https://customer-churn-ml-1-j47f.onrender.com
```

### Backend

```text
https://customer-churn-ml-api-juuf.onrender.com
```

The production deployment has been verified by:

* Successful frontend deployment
* Successful backend health check
* Successful production prediction
* Successful frontend build
* Successful backend test suite
* Successful GitHub Actions workflows

## Configuration

The main configuration file is:

```text
config/config.yaml
```

Configurable components include:

* Dataset path
* Target column
* Task type
* Train test split
* Random state
* Stratification
* Missing value strategy
* Categorical encoding
* Numerical scaling
* Feature selection
* Number of selected features
* Model
* Model parameters
* Evaluation metrics
* Model persistence path

This keeps problem specific settings separate from the reusable ML framework.

## Reporting

Evaluation reports are generated in:

```text
reports/metrics/
```

Current reports:

```text
production_metrics.json
model_benchmark.csv
cross_validation.csv
```

The reporting system supports machine readable JSON and CSV outputs.

## Logging

Application logging is configured through:

```text
config/logging.yaml
```

The project uses Python's logging framework with:

* Console logging
* Rotating file logging
* Configurable log levels
* Structured log formatting

## Limitations

The current model has several limitations.

### Churn class recall

The holdout evaluation produced approximately 52% recall for the churn class. This means a substantial number of actual churn customers were classified as non churn.

### Class imbalance

The dataset contains substantially more non churn observations than churn observations. Multiple evaluation metrics are therefore considered instead of accuracy alone.

### Dataset scope

Performance depends on the characteristics and quality of the available dataset. Results may not generalize to different customer populations without additional validation.

### External validation

The current evaluation uses the available dataset with an isolated holdout test set. Independent external validation has not been performed.

### Model optimization

The current implementation does not include extensive hyperparameter optimization or production model monitoring.

## Future Improvements

Potential future development includes:

* Hyperparameter optimization
* Class imbalance strategies
* Prediction threshold analysis
* Additional model evaluation
* ROC and Precision Recall curves
* Explainability using SHAP or similar methods
* Model monitoring
* External validation
* Frontend and backend CI validation
* Additional production observability

## What I Learned

This project provided practical experience across the full machine learning development workflow.

Key areas of learning include:

* Designing reusable ML systems
* Configuration driven software design
* Data preparation and validation
* Numerical and categorical preprocessing
* Feature selection
* Model benchmarking
* Cross validation
* Model persistence
* Evaluation and reporting
* Building REST APIs with FastAPI
* Building interfaces with React
* Connecting frontend and backend services
* Automated testing with pytest
* Continuous integration with GitHub Actions
* Docker based deployment
* Production API configuration
* CORS configuration
* Debugging deployment issues
* Maintaining a clean separation between ML logic and application logic

## Internship Requirements

The six core Machine Learning Internship tasks were implemented:

```text
[x] Data Preparation
[x] Data Splitting
[x] Feature Selection
[x] Model Selection
[x] Model Training
[x] Model Evaluation
```

The project extends these requirements with:

```text
[x] Configuration management
[x] Dataset validation
[x] Dataset profiling
[x] Automated preprocessing
[x] Cross validation
[x] Model persistence
[x] Metrics reporting
[x] Structured logging
[x] Automated testing
[x] FastAPI API
[x] React frontend
[x] Docker deployment
[x] GitHub Actions
```

## Technologies

### Machine Learning

* Python 3.11
* Pandas
* NumPy
* scikit learn
* PyYAML
* Joblib

### Backend

* FastAPI
* Uvicorn
* Pydantic
* HTTPX

### Frontend

* React 19
* Vite 8
* Tailwind CSS 4
* React Router
* Recharts
* Lucide React
* ESLint

### Testing and Deployment

* Pytest
* Git
* GitHub
* GitHub Actions
* Docker
* Render

## Project Status

**Status: Completed internship implementation with a deployed full stack application.**

### Verified status

```text
Backend tests:              129 passed
Frontend lint:              Passed
Frontend production build:  Passed
Production API health:      Passed
Production prediction:      Passed
Frontend deployment:        Live
Backend deployment:         Live
Git working tree:           Clean
```

### Production model

```text
GradientBoostingClassifier
```

### Holdout ROC AUC

```text
83.44%
```

### 5 Fold Cross Validation ROC AUC

```text
84.48% +/- 0.61%
```

## About the Author

**Ernest Edem Dzisah** is a Computer Science and Engineering student focused on Software Engineering and Artificial Intelligence and Machine Learning.

His technical interests include machine learning, data science, Python development, and building practical software systems that combine data, automation, and intelligent decision making.

### Research Interests

* Machine Learning and Artificial Intelligence
* Explainable and Interpretable Machine Learning
* Applied Machine Learning for Healthcare
* Predictive Modeling
* Natural Language Processing
* Computer Vision
* Responsible and Trustworthy AI
* Machine Learning Systems and MLOps
* Data Science and Applied Statistics
* AI driven Software Engineering

He is interested in research opportunities that connect machine learning with practical problems and lead to reproducible, useful, and deployable solutions.

## Contact

* **Email:** [ernestedem.d@gmail.com](mailto:ernestedem.d@gmail.com)
* **GitHub:** https://github.com/ernest-edem
* **LinkedIn:** https://www.linkedin.com/in/ernest-edem-dzisah

For research collaborations, software engineering opportunities, AI/ML projects, internships, or other professional opportunities, contact Ernest through the channels above.

This project was developed as part of the **Saiket Systems Machine Learning Internship**.

## License

This project was developed for educational and internship purposes as part of the Saiket Systems Machine Learning Internship.
