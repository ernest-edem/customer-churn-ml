from pathlib import Path

import pandas as pd
import pytest

from ml_system.config.schemas import (
    DataSettings,
    EncodingSettings,
    EvaluationSettings,
    FeatureSelectionSettings,
    FeatureSettings,
    MissingValueSettings,
    ModelSettings,
    PersistenceSettings,
    PreprocessingSettings,
    ProjectSettings,
    ScalingSettings,
    Settings,
    SplitSettings,
    TaskSettings,
)
from ml_system.data import load_dataset, split_dataset
from ml_system.exceptions.errors import DataError
from ml_system.persistence import load_model
from ml_system.pipeline import TrainingResult, run_training_pipeline


@pytest.fixture
def integration_dataset(tmp_path: Path) -> Path:
    dataset = pd.DataFrame(
        {
            "Age": [
                21,
                25,
                31,
                35,
                42,
                48,
                53,
                61,
                67,
                72,
                28,
                39,
                45,
                57,
                64,
                70,
                23,
                34,
                51,
                68,
            ],
            "MonthlyCharges": [
                20.0,
                25.0,
                31.0,
                35.0,
                42.0,
                48.0,
                53.0,
                61.0,
                67.0,
                72.0,
                28.0,
                39.0,
                45.0,
                57.0,
                64.0,
                70.0,
                23.0,
                34.0,
                51.0,
                68.0,
            ],
            "Contract": [
                "Month-to-month",
                "One year",
                "Month-to-month",
                "Two year",
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
                "Two year",
                "One year",
                "Month-to-month",
                "Two year",
                "One year",
                "Month-to-month",
                "Two year",
                "One year",
                "Month-to-month",
                "Two year",
                "One year",
                "Month-to-month",
            ],
            "Churn": [
                "No",
                "No",
                "Yes",
                "No",
                "Yes",
                "No",
                "Yes",
                "No",
                "Yes",
                "No",
                "Yes",
                "No",
                "Yes",
                "No",
                "Yes",
                "No",
                "Yes",
                "No",
                "Yes",
                "No",
            ],
        }
    )

    dataset_path = tmp_path / "customer_churn.csv"
    dataset.to_csv(dataset_path, index=False)

    return dataset_path


@pytest.fixture
def settings(
    integration_dataset: Path,
    tmp_path: Path,
) -> Settings:
    return Settings(
        project=ProjectSettings(
            name="customer_churn_ml",
            version="1.0",
        ),
        data=DataSettings(
            path=str(integration_dataset),
            target_column="Churn",
        ),
        task=TaskSettings(
            type="classification",
        ),
        split=SplitSettings(
            test_size=0.20,
            random_state=42,
            stratify=True,
        ),
        preprocessing=PreprocessingSettings(
            missing_values=MissingValueSettings(
                strategy="median",
            ),
            categorical_encoding=EncodingSettings(
                strategy="onehot",
            ),
            numerical_scaling=ScalingSettings(
                strategy="standard",
            ),
        ),
        features=FeatureSettings(
            selection=FeatureSelectionSettings(
                enabled=True,
                method="mutual_information",
                top_k=2,
            ),
        ),
        model=ModelSettings(
            name="random_forest",
            parameters={
                "n_estimators": 20,
                "random_state": 42,
            },
        ),
        evaluation=EvaluationSettings(
            metrics=[
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
            ],
        ),
        persistence=PersistenceSettings(
            model_path=str(
                tmp_path / "models" / "model.joblib"
            ),
        ),
    )


def test_run_training_pipeline_completes(
    settings: Settings,
):
    result = run_training_pipeline(settings)

    assert isinstance(result, TrainingResult)
    assert result.model is not None
    assert isinstance(result.metrics, dict)
    assert result.model_path.exists()


def test_run_training_pipeline_returns_required_metrics(
    settings: Settings,
):
    result = run_training_pipeline(settings)

    assert "accuracy" in result.metrics
    assert "precision" in result.metrics
    assert "recall" in result.metrics
    assert "f1" in result.metrics
    assert "roc_auc" in result.metrics


def test_run_training_pipeline_returns_evaluation_details(
    settings: Settings,
):
    result = run_training_pipeline(settings)

    assert "confusion_matrix" in result.metrics
    assert "classification_report" in result.metrics

    assert isinstance(
        result.metrics["confusion_matrix"],
        list,
    )

    assert isinstance(
        result.metrics["classification_report"],
        str,
    )


def test_run_training_pipeline_persists_configured_model(
    settings: Settings,
):
    result = run_training_pipeline(settings)

    assert result.model_path == Path(
        settings.persistence.model_path
    )
    assert result.model_path.is_file()


def test_run_training_pipeline_model_can_predict(
    settings: Settings,
):
    result = run_training_pipeline(settings)

    predictions = result.model.predict(
        pd.DataFrame(
            {
                "Age": [40, 60],
                "MonthlyCharges": [40.0, 60.0],
                "Contract": [
                    "One year",
                    "Month-to-month",
                ],
            }
        )
    )

    assert len(predictions) == 2


def test_run_training_pipeline_uses_configuration(
    settings: Settings,
):
    settings.model = ModelSettings(
        name="logistic_regression",
        parameters={
            "max_iter": 1000,
            "random_state": 42,
        },
    )

    settings.features.selection.enabled = False

    result = run_training_pipeline(settings)

    assert result.model.named_steps["model"].__class__.__name__ == (
        "LogisticRegression"
    )

    assert "feature_selection" not in result.model.named_steps


def test_run_training_pipeline_rejects_missing_dataset(
    settings: Settings,
):
    settings.data.path = "missing/customer_churn.csv"

    with pytest.raises(
        DataError,
        match="Dataset file not found",
    ):
        run_training_pipeline(settings)


def test_persisted_model_can_be_loaded_and_predicts_same_results(
    settings: Settings,
):
    """Verify the persisted pipeline produces identical predictions."""
    result = run_training_pipeline(settings)

    loaded_model = load_model(result.model_path)

    dataframe = load_dataset(settings.data.path)

    _, X_test, _, _ = split_dataset(
        dataframe=dataframe,
        target_column=settings.data.target_column,
        settings=settings.split,
    )

    original_predictions = result.model.predict(X_test)
    loaded_predictions = loaded_model.predict(X_test)

    assert list(loaded_predictions) == list(original_predictions)


def test_loaded_model_preserves_probability_predictions(
    settings: Settings,
):
    """Verify the persisted pipeline preserves probability predictions."""
    result = run_training_pipeline(settings)

    loaded_model = load_model(result.model_path)

    dataframe = load_dataset(settings.data.path)

    _, X_test, _, _ = split_dataset(
        dataframe=dataframe,
        target_column=settings.data.target_column,
        settings=settings.split,
    )

    original_probabilities = result.model.predict_proba(X_test)
    loaded_probabilities = loaded_model.predict_proba(X_test)

    assert original_probabilities.shape == loaded_probabilities.shape
    assert original_probabilities.tolist() == loaded_probabilities.tolist()