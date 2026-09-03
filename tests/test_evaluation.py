import pandas as pd
import pytest

from ml_system.config.schemas import (
    EncodingSettings,
    FeatureSelectionSettings,
    MissingValueSettings,
    ModelSettings,
    PreprocessingSettings,
    ScalingSettings,
)
from ml_system.evaluation import evaluate_model
from ml_system.exceptions.errors import DataError, EvaluationError
from ml_system.models import build_model
from ml_system.training import (
    build_training_pipeline,
    train_model,
)


def create_settings() -> PreprocessingSettings:
    return PreprocessingSettings(
        missing_values=MissingValueSettings(
            strategy="median"
        ),
        categorical_encoding=EncodingSettings(
            strategy="onehot"
        ),
        numerical_scaling=ScalingSettings(
            strategy="standard"
        ),
    )


def create_feature_settings() -> FeatureSelectionSettings:
    return FeatureSelectionSettings(
        enabled=False,
        method="mutual_information",
        top_k=3,
    )


def create_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [
                25, 31, 42, 28, 56, 63,
                37, 49, 22, 71, 34, 45,
                29, 52, 40, 67, 33, 58,
                24, 47,
            ],
            "monthly_charges": [
                50.0, 65.0, 80.0, 55.0, 95.0, 110.0,
                70.0, 85.0, 45.0, 120.0, 60.0, 90.0,
                52.0, 100.0, 72.0, 115.0, 62.0, 98.0,
                48.0, 88.0,
            ],
            "contract": [
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
                "One year",
                "Month-to-month",
                "Two year",
                "One year",
                "Month-to-month",
                "Month-to-month",
                "Two year",
                "One year",
                "Two year",
                "Month-to-month",
                "One year",
                "Month-to-month",
                "Two year",
            ],
            "churn": [
                0, 0, 1, 0, 1, 1,
                0, 1, 0, 1, 0, 1,
                0, 1, 0, 1, 0, 1,
                0, 0,
            ],
        }
    )


def create_trained_pipeline(
    dataframe: pd.DataFrame,
):
    model = build_model(
        ModelSettings(
            name="random_forest",
            parameters={
                "n_estimators": 20,
                "random_state": 42,
            },
        )
    )

    pipeline = build_training_pipeline(
        dataframe=dataframe,
        target_column="churn",
        preprocessing_settings=create_settings(),
        feature_selection_settings=create_feature_settings(),
        model=model,
    )

    return train_model(
        pipeline=pipeline,
        training_data=dataframe,
        target_column="churn",
    )


def test_evaluate_model_returns_configured_metrics():
    dataframe = create_dataset()
    pipeline = create_trained_pipeline(dataframe)

    results = evaluate_model(
        pipeline=pipeline,
        test_data=dataframe,
        target_column="churn",
        metrics=[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ],
    )

    assert set(
        [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "confusion_matrix",
            "classification_report",
        ]
    ).issubset(results.keys())

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]:
        assert 0.0 <= results[metric] <= 1.0

    assert isinstance(
        results["confusion_matrix"],
        list,
    )

    assert isinstance(
        results["classification_report"],
        str,
    )


def test_evaluate_model_supports_individual_metrics():
    dataframe = create_dataset()
    pipeline = create_trained_pipeline(dataframe)

    results = evaluate_model(
        pipeline=pipeline,
        test_data=dataframe,
        target_column="churn",
        metrics=["accuracy"],
    )

    assert "accuracy" in results
    assert "precision" not in results
    assert "recall" not in results
    assert "f1" not in results
    assert "roc_auc" not in results

    assert "confusion_matrix" in results
    assert "classification_report" in results


def test_evaluate_model_normalizes_metric_names():
    dataframe = create_dataset()
    pipeline = create_trained_pipeline(dataframe)

    results = evaluate_model(
        pipeline=pipeline,
        test_data=dataframe,
        target_column="churn",
        metrics=[" Accuracy "],
    )

    assert "accuracy" in results


def test_evaluate_model_rejects_empty_metrics():
    dataframe = create_dataset()
    pipeline = create_trained_pipeline(dataframe)

    with pytest.raises(
        EvaluationError,
        match="At least one evaluation metric",
    ):
        evaluate_model(
            pipeline=pipeline,
            test_data=dataframe,
            target_column="churn",
            metrics=[],
        )


def test_evaluate_model_rejects_unsupported_metric():
    dataframe = create_dataset()
    pipeline = create_trained_pipeline(dataframe)

    with pytest.raises(
        EvaluationError,
        match="Unsupported evaluation metric",
    ):
        evaluate_model(
            pipeline=pipeline,
            test_data=dataframe,
            target_column="churn",
            metrics=["mae"],
        )


def test_evaluate_model_rejects_missing_target():
    dataframe = create_dataset()
    pipeline = create_trained_pipeline(dataframe)

    with pytest.raises(
        DataError,
        match="Target column not found",
    ):
        evaluate_model(
            pipeline=pipeline,
            test_data=dataframe,
            target_column="customer_status",
            metrics=["accuracy"],
        )


def test_evaluate_model_rejects_missing_test_target():
    dataframe = create_dataset()
    pipeline = create_trained_pipeline(dataframe)

    dataframe.loc[0, "churn"] = None

    with pytest.raises(
        DataError,
        match="Test target contains missing values",
    ):
        evaluate_model(
            pipeline=pipeline,
            test_data=dataframe,
            target_column="churn",
            metrics=["accuracy"],
        )