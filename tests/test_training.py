import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from ml_system.config.schemas import (
    EncodingSettings,
    FeatureSelectionSettings,
    MissingValueSettings,
    PreprocessingSettings,
    ScalingSettings,
)
from ml_system.exceptions.errors import DataError
from ml_system.models import build_model
from ml_system.training import (
    build_training_pipeline,
    train_model,
)


def create_preprocessing_settings() -> PreprocessingSettings:
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


def create_feature_selection_settings() -> FeatureSelectionSettings:
    return FeatureSelectionSettings(
        enabled=True,
        method="mutual_information",
        top_k=3,
    )


def create_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [
                25, 31, 42, 28, 56, 63,
                37, 49, 22, 71, 34, 45,
            ],
            "monthly_charges": [
                50.0, 65.0, 80.0, 55.0, 95.0, 110.0,
                70.0, 85.0, 45.0, 120.0, 60.0, 90.0,
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
            ],
            "churn": [
                0, 0, 1, 0, 1, 1,
                0, 1, 0, 1, 0, 1,
            ],
        }
    )


def create_model() -> RandomForestClassifier:
    return build_model(
        settings=__import__(
            "ml_system.config.schemas",
            fromlist=["ModelSettings"],
        ).ModelSettings(
            name="random_forest",
            parameters={
                "n_estimators": 10,
                "random_state": 42,
            },
        )
    )


def test_build_training_pipeline():
    dataframe = create_dataset()

    pipeline = build_training_pipeline(
        dataframe=dataframe,
        target_column="churn",
        preprocessing_settings=create_preprocessing_settings(),
        feature_selection_settings=create_feature_selection_settings(),
        model=create_model(),
    )

    assert isinstance(pipeline, Pipeline)
    assert "preprocessing" in pipeline.named_steps
    assert "feature_selection" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_build_training_pipeline_without_feature_selection():
    dataframe = create_dataset()

    settings = FeatureSelectionSettings(
        enabled=False,
        method="mutual_information",
        top_k=3,
    )

    pipeline = build_training_pipeline(
        dataframe=dataframe,
        target_column="churn",
        preprocessing_settings=create_preprocessing_settings(),
        feature_selection_settings=settings,
        model=create_model(),
    )

    assert isinstance(pipeline, Pipeline)
    assert "preprocessing" in pipeline.named_steps
    assert "feature_selection" not in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_train_model():
    dataframe = create_dataset()

    pipeline = build_training_pipeline(
        dataframe=dataframe,
        target_column="churn",
        preprocessing_settings=create_preprocessing_settings(),
        feature_selection_settings=create_feature_selection_settings(),
        model=create_model(),
    )

    trained_pipeline = train_model(
        pipeline=pipeline,
        training_data=dataframe,
        target_column="churn",
    )

    assert isinstance(trained_pipeline, Pipeline)
    assert hasattr(trained_pipeline, "predict")

    predictions = trained_pipeline.predict(
        dataframe.drop(columns=["churn"])
    )

    assert len(predictions) == len(dataframe)


def test_train_model_rejects_missing_target():
    dataframe = create_dataset()

    pipeline = build_training_pipeline(
        dataframe=dataframe,
        target_column="churn",
        preprocessing_settings=create_preprocessing_settings(),
        feature_selection_settings=create_feature_selection_settings(),
        model=create_model(),
    )

    with pytest.raises(
        DataError,
        match="Target column not found",
    ):
        train_model(
            pipeline=pipeline,
            training_data=dataframe,
            target_column="customer_status",
        )


def test_train_model_rejects_missing_target_values():
    dataframe = create_dataset()
    dataframe.loc[0, "churn"] = None

    pipeline = build_training_pipeline(
        dataframe=create_dataset(),
        target_column="churn",
        preprocessing_settings=create_preprocessing_settings(),
        feature_selection_settings=create_feature_selection_settings(),
        model=create_model(),
    )

    with pytest.raises(
        DataError,
        match="Training target contains missing values",
    ):
        train_model(
            pipeline=pipeline,
            training_data=dataframe,
            target_column="churn",
        )


def test_train_model_rejects_single_class_target():
    dataframe = create_dataset()
    dataframe["churn"] = 0

    pipeline = build_training_pipeline(
        dataframe=create_dataset(),
        target_column="churn",
        preprocessing_settings=create_preprocessing_settings(),
        feature_selection_settings=create_feature_selection_settings(),
        model=create_model(),
    )

    with pytest.raises(
        DataError,
        match="at least two classes",
    ):
        train_model(
            pipeline=pipeline,
            training_data=dataframe,
            target_column="churn",
        )