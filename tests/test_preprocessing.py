import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ml_system.config.schemas import (
    EncodingSettings,
    MissingValueSettings,
    PreprocessingSettings,
    ScalingSettings,
)
from ml_system.data import load_dataset
from ml_system.exceptions.errors import DataError
from ml_system.preprocessing import build_preprocessing_pipeline


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


def test_build_preprocessing_pipeline():
    dataframe = pd.DataFrame(
        {
            "age": [25, 35, 45, None],
            "monthly_charges": [50.0, 75.0, None, 90.0],
            "contract": [
                "Month-to-month",
                "One year",
                "Two year",
                "One year",
            ],
            "churn": [0, 1, 0, 1],
        }
    )

    pipeline = build_preprocessing_pipeline(
        dataframe,
        "churn",
        create_preprocessing_settings(),
    )

    assert isinstance(pipeline, ColumnTransformer)

    pipeline.fit(
        dataframe.drop(columns=["churn"])
    )

    transformed = pipeline.transform(
        dataframe.drop(columns=["churn"])
    )

    assert transformed.shape[0] == 4
    assert transformed.shape[1] == 5


def test_preprocessing_pipeline_handles_unknown_categories():
    dataframe = pd.DataFrame(
        {
            "age": [25, 35, 45],
            "contract": [
                "Month-to-month",
                "One year",
                "Two year",
            ],
            "churn": [0, 1, 0],
        }
    )

    pipeline = build_preprocessing_pipeline(
        dataframe,
        "churn",
        create_preprocessing_settings(),
    )

    pipeline.fit(
        dataframe.drop(columns=["churn"])
    )

    new_data = pd.DataFrame(
        {
            "age": [30],
            "contract": ["Unknown Contract"],
        }
    )

    transformed = pipeline.transform(new_data)

    assert transformed.shape[0] == 1
    assert transformed.shape[1] == 4


def test_preprocessing_pipeline_rejects_missing_target():
    dataframe = pd.DataFrame(
        {
            "age": [25, 35, 45],
            "churn": [0, 1, 0],
        }
    )

    with pytest.raises(
        DataError,
        match="Target column not found",
    ):
        build_preprocessing_pipeline(
            dataframe,
            "customer_status",
            create_preprocessing_settings(),
        )


def test_preprocessing_pipeline_rejects_empty_dataset():
    dataframe = pd.DataFrame()

    with pytest.raises(
        DataError,
        match="empty dataset",
    ):
        build_preprocessing_pipeline(
            dataframe,
            "churn",
            create_preprocessing_settings(),
        )


def test_preprocessing_pipeline_rejects_invalid_encoding_strategy():
    dataframe = pd.DataFrame(
        {
            "age": [25, 35, 45],
            "contract": [
                "Month-to-month",
                "One year",
                "Two year",
            ],
            "churn": [0, 1, 0],
        }
    )

    settings = PreprocessingSettings(
        missing_values=MissingValueSettings(
            strategy="median"
        ),
        categorical_encoding=EncodingSettings(
            strategy="label"
        ),
        numerical_scaling=ScalingSettings(
            strategy="standard"
        ),
    )

    with pytest.raises(
        DataError,
        match="Unsupported categorical encoding strategy",
    ):
        build_preprocessing_pipeline(
            dataframe,
            "churn",
            settings,
        )