from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from ml_system.config.schemas import (
    EncodingSettings,
    FeatureSelectionSettings,
    MissingValueSettings,
    ModelSettings,
    PreprocessingSettings,
    ScalingSettings,
)
from ml_system.exceptions.errors import ModelError
from ml_system.models import build_model
from ml_system.persistence import load_model, save_model
from ml_system.training import build_training_pipeline, train_model


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


def create_trained_model() -> Pipeline:
    dataframe = create_dataset()

    model = build_model(
        ModelSettings(
            name="random_forest",
            parameters={
                "n_estimators": 10,
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


def test_save_model(tmp_path: Path):
    model = create_trained_model()
    model_path = tmp_path / "models" / "model.joblib"

    save_model(model, model_path)

    assert model_path.exists()
    assert model_path.is_file()


def test_load_model(tmp_path: Path):
    model = create_trained_model()
    model_path = tmp_path / "model.joblib"

    save_model(model, model_path)

    loaded_model = load_model(model_path)

    assert isinstance(loaded_model, Pipeline)

    predictions = loaded_model.predict(
        create_dataset().drop(columns=["churn"])
    )

    assert len(predictions) == len(create_dataset())


def test_save_and_load_preserves_predictions(tmp_path: Path):
    model = create_trained_model()
    dataframe = create_dataset()

    original_predictions = model.predict(
        dataframe.drop(columns=["churn"])
    )

    model_path = tmp_path / "model.joblib"

    save_model(model, model_path)
    loaded_model = load_model(model_path)

    loaded_predictions = loaded_model.predict(
        dataframe.drop(columns=["churn"])
    )

    assert list(original_predictions) == list(
        loaded_predictions
    )


def test_save_model_creates_parent_directory(tmp_path: Path):
    model = create_trained_model()

    model_path = (
        tmp_path
        / "nested"
        / "models"
        / "model.joblib"
    )

    save_model(model, model_path)

    assert model_path.exists()


def test_save_model_rejects_invalid_object(tmp_path: Path):
    model_path = tmp_path / "model.joblib"

    with pytest.raises(
        ModelError,
        match="Only scikit-learn Pipeline",
    ):
        save_model(
            model="not a model",
            path=model_path,
        )


def test_save_model_rejects_invalid_extension(tmp_path: Path):
    model = create_trained_model()
    model_path = tmp_path / "model.pkl"

    with pytest.raises(
        ModelError,
        match=r"\.joblib",
    ):
        save_model(model, model_path)


def test_load_model_rejects_missing_file(tmp_path: Path):
    model_path = tmp_path / "missing.joblib"

    with pytest.raises(
        ModelError,
        match="Model file not found",
    ):
        load_model(model_path)