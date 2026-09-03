from __future__ import annotations

import logging

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from ml_system.config.schemas import (
    FeatureSelectionSettings,
    PreprocessingSettings,
)
from ml_system.exceptions.errors import DataError, ModelError
from ml_system.features import build_feature_selector
from ml_system.preprocessing import build_preprocessing_pipeline


logger = logging.getLogger("ml_system")


def build_training_pipeline(
    dataframe: pd.DataFrame,
    target_column: str,
    preprocessing_settings: PreprocessingSettings,
    feature_selection_settings: FeatureSelectionSettings,
    model: BaseEstimator,
) -> Pipeline:
    """
    Build the complete preprocessing, feature-selection, and model pipeline.

    The returned pipeline is unfitted.

    Parameters
    ----------
    dataframe:
        Training dataframe used to determine feature structure.
    target_column:
        Name of the target column.
    preprocessing_settings:
        Preprocessing configuration.
    feature_selection_settings:
        Feature-selection configuration.
    model:
        Configured machine-learning estimator.

    Returns
    -------
    Pipeline
        Unfitted training pipeline.

    Raises
    ------
    DataError
        If the training data is invalid.
    ModelError
        If the model is invalid.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise DataError("Training data must be a pandas DataFrame.")

    if dataframe.empty:
        raise DataError("Training data cannot be empty.")

    if target_column not in dataframe.columns:
        raise DataError(
            f"Target column not found in training data: {target_column}"
        )

    if not isinstance(model, BaseEstimator):
        raise ModelError(
            "Model must be a valid scikit-learn estimator."
        )

    features = dataframe.drop(columns=[target_column])

    if features.shape[1] == 0:
        raise DataError(
            "Training data contains no features after removing the target."
        )

    preprocessing_pipeline = build_preprocessing_pipeline(
        dataframe=dataframe,
        target_column=target_column,
        settings=preprocessing_settings,
    )

    preprocessing_pipeline.fit(features)

    transformed_feature_count = preprocessing_pipeline.transform(
        features
    ).shape[1]

    feature_selector = build_feature_selector(
        settings=feature_selection_settings,
        feature_count=transformed_feature_count,
    )

    steps: list[tuple[str, object]] = [
        ("preprocessing", preprocessing_pipeline)
    ]

    if feature_selector is not None:
        steps.append(
            ("feature_selection", feature_selector)
        )

    steps.append(("model", model))

    pipeline = Pipeline(steps=steps)

    logger.info(
        "Training pipeline created with %d steps.",
        len(steps),
    )

    return pipeline


def train_model(
    pipeline: Pipeline,
    training_data: pd.DataFrame,
    target_column: str,
) -> Pipeline:
    """
    Fit the training pipeline using training data.

    Parameters
    ----------
    pipeline:
        Unfitted machine-learning pipeline.
    training_data:
        Training dataframe.
    target_column:
        Name of the target column.

    Returns
    -------
    Pipeline
        Fitted training pipeline.

    Raises
    ------
    DataError
        If training data is invalid.
    ModelError
        If training fails.
    """

    if not isinstance(pipeline, Pipeline):
        raise ModelError(
            "Training pipeline must be a scikit-learn Pipeline."
        )

    if not isinstance(training_data, pd.DataFrame):
        raise DataError(
            "Training data must be a pandas DataFrame."
        )

    if training_data.empty:
        raise DataError(
            "Training data cannot be empty."
        )

    if target_column not in training_data.columns:
        raise DataError(
            f"Target column not found in training data: {target_column}"
        )

    X_train = training_data.drop(
        columns=[target_column]
    )
    y_train = training_data[target_column]

    if y_train.isna().any():
        raise DataError(
            "Training target contains missing values."
        )

    if y_train.nunique() < 2:
        raise DataError(
            "Training target must contain at least two classes."
        )

    logger.info(
        "Starting model training: %d samples, %d features",
        X_train.shape[0],
        X_train.shape[1],
    )

    try:
        pipeline.fit(X_train, y_train)
    except (ValueError, TypeError) as exc:
        raise ModelError(
            f"Model training failed: {exc}"
        ) from exc

    logger.info("Model training completed successfully.")

    return pipeline