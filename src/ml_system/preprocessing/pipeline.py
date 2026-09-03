from __future__ import annotations

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_system.config.schemas import PreprocessingSettings
from ml_system.exceptions.errors import DataError


logger = logging.getLogger("ml_system")


def build_preprocessing_pipeline(
    dataframe: pd.DataFrame,
    target_column: str,
    settings: PreprocessingSettings,
) -> ColumnTransformer:
    """
    Build a preprocessing pipeline from configuration.

    Numerical features receive missing-value imputation and,
    when configured, standard scaling.

    Categorical features receive missing-value imputation and,
    when configured, one-hot encoding.

    The target column is excluded from preprocessing.

    Parameters
    ----------
    dataframe:
        Input dataset used to identify feature types.
    target_column:
        Name of the target column.
    settings:
        Preprocessing configuration.

    Returns
    -------
    ColumnTransformer
        Configured preprocessing transformer.

    Raises
    ------
    DataError
        If the dataset or preprocessing configuration is invalid.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise DataError("Dataset must be a pandas DataFrame.")

    if dataframe.empty:
        raise DataError("Cannot build preprocessing pipeline for an empty dataset.")

    if target_column not in dataframe.columns:
        raise DataError(
            f"Target column not found in dataset: {target_column}"
        )

    feature_dataframe = dataframe.drop(columns=[target_column])

    if feature_dataframe.shape[1] == 0:
        raise DataError("Dataset contains no features after removing the target.")

    numerical_columns = feature_dataframe.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = feature_dataframe.select_dtypes(
        exclude="number"
    ).columns.tolist()

    numerical_strategy = settings.missing_values.strategy
    encoding_strategy = settings.categorical_encoding.strategy
    scaling_strategy = settings.numerical_scaling.strategy

    if numerical_strategy not in {"mean", "median", "most_frequent"}:
        raise DataError(
            f"Unsupported numerical missing-value strategy: "
            f"{numerical_strategy}"
        )

    if encoding_strategy != "onehot":
        raise DataError(
            f"Unsupported categorical encoding strategy: "
            f"{encoding_strategy}"
        )

    if scaling_strategy not in {"standard", "none"}:
        raise DataError(
            f"Unsupported numerical scaling strategy: "
            f"{scaling_strategy}"
        )

    transformers = []

    if numerical_columns:
        numerical_steps = [
            (
                "imputer",
                SimpleImputer(strategy=numerical_strategy),
            )
        ]

        if scaling_strategy == "standard":
            numerical_steps.append(
                (
                    "scaler",
                    StandardScaler(),
                )
            )

        numerical_pipeline = Pipeline(
            steps=numerical_steps
        )

        transformers.append(
            (
                "numerical",
                numerical_pipeline,
                numerical_columns,
            )
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    if not transformers:
        raise DataError("Dataset contains no usable features.")

    preprocessing_pipeline = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    logger.info(
        "Preprocessing pipeline created: %d numerical features, "
        "%d categorical features",
        len(numerical_columns),
        len(categorical_columns),
    )

    return preprocessing_pipeline