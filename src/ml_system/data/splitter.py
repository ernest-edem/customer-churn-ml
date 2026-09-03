from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_system.config.schemas import SplitSettings
from ml_system.exceptions.errors import DataError


logger = logging.getLogger("ml_system")


def split_dataset(
    dataframe: pd.DataFrame,
    target_column: str,
    settings: SplitSettings,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Split a dataset into training and testing subsets.

    Parameters
    ----------
    dataframe:
        Complete dataset containing features and target.
    target_column:
        Name of the target column.
    settings:
        Configuration controlling test size, random state,
        and stratification.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test.

    Raises
    ------
    DataError
        If the dataset or split configuration is invalid.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise DataError("Dataset must be a pandas DataFrame.")

    if dataframe.empty:
        raise DataError("Cannot split an empty dataset.")

    if target_column not in dataframe.columns:
        raise DataError(
            f"Target column not found in dataset: {target_column}"
        )

    if not 0 < settings.test_size < 1:
        raise DataError(
            "test_size must be between 0 and 1."
        )

    if settings.random_state is not None and not isinstance(
        settings.random_state,
        int,
    ):
        raise DataError(
            "random_state must be an integer or None."
        )

    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]

    if X.shape[1] == 0:
        raise DataError(
            "Dataset contains no features after removing the target."
        )

    if y.isna().any():
        raise DataError(
            "Target column contains missing values."
        )

    if y.nunique() < 2:
        raise DataError(
            "Target column must contain at least two classes."
        )

    stratify = y if settings.stratify else None

    if settings.stratify:
        class_counts = y.value_counts()

        if class_counts.min() < 2:
            raise DataError(
                "Stratified splitting requires at least two "
                "observations in every target class."
            )

    logger.info(
        "Splitting dataset: test_size=%.2f, random_state=%s, "
        "stratify=%s",
        settings.test_size,
        settings.random_state,
        settings.stratify,
    )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=settings.test_size,
            random_state=settings.random_state,
            stratify=stratify,
        )
    except ValueError as exc:
        raise DataError(
            f"Dataset splitting failed: {exc}"
        ) from exc

    logger.info(
        "Dataset split successfully: %d training samples, "
        "%d testing samples",
        len(X_train),
        len(X_test),
    )

    return X_train, X_test, y_train, y_test