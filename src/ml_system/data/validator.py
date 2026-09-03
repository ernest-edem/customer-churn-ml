from __future__ import annotations

import logging

import pandas as pd

from ml_system.exceptions.errors import DataError


logger = logging.getLogger("ml_system")


def validate_dataset(
    dataframe: pd.DataFrame,
    target_column: str,
) -> None:
    """
    Validate the basic structure and target column of a dataset.

    Parameters
    ----------
    dataframe:
        Dataset to validate.
    target_column:
        Name of the configured target column.

    Raises
    ------
    DataError
        If the dataset is invalid.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise DataError("Dataset must be a pandas DataFrame.")

    if dataframe.empty:
        raise DataError("Dataset is empty.")

    if dataframe.columns.empty:
        raise DataError("Dataset contains no columns.")

    if not target_column.strip():
        raise DataError("Target column cannot be empty.")

    if target_column not in dataframe.columns:
        raise DataError(
            f"Target column not found in dataset: {target_column}"
        )

    target = dataframe[target_column]

    if target.isna().all():
        raise DataError(
            f"Target column contains no valid observations: {target_column}"
        )

    logger.info(
        "Dataset validation successful: %d rows, %d columns, target='%s'",
        dataframe.shape[0],
        dataframe.shape[1],
        target_column,
    )