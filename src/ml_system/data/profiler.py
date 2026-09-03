from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ml_system.exceptions.errors import DataError


logger = logging.getLogger("ml_system")


def profile_dataset(
    dataframe: pd.DataFrame,
    target_column: str,
) -> dict[str, Any]:
    """
    Generate a basic profile of a validated dataset.

    Parameters
    ----------
    dataframe:
        Dataset to profile.
    target_column:
        Name of the target column.

    Returns
    -------
    dict[str, Any]
        Dataset profiling information.

    Raises
    ------
    DataError
        If the dataset is not a DataFrame or the target column
        does not exist.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise DataError("Dataset must be a pandas DataFrame.")

    if dataframe.empty:
        raise DataError("Cannot profile an empty dataset.")

    if target_column not in dataframe.columns:
        raise DataError(
            f"Target column not found in dataset: {target_column}"
        )

    numerical_columns = dataframe.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = dataframe.select_dtypes(
        exclude="number"
    ).columns.tolist()

    missing_values = {
        column: int(count)
        for column, count in dataframe.isna().sum().items()
        if count > 0
    }

    target_distribution = {
        str(value): int(count)
        for value, count in dataframe[target_column]
        .value_counts(dropna=False)
        .items()
    }

    profile = {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "column_names": dataframe.columns.tolist(),
        "data_types": {
            column: str(dtype)
            for column, dtype in dataframe.dtypes.items()
        },
        "missing_values": missing_values,
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "target_column": target_column,
        "target_distribution": target_distribution,
    }

    logger.info(
        "Dataset profile generated: %d rows, %d columns, %d duplicate rows",
        profile["rows"],
        profile["columns"],
        profile["duplicate_rows"],
    )

    return profile