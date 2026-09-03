from pathlib import Path

import pandas as pd

from ml_system.exceptions.errors import DataError
from ml_system.logging import setup_logging
import logging


logger = logging.getLogger("ml_system")


SUPPORTED_FORMATS = {
    ".csv": "csv",
}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """
    Load a dataset from a supported file format.

    Parameters
    ----------
    path:
        Path to the dataset file.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.

    Raises
    ------
    DataError
        If the file does not exist, the format is unsupported,
        or the dataset cannot be loaded.
    """

    dataset_path = Path(path)

    if not dataset_path.exists():
        raise DataError(
            f"Dataset file not found: {dataset_path}"
        )

    if not dataset_path.is_file():
        raise DataError(
            f"Dataset path is not a file: {dataset_path}"
        )

    suffix = dataset_path.suffix.lower()

    if suffix not in SUPPORTED_FORMATS:
        supported = ", ".join(SUPPORTED_FORMATS)

        raise DataError(
            f"Unsupported dataset format '{suffix}'. "
            f"Supported formats: {supported}"
        )

    logger.info("Loading dataset: %s", dataset_path)

    try:
        if suffix == ".csv":
            dataframe = pd.read_csv(dataset_path)
        else:
            # This branch should never be reached because the
            # format is checked above.
            raise DataError(
                f"Unsupported dataset format: {suffix}"
            )

    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataError(
            f"Failed to load dataset: {dataset_path}"
        ) from exc

    logger.info(
        "Dataset loaded successfully: %d rows, %d columns",
        dataframe.shape[0],
        dataframe.shape[1],
    )

    return dataframe