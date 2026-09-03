from pathlib import Path

import pandas as pd
import pytest

from ml_system.data import load_dataset
from ml_system.exceptions.errors import DataError


def test_load_csv_dataset(tmp_path: Path):
    dataset_path = tmp_path / "dataset.csv"

    dataframe = pd.DataFrame(
        {
            "age": [25, 35, 45],
            "churn": [0, 1, 0],
        }
    )

    dataframe.to_csv(dataset_path, index=False)

    result = load_dataset(dataset_path)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 2)
    assert list(result.columns) == ["age", "churn"]


def test_load_dataset_file_not_found():
    with pytest.raises(DataError, match="Dataset file not found"):
        load_dataset("data/raw/nonexistent.csv")


def test_load_dataset_unsupported_format(tmp_path: Path):
    dataset_path = tmp_path / "dataset.txt"
    dataset_path.write_text("some data", encoding="utf-8")

    with pytest.raises(DataError, match="Unsupported dataset format"):
        load_dataset(dataset_path)