from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]

RAW_DATASET = (
    PROJECT_ROOT
    / "src"
    / "ml_system"
    / "data"
    / "dataset"
    / "Churn_Dataset.csv"
)

PROCESSED_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_churn_processed.csv"
)


def prepare_dataset() -> None:
    """Prepare the customer churn dataset for model training."""

    dataframe = pd.read_csv(RAW_DATASET)

    if "customerID" in dataframe.columns:
        dataframe = dataframe.drop(columns=["customerID"])

    dataframe["TotalCharges"] = pd.to_numeric(
        dataframe["TotalCharges"],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=["TotalCharges"]
    )

    PROCESSED_DATASET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        PROCESSED_DATASET,
        index=False,
    )

    print(
        f"Processed dataset saved to: {PROCESSED_DATASET}"
    )
    print(
        f"Shape: {dataframe.shape}"
    )


if __name__ == "__main__":
    prepare_dataset()