from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from ml_system.exceptions.errors import DataError, EvaluationError


logger = logging.getLogger("ml_system")


SUPPORTED_METRICS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
}


def evaluate_model(
    pipeline: Pipeline,
    test_data: pd.DataFrame,
    target_column: str,
    metrics: list[str],
) -> dict[str, Any]:
    """
    Evaluate a trained classification pipeline.

    Parameters
    ----------
    pipeline:
        Fitted scikit-learn pipeline.

    test_data:
        Test dataset containing features and target.

    target_column:
        Name of the target column.

    metrics:
        List of configured evaluation metrics.

    Returns
    -------
    dict[str, Any]
        Evaluation results containing configured metrics,
        confusion matrix, and classification report.

    Raises
    ------
    DataError
        If test data or target configuration is invalid.

    EvaluationError
        If evaluation fails or an unsupported metric is requested.
    """

    if not isinstance(pipeline, Pipeline):
        raise EvaluationError(
            "Evaluation requires a scikit-learn Pipeline."
        )

    if not isinstance(test_data, pd.DataFrame):
        raise DataError(
            "Test data must be a pandas DataFrame."
        )

    if test_data.empty:
        raise DataError(
            "Test data cannot be empty."
        )

    if target_column not in test_data.columns:
        raise DataError(
            f"Target column not found in test data: {target_column}"
        )

    if not metrics:
        raise EvaluationError(
            "At least one evaluation metric must be configured."
        )

    normalized_metrics = [
        metric.strip().lower()
        for metric in metrics
    ]

    unsupported_metrics = [
        metric
        for metric in normalized_metrics
        if metric not in SUPPORTED_METRICS
    ]

    if unsupported_metrics:
        raise EvaluationError(
            "Unsupported evaluation metric(s): "
            + ", ".join(unsupported_metrics)
        )

    X_test = test_data.drop(
        columns=[target_column]
    )
    y_test = test_data[target_column]

    if y_test.isna().any():
        raise DataError(
            "Test target contains missing values."
        )

    if y_test.nunique() < 2:
        raise EvaluationError(
            "Evaluation requires at least two target classes "
            "in the test data."
        )

    try:
        predictions = pipeline.predict(X_test)

        results: dict[str, Any] = {}

        if "accuracy" in normalized_metrics:
            results["accuracy"] = float(
                accuracy_score(
                    y_test,
                    predictions,
                )
            )

        if "precision" in normalized_metrics:
            results["precision"] = float(
                precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            )

        if "recall" in normalized_metrics:
            results["recall"] = float(
                recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            )

        if "f1" in normalized_metrics:
            results["f1"] = float(
                f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            )

        if "roc_auc" in normalized_metrics:
            if not hasattr(pipeline, "predict_proba"):
                raise EvaluationError(
                    "ROC-AUC requires a model that supports "
                    "probability predictions."
                )

            probabilities = pipeline.predict_proba(X_test)

            if probabilities.ndim != 2:
                raise EvaluationError(
                    "ROC-AUC requires a two-dimensional "
                    "probability prediction array."
                )

            if probabilities.shape[1] != 2:
                raise EvaluationError(
                    "ROC-AUC currently requires a binary "
                    "classification problem."
                )

            results["roc_auc"] = float(
                roc_auc_score(
                    y_test,
                    probabilities[:, 1],
                )
            )

        results["confusion_matrix"] = (
            confusion_matrix(
                y_test,
                predictions,
            ).tolist()
        )

        results["classification_report"] = (
            classification_report(
                y_test,
                predictions,
                zero_division=0,
            )
        )

    except EvaluationError:
        raise

    except (ValueError, TypeError, AttributeError) as exc:
        raise EvaluationError(
            f"Model evaluation failed: {exc}"
        ) from exc

    logger.info(
        "Model evaluation completed successfully on %d samples.",
        len(test_data),
    )

    return results