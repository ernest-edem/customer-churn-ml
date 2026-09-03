from __future__ import annotations

import logging
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from ml_system.exceptions.errors import ModelError


logger = logging.getLogger("ml_system")


def save_model(
    model: Pipeline,
    path: str | Path,
) -> None:
    """
    Persist a trained model pipeline to disk.

    Parameters
    ----------
    model:
        Trained scikit-learn pipeline.
    path:
        Destination path for the serialized model.

    Raises
    ------
    ModelError
        If the model is invalid or cannot be saved.
    """

    if not isinstance(model, Pipeline):
        raise ModelError(
            "Only scikit-learn Pipeline objects can be persisted."
        )

    model_path = Path(path)

    if model_path.suffix.lower() != ".joblib":
        raise ModelError(
            "Model persistence path must use the .joblib extension."
        )

    try:
        model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(model, model_path)

    except (OSError, ValueError, TypeError) as exc:
        raise ModelError(
            f"Failed to save model: {model_path}"
        ) from exc

    logger.info(
        "Model saved successfully: %s",
        model_path,
    )


def load_model(
    path: str | Path,
) -> Pipeline:
    """
    Load a persisted model pipeline from disk.

    Parameters
    ----------
    path:
        Path to the serialized model.

    Returns
    -------
    Pipeline
        Loaded scikit-learn pipeline.

    Raises
    ------
    ModelError
        If the model file does not exist, has an invalid extension,
        or cannot be loaded.
    """

    model_path = Path(path)

    if not model_path.exists():
        raise ModelError(
            f"Model file not found: {model_path}"
        )

    if not model_path.is_file():
        raise ModelError(
            f"Model path is not a file: {model_path}"
        )

    if model_path.suffix.lower() != ".joblib":
        raise ModelError(
            "Model persistence path must use the .joblib extension."
        )

    try:
        model = joblib.load(model_path)

    except (
        OSError,
        ValueError,
        TypeError,
        EOFError,
        AttributeError,
    ) as exc:
        raise ModelError(
            f"Failed to load model: {model_path}"
        ) from exc

    if not isinstance(model, Pipeline):
        raise ModelError(
            "Persisted object is not a scikit-learn Pipeline."
        )

    logger.info(
        "Model loaded successfully: %s",
        model_path,
    )

    return model