from __future__ import annotations

import logging
from typing import Any

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from ml_system.config.schemas import ModelSettings
from ml_system.exceptions.errors import ModelError


logger = logging.getLogger("ml_system")


MODEL_REGISTRY: dict[str, type[Any]] = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


def build_model(settings: ModelSettings) -> Any:
    """
    Build a machine learning model from configuration.

    Parameters
    ----------
    settings:
        Model configuration containing the model name and parameters.

    Returns
    -------
    Any
        Instantiated scikit-learn model.

    Raises
    ------
    ModelError
        If the model name is unsupported or model parameters are invalid.
    """

    if not isinstance(settings.name, str) or not settings.name.strip():
        raise ModelError("Model name cannot be empty.")

    model_name = settings.name.strip().lower()

    model_class = MODEL_REGISTRY.get(model_name)

    if model_class is None:
        supported_models = ", ".join(
            sorted(MODEL_REGISTRY.keys())
        )

        raise ModelError(
            f"Unsupported model '{settings.name}'. "
            f"Supported models: {supported_models}"
        )

    parameters = settings.parameters or {}

    if not isinstance(parameters, dict):
        raise ModelError("Model parameters must be a dictionary.")

    try:
        model = model_class(**parameters)
    except (TypeError, ValueError) as exc:
        raise ModelError(
            f"Invalid parameters for model '{model_name}': {exc}"
        ) from exc

    logger.info(
        "Model created successfully: %s",
        model_name,
    )

    return model