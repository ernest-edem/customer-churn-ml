import pytest
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from ml_system.config.schemas import ModelSettings
from ml_system.exceptions.errors import ModelError
from ml_system.models import build_model


def test_build_logistic_regression():
    settings = ModelSettings(
        name="logistic_regression",
        parameters={
            "max_iter": 500,
            "random_state": 42,
        },
    )

    model = build_model(settings)

    assert isinstance(model, LogisticRegression)
    assert model.max_iter == 500
    assert model.random_state == 42


def test_build_decision_tree():
    settings = ModelSettings(
        name="decision_tree",
        parameters={
            "max_depth": 5,
            "random_state": 42,
        },
    )

    model = build_model(settings)

    assert isinstance(model, DecisionTreeClassifier)
    assert model.max_depth == 5
    assert model.random_state == 42


def test_build_random_forest():
    settings = ModelSettings(
        name="random_forest",
        parameters={
            "n_estimators": 200,
            "random_state": 42,
        },
    )

    model = build_model(settings)

    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 200
    assert model.random_state == 42


def test_build_gradient_boosting():
    settings = ModelSettings(
        name="gradient_boosting",
        parameters={
            "n_estimators": 100,
            "random_state": 42,
        },
    )

    model = build_model(settings)

    assert isinstance(model, GradientBoostingClassifier)
    assert model.n_estimators == 100
    assert model.random_state == 42


def test_build_model_without_parameters():
    settings = ModelSettings(
        name="random_forest",
    )

    model = build_model(settings)

    assert isinstance(model, RandomForestClassifier)


def test_model_names_are_case_insensitive():
    settings = ModelSettings(
        name="Random_Forest",
        parameters={
            "n_estimators": 10,
            "random_state": 42,
        },
    )

    model = build_model(settings)

    assert isinstance(model, RandomForestClassifier)


def test_build_model_rejects_empty_name():
    settings = ModelSettings(
        name="   ",
    )

    with pytest.raises(
        ModelError,
        match="Model name cannot be empty",
    ):
        build_model(settings)


def test_build_model_rejects_unsupported_model():
    settings = ModelSettings(
        name="support_vector_machine",
    )

    with pytest.raises(
        ModelError,
        match="Unsupported model",
    ):
        build_model(settings)


def test_build_model_rejects_invalid_parameters():
    settings = ModelSettings(
        name="random_forest",
        parameters={
            "invalid_parameter": True,
        },
    )

    with pytest.raises(
        ModelError,
        match="Invalid parameters",
    ):
        build_model(settings)


def test_build_model_rejects_non_dictionary_parameters():
    settings = ModelSettings(
        name="random_forest",
        parameters=["invalid"],
    )

    with pytest.raises(
        ModelError,
        match="Model parameters must be a dictionary",
    ):
        build_model(settings)