from pathlib import Path
from typing import Any

import yaml

from ml_system.config.schemas import (
    DataSettings,
    EncodingSettings,
    EvaluationSettings,
    FeatureSelectionSettings,
    FeatureSettings,
    MissingValueSettings,
    ModelSettings,
    PersistenceSettings,
    PreprocessingSettings,
    ProjectSettings,
    ScalingSettings,
    Settings,
    SplitSettings,
    TaskSettings,
)
from ml_system.exceptions.errors import ConfigurationError


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load configuration from a YAML file."""

    config_path = Path(path)

    if not config_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {config_path}"
        )

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Invalid YAML configuration: {config_path}"
        ) from exc

    if not isinstance(config, dict):
        raise ConfigurationError(
            "Configuration must contain a YAML mapping."
        )

    return config


def build_settings(config: dict[str, Any]) -> Settings:
    """Convert raw configuration into typed settings."""

    try:
        preprocessing = config["preprocessing"]

        return Settings(
            project=ProjectSettings(**config["project"]),
            data=DataSettings(**config["data"]),
            task=TaskSettings(**config["task"]),
            split=SplitSettings(**config["split"]),
            preprocessing=PreprocessingSettings(
                missing_values=MissingValueSettings(
                    **preprocessing["missing_values"]
                ),
                categorical_encoding=EncodingSettings(
                    **preprocessing["categorical_encoding"]
                ),
                numerical_scaling=ScalingSettings(
                    **preprocessing["numerical_scaling"]
                ),
            ),
            features=FeatureSettings(
                selection=FeatureSelectionSettings(
                    **config["features"]["selection"]
                )
            ),
            model=ModelSettings(**config["model"]),
            evaluation=EvaluationSettings(
                **config["evaluation"]
            ),
            persistence=PersistenceSettings(
                **config["persistence"]
            ),
        )

    except (KeyError, TypeError) as exc:
        raise ConfigurationError(
            f"Invalid configuration structure: {exc}"
        ) from exc


def load_settings(path: str | Path) -> Settings:
    """Load and construct application settings."""

    config = load_yaml(path)
    settings = build_settings(config)

    validate_settings(settings)

    return settings

def validate_settings(settings: Settings) -> None:
    """Validate configuration values."""

    if not settings.data.path:
        raise ConfigurationError(
            "Dataset path cannot be empty."
        )

    if not settings.data.target_column:
        raise ConfigurationError(
            "Target column cannot be empty."
        )

    if not 0 < settings.split.test_size < 1:
        raise ConfigurationError(
            "test_size must be between 0 and 1."
        )

    if settings.model.name.strip() == "":
        raise ConfigurationError(
            "Model name cannot be empty."
        )

    if not settings.evaluation.metrics:
        raise ConfigurationError(
            "At least one evaluation metric must be configured."
        )