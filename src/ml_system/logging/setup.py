import logging.config
from pathlib import Path

import yaml

from ml_system.exceptions.errors import ConfigurationError


def setup_logging(config_path: str | Path) -> None:
    """Configure application logging from a YAML file."""

    path = Path(config_path)

    if not path.exists():
        raise ConfigurationError(
            f"Logging configuration file not found: {path}"
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        logging.config.dictConfig(config)

    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Invalid logging configuration: {path}"
        ) from exc