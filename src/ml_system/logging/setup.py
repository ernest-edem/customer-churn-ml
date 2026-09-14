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

        file_handler = config.get("handlers", {}).get("file")
        if file_handler and "filename" in file_handler:
            log_path = Path(file_handler["filename"])
            log_path.parent.mkdir(parents=True, exist_ok=True)

        logging.config.dictConfig(config)

    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Invalid logging configuration: {path}"
        ) from exc