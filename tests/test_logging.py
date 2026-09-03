import logging

from ml_system.logging import setup_logging


def test_setup_logging():
    setup_logging("config/logging.yaml")

    logger = logging.getLogger("ml_system")

    assert logger.level == logging.INFO
    assert logger.propagate is False