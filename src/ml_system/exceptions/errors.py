class MLSystemError(Exception):
    """Base exception for the ML system."""


class ConfigurationError(MLSystemError):
    """Raised when configuration is invalid."""


class DataError(MLSystemError):
    """Raised when a dataset cannot be processed."""


class ModelError(MLSystemError):
    """Raised when model creation or training fails."""


class EvaluationError(MLSystemError):
    """Raised when model evaluation fails."""