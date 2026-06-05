"""Logging setup helpers."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a configured project logger.

    Args:
        name: Logger name, usually ``__name__``.

    Returns:
        Configured logger instance.

    Raises:
        NotImplementedError: Placeholder until logging is standardized.
    """
    raise NotImplementedError("Logger setup is not implemented yet.")
