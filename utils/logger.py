"""Structured logger configuration for AskAnything."""
import logging
import os
import sys
from typing import Optional


def setup_logger(name: str = "askanything", level: Optional[str] = None) -> logging.Logger:
    """Configures and returns a structured logger.

    Args:
        name: Name of the logger component.
        level: Optional log level string (DEBUG, INFO, WARNING, ERROR).
               Defaults to LOG_LEVEL from environment or INFO.

    Returns:
        Configured logging.Logger instance.
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    numeric_level = getattr(logging, level, logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if already added
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


# Default application logger
logger = setup_logger("askanything")
