"""
logger_config.py — Centralized logging configuration for the Sentiment Analysis API.

Provides consistent, structured logging across the application with
separate log levels for development and production.
"""

import logging
import sys
from os import getenv


def configure_logging(app_name: str = "sentiment-api") -> logging.Logger:
    """
    Configure application logging with console output.

    Production: WARNING and above
    Development: INFO and above (set LOG_LEVEL=DEBUG for more verbosity)
    """
    log_level = getenv("LOG_LEVEL", "INFO" if getenv("FLASK_ENV") == "development" else "WARNING")

    logger = logging.getLogger(app_name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Avoid adding duplicate handlers on reload
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logger.level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_logger(name: str = "sentiment-api") -> logging.Logger:
    """Get a configured logger instance."""
    return logging.getLogger(name)
