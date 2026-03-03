"""
Structured logging helpers.

Sets up a module-level logger with a consistent format.
Experiments call get_logger(__name__) to get a namespaced logger.
"""

import logging
import sys
from typing import Optional


_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger configured with a StreamHandler to stdout.

    Calling this multiple times with the same name returns the same logger
    (standard Python logging behaviour).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def log_config(logger: logging.Logger, config_dict: dict, title: str = "Config") -> None:
    """Pretty-print a configuration dictionary."""
    logger.info(f"── {title} ──")
    for k, v in config_dict.items():
        logger.info(f"  {k}: {v}")
