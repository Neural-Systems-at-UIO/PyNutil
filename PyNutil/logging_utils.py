"""Logging configuration utilities for PyNutil.

This module provides centralized logging setup, extracted from the PyNutil class
to follow single-responsibility principle. Logging configuration should happen
at the application level, not inside data processing classes.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union


# Module-level logger for PyNutil
_pynutil_logger: Optional[logging.Logger] = None


def configure_logging(
    log_file: Optional[Union[str, Path]] = "nutil.log",
    file_level: int = logging.DEBUG,
    console_level: int = logging.INFO,
    log_format: str = "%(asctime)s %(levelname)s %(name)s: %(message)s",
) -> logging.Logger:
    """Configure logging for PyNutil.

    This function sets up both file and console logging handlers for the
    PyNutil logger. It should be called once at application startup.

    Parameters
    ----------
    log_file : str or Path, optional
        Path to the log file. Set to None to disable file logging.
        Default is "nutil.log".
    file_level : int, optional
        Logging level for file handler. Default is logging.DEBUG.
    console_level : int, optional
        Logging level for console handler. Default is logging.INFO.
    log_format : str, optional
        Format string for log messages.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Examples
    --------
    >>> from PyNutil.logging_utils import configure_logging
    >>> logger = configure_logging(log_file="my_analysis.log")
    >>> logger.info("Starting analysis...")
    """
    global _pynutil_logger

    logger = logging.getLogger("PyNutil")
    logger.setLevel(logging.DEBUG)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        file_handler = logging.FileHandler(str(log_file))
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Don't propagate to root logger
    logger.propagate = False

    _pynutil_logger = logger
    return logger
