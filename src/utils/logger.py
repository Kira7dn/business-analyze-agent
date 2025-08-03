"""
Logging utilities
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str, level: int = logging.INFO, format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger with consistent formatting

    Args:
        name: Logger name
        level: Logging level
        format_string: Custom format string

    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger
