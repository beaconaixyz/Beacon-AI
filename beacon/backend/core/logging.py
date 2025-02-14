"""
Logging configuration for BEACON API

This module sets up logging configuration for the API.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from beacon.backend.core.config import settings

def setup_logging() -> logging.Logger:
    """Setup logging configuration.

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("beacon")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL))

    # Create formatters
    formatter = logging.Formatter(settings.LOG_FORMAT)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler
    file_handler = RotatingFileHandler(
        settings.LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger 