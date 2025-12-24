"""
Logging Configuration

This module provides centralized logging configuration for the application.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Configure centralized logging for the application.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional path to log file (default: logs/app.log)
        format_string: Custom format string (default: standard format)
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Create handlers
    handlers = [logging.StreamHandler(sys.stdout)]

    # Add file handler if log_file specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    # Configure logging
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers
    )

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Name for the logger (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
