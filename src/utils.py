"""
Utility functions shared across the project.

Day 10: Added logging setup for production-ready output.
"""

import logging
import time
from functools import wraps


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Create a configured logger.

    Why a custom logger instead of print()?
    - Loggers have levels (DEBUG, INFO, WARNING, ERROR) — you can filter noise
    - Loggers include timestamps — helpful for debugging in production
    - Loggers can write to files, not just the console
    - In production, you can turn off DEBUG logs without changing code

    Args:
        name: Logger name (usually the module name)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only add handler if the logger doesn't already have one
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def timer(func):
    """
    Decorator that measures and logs function execution time.

    Usage:
        @timer
        def my_slow_function():
            ...

    This will log: "my_slow_function completed in 1.23s"
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger = setup_logger("timer")
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper
