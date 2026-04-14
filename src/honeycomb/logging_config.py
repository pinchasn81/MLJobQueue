"""
Logging configuration.

Design choice: stdlib logging over loguru or structlog. The standard library's logging
module is universally understood, requires zero dependencies, and integrates with every
log aggregator (CloudWatch, Datadog, ELK, etc.) out of the box. Third-party loggers add
developer ergonomics but introduce opinions about format and transport that vary by
deployment environment.
"""

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure root logger with a structured, human-readable format.

    Called once at application startup (in the FastAPI lifespan).
    All module loggers inherit this config via the root logger.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers if setup_logging is called more than once (e.g., in tests)
    if not root.handlers:
        root.addHandler(handler)
    else:
        root.handlers.clear()
        root.addHandler(handler)

    # Silence noisy third-party loggers that would flood output at DEBUG level
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
