from __future__ import annotations

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name."""
    return logging.getLogger(name)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a human-readable format to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logging.basicConfig(level=level, handlers=[handler])
