"""Structured logging setup shared across pipelines.

We use structlog so every pipeline step emits key/value events that are easy to
grep in CI logs and to parse into the calibration audit trail later.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure(level: int = logging.INFO) -> None:
    """Configure structlog once, at process entry (pipelines/CLI)."""
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger; call :func:`configure` first at entrypoints."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
