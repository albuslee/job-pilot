"""structlog configuration; idempotent."""

from __future__ import annotations

import logging
import sys

import structlog

from jobpilot.config import LogFormat


def configure_logging(log_format: LogFormat = "console", level: int = logging.INFO) -> None:
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
    ]
    renderer: structlog.types.Processor = (
        structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
        if log_format == "console"
        else structlog.processors.JSONRenderer()
    )
    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]
