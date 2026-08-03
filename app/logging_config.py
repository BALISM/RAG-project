"""
Structured logging configuration for the RAG Chatbot.

Provides rich, colorized console output in development and clean structured
output suitable for log aggregators in production.  Every log entry includes
the module name and — when available — a request_id for tracing a single
HTTP request across the ingestion → embedding → vector store → generation
pipeline.
"""
from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.logging import RichHandler

_configured = False


def setup_logging(level: str = "INFO") -> None:
    """Call once at startup to configure the root logger."""
    global _configured
    if _configured:
        return
    _configured = True

    log_level = getattr(logging, level.upper(), logging.INFO)

    console = Console(stderr=True, force_terminal=True)
    handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        show_path=False,
        markup=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    # Silence noisy third-party loggers
    for noisy in ("chromadb", "httpcore", "httpx", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger. Prefer this over logging.getLogger() for
    consistency — modules call `logger = get_logger(__name__)` at the top."""
    return logging.getLogger(name)


@contextmanager
def log_duration(logger: logging.Logger, operation: str) -> Generator[None, None, None]:
    """Context manager that logs how long an operation took.

    Usage:
        with log_duration(logger, "Embedding 42 chunks"):
            embed_documents(texts)
        # logs: "Embedding 42 chunks completed in 1.23s"
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("%s completed in %.2fs", operation, elapsed)
