"""
Nexia Production Structured Logging & Correlation Architecture.
"""
import contextvars
import logging
import sys
import time
from typing import Any, Dict, Optional

# Context variable to hold request correlation ID across async execution paths
current_request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("current_request_id", default=None)


class StructuredFormatter(logging.Formatter):
    """Formats log records with correlation ID, component tags, and ISO timestamps."""

    def format(self, record: logging.LogRecord) -> str:
        req_id = getattr(record, "request_id", None) or current_request_id.get() or "-"
        component = getattr(record, "component", record.name)
        timestamp = self.formatTime(record, self.datefmt or "%Y-%m-%d %H:%M:%S")
        return f"{timestamp} [{record.levelname:<5}] [{req_id}] [{component}]: {record.getMessage()}"


def setup_logging(level: str = "INFO") -> None:
    """Initialize root and Nexia loggers with structured formatting."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing stream handlers to avoid double output
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(handler)

    # Core loggers
    logging.getLogger("nexia").setLevel(log_level)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(component_name: str) -> logging.LoggerAdapter:
    """Return a logger adapter bound to a specific component."""
    base_logger = logging.getLogger(f"nexia.{component_name}")
    return ComponentLoggerAdapter(base_logger, {"component": component_name})


class ComponentLoggerAdapter(logging.LoggerAdapter):
    """Adapter that injects component name and current correlation ID into log records."""

    def process(self, msg: str, kwargs: Any) -> tuple[str, Any]:
        extra = kwargs.setdefault("extra", {})
        if "request_id" not in extra:
            extra["request_id"] = current_request_id.get() or "-"
        if "component" not in extra and "component" in self.extra:
            extra["component"] = self.extra["component"]
        return msg, kwargs
