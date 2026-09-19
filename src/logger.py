import logging
import sys
import traceback
from typing import Any, Literal
import structlog

BotTarget = Literal["dost", "sathi", "orchestrator"]


def configure_logger(log_level: str = "INFO") -> None:
    """Configure structlog with JSON rendering for structured logging."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Silence verbose HTTP/network loggers
    for noisy_logger in ["groq", "httpx", "httpcore", "urllib3"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(ensure_ascii=False),
    ]

    structlog.configure(
        processors=processors,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=False,
    )



# Initial default configuration
configure_logger()

logger: structlog.stdlib.BoundLogger = structlog.get_logger("roxstar_voice_assistant")


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger instance for a given module name."""
    if name:
        return structlog.get_logger(name)
    return logger


def log_structured_event(
    event_name: str,
    target_bot: BotTarget = "orchestrator",
    latency_ms: float | None = None,
    fallback_used: bool = False,
    error: Exception | None = None,
    level: str = "info",
    **extra_context: Any,
) -> None:
    """Log a structured telemetry event with latency tracking and metadata.

    Args:
        event_name: Name of the event (e.g., 'stt_latency', 'llm_ttfb', 'tts_latency').
        target_bot: Target bot component ('dost', 'sathi', or 'orchestrator').
        latency_ms: Execution duration in milliseconds.
        fallback_used: Flag indicating whether fallback execution logic was invoked.
        error: Optional Exception object if an error occurred.
        level: Logging severity ('debug', 'info', 'warning', 'error', 'critical').
        **extra_context: Additional key-value pairs to record in structured log.
    """
    log_data: dict[str, Any] = {
        "event_name": event_name,
        "target_bot": target_bot,
        "fallback_used": fallback_used,
    }

    if latency_ms is not None:
        log_data["latency_ms"] = round(latency_ms, 3)

    if error is not None:
        log_data["error_type"] = type(error).__name__
        log_data["error_message"] = str(error)
        log_data["traceback"] = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )

    log_data.update(extra_context)

    bound_logger = get_logger()
    log_method = getattr(bound_logger, level.lower(), bound_logger.info)
    log_method(event_name, **log_data)
