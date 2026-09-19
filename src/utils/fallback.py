import asyncio
import time
from typing import Any, Callable
from src.logger import BotTarget, log_structured_event


async def safe_external_call(
    service_name: str,
    func: Callable[..., Any],
    *args: Any,
    fallback_response: Any = None,
    target_bot: BotTarget = "orchestrator",
    **kwargs: Any,
) -> Any:
    """Safely execute external network calls with timing, logging,

    and deterministic fallback responses.

    Args:
        service_name: Identifier for the service being called (e.g. 'Deepgram', 'Groq', 'Azure').
        func: Async or sync callable to execute.
        *args: Positional arguments passed to func.
        fallback_response: Deterministic fallback object returned on failure.
        target_bot: Target bot component ('dost', 'sathi', or 'orchestrator').
        **kwargs: Keyword arguments passed to func.

    Returns:
        Function return value on success, or fallback_response on exception.
    """
    start_time = time.perf_counter()
    try:
        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            res = func(*args, **kwargs)
            if asyncio.iscoroutine(res):
                result = await res
            else:
                result = res

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        log_structured_event(
            event_name=f"{service_name}_call_success",
            target_bot=target_bot,
            latency_ms=elapsed_ms,
            fallback_used=False,
            level="info",
        )
        return result

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        log_structured_event(
            event_name=f"{service_name}_call_error",
            target_bot=target_bot,
            latency_ms=elapsed_ms,
            fallback_used=True,
            error=exc,
            level="warning",
        )
        return fallback_response
