import asyncio
import pytest
from src.config import Settings, get_settings
from src.logger import log_structured_event
from src.utils.fallback import safe_external_call


def test_settings_load():
    """Verify settings instance loads default values cleanly."""
    config = get_settings()
    assert isinstance(config, Settings)
    assert config.livekit_url.startswith("wss://")
    assert config.log_level == "INFO"


def test_log_structured_event(capsys):
    """Verify structured logger formats events into JSON."""
    log_structured_event(
        event_name="stt_latency",
        target_bot="dost",
        latency_ms=123.45,
        fallback_used=False,
    )
    captured = capsys.readouterr()
    assert "stt_latency" in captured.out
    assert "dost" in captured.out
    assert "123.45" in captured.out


def test_log_unicode_devanagari_rendering(capsys):
    """Verify Devanagari Hindi text renders directly without ASCII escaping."""
    hindi_text = "क्या आप मुझे sun सकते हैं?"
    log_structured_event(
        event_name="stt_transcription",
        target_bot="dost",
        text=hindi_text,
    )
    captured = capsys.readouterr()
    assert hindi_text in captured.out
    assert "\\u0915" not in captured.out


@pytest.mark.asyncio
async def test_safe_external_call_success():
    """Verify safe_external_call returns result when inner call succeeds."""
    async def sample_async_func(val: int) -> int:
        await asyncio.sleep(0.01)
        return val * 2

    res = await safe_external_call(
        "Deepgram",
        sample_async_func,
        5,
        fallback_response=0,
        target_bot="dost",
    )
    assert res == 10


@pytest.mark.asyncio
async def test_safe_external_call_failure():
    """Verify safe_external_call catches exception and returns fallback."""
    async def failing_async_func():
        await asyncio.sleep(0.01)
        raise RuntimeError("External API timeout")

    res = await safe_external_call(
        "Groq",
        failing_async_func,
        fallback_response="Fallback LLM response",
        target_bot="sathi",
    )
    assert res == "Fallback LLM response"
