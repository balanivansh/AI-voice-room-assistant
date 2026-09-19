import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.audio.stt import create_stt
from src.audio.transcription_handler import TranscriptionHandler, UtteranceNormalizer
from src.audio.vad import create_vad


def test_utterance_normalizer_fields():
    """Verify UtteranceNormalizer instantiation and dict serialization."""
    item = UtteranceNormalizer(
        identity="user_123",
        text="  Namaste, how are you?  ",
        source="voice",
        timestamp=1000.0,
    )
    assert item.identity == "user_123"
    assert item.text == "  Namaste, how are you?  "
    assert item.source == "voice"
    assert item.timestamp == 1000.0
    data = item.to_dict()
    assert data["identity"] == "user_123"
    assert data["source"] == "voice"


@pytest.mark.asyncio
async def test_transcription_handler_dispatch():
    """Verify TranscriptionHandler sanitizes text, enqueues items, and invokes callbacks."""
    handler = TranscriptionHandler()
    callback_mock = AsyncMock()

    handler.register_callback(callback_mock)

    # Empty text should be ignored
    res_empty = await handler.normalize_and_dispatch(
        identity="user_1",
        text="   ",
        source="text",
    )
    assert res_empty is None
    assert handler.queue_size == 0
    callback_mock.assert_not_called()

    # Valid text should be sanitized, queued, and dispatched to callback
    res_valid = await handler.normalize_and_dispatch(
        identity="user_1",
        text="  Hello Dost  ",
        source="voice",
    )
    assert res_valid is not None
    assert res_valid.text == "Hello Dost"
    assert handler.queue_size == 1

    queued_item = await handler.get_next_utterance()
    assert queued_item.text == "Hello Dost"
    assert queued_item.identity == "user_1"
    assert queued_item.source == "voice"

    callback_mock.assert_called_once()
    assert callback_mock.call_args[0][0].text == "Hello Dost"


def test_create_vad_factory():
    """Verify create_vad returns a valid Silero VAD instance."""
    vad_inst = create_vad(min_speech_duration=0.1)
    assert vad_inst is not None


@pytest.mark.asyncio
async def test_create_stt_factory():
    """Verify create_stt returns a Deepgram STT instance with nova-2 and hi language config."""
    stt_inst = await create_stt(
        model="nova-2",
        language="hi",
        api_key="test_key",
    )
    assert stt_inst is not None
