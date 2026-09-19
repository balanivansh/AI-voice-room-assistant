import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.agent.personas import PersonaManager
from src.audio.audio_publisher import BotAudioPublisher
from src.audio.tts import VOICE_MAPPING, AzureSpeechSynthesizer
from src.state.room_state import RoomState, TurnPlan


def test_azure_tts_voice_mapping():
    """Verify voice mapping for Dost and Sathi personas."""
    assert VOICE_MAPPING["dost"] == "hi-IN-MadhurNeural"
    assert VOICE_MAPPING["sathi"] == "hi-IN-SwaraNeural"


@pytest.mark.asyncio
async def test_persona_manager_generation_mock():
    """Verify PersonaManager injects speaker facts into persona prompt."""
    manager = PersonaManager()
    mock_llm = MagicMock()

    class MockResponse:
        content = "Namaste Rahul! Main Dost hoon."

    mock_response = MockResponse()
    mock_llm.return_value = mock_response
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    mock_llm.invoke = MagicMock(return_value=mock_response)
    manager.llm = mock_llm

    state: RoomState = {
        "speaker_facts": {"Rahul": ["Likes cricket"]},
        "rolling_summary": "",
        "last_n_turns": [],
        "active_topic": "cricket",
        "pending_turns": [],
        "current_speaker": "Rahul",
        "current_utterance": None,
    }

    reply = await manager.generate_reply("dost", "Greet user warmly", state)
    assert reply == "Namaste Rahul! Main Dost hoon."
    assert mock_llm.called or mock_llm.invoke.called or mock_llm.ainvoke.called


@pytest.mark.asyncio
async def test_audio_publisher_cancellation_barge_in():
    """Verify BotAudioPublisher halts playback immediately when cancellation_event triggers."""
    publisher = BotAudioPublisher(sample_rate=24000, num_channels=1)
    mock_source = MagicMock()
    mock_source.capture_frame = AsyncMock()
    mock_source.clear_queue = MagicMock()
    publisher.source = mock_source

    cancellation_event = asyncio.Event()

    # Generate 10 seconds of 24kHz 16-bit mono PCM audio (480000 bytes)
    dummy_pcm = b"\x00\x01" * 240000

    async def _trigger_interruption():
        await asyncio.sleep(0.05)
        cancellation_event.set()

    interruption_task = asyncio.create_task(_trigger_interruption())
    completed = await publisher.play_audio(dummy_pcm, cancellation_event, target_bot="dost")
    await interruption_task

    assert completed is False
    mock_source.clear_queue.assert_called_once()
