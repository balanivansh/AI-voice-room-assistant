import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.agent.personas import DOST_SYSTEM_PROMPT, SATHI_SYSTEM_PROMPT
from src.agent.router import ROUTER_SYSTEM_PROMPT
from src.agent.worker import VoiceAssistantWorker
from src.audio.tts import AzureSpeechSynthesizer, STATIC_FALLBACK_PCM
from src.state.room_state import TurnPlan


@pytest.mark.asyncio
async def test_livekit_room_text_chat_support():
    """Verify LiveKit text chat handling, logging, routing, and text reply publishing."""
    mock_transcription_handler = AsyncMock()
    mock_room = MagicMock()
    mock_local_participant = AsyncMock()
    mock_room.local_participant = mock_local_participant

    worker = VoiceAssistantWorker(
        transcription_handler=mock_transcription_handler,
    )
    worker.room = mock_room

    # 1. Test handle_text_message
    await worker.handle_text_message(msg="Explain quantum computing", sender="Rahul")

    mock_transcription_handler.normalize_and_dispatch.assert_called_once_with(
        identity="Rahul",
        text="Explain quantum computing",
        source="text",
        target_bot="orchestrator",
    )

    # 2. Test execute_turn_plan text reply publishing back to room chat topic
    turn = TurnPlan(target="dost", task="Explain quantum computing in Hinglish")
    
    with patch.object(worker.persona_manager, "generate_reply", new_callable=AsyncMock) as mock_gen, \
         patch.object(worker.tts_synthesizer, "synthesize", new_callable=AsyncMock) as mock_tts, \
         patch.object(worker.dost_publisher, "play_audio", new_callable=AsyncMock) as mock_play:
        
        mock_gen.return_value = "Quantum computing ek advanced computing technology hai..."
        mock_tts.return_value = b"\x00" * 9600
        mock_play.return_value = True

        await worker.execute_turn_plan(turn)

        # Check publish_data was called with topic='chat' and '[DOST]: ...'
        mock_local_participant.publish_data.assert_called_once()
        call_kwargs = mock_local_participant.publish_data.call_args[1]
        assert call_kwargs.get("topic") == "chat"
        assert call_kwargs.get("reliable") is True
        assert call_kwargs.get("payload") == b"[DOST]: Quantum computing ek advanced computing technology hai..."


@pytest.mark.asyncio
async def test_scenario_2_hinglish_enforcement():
    """Verify system prompts enforce natural Indian Hinglish for English questions."""
    assert "HINGLISH LANGUAGE RULE" in DOST_SYSTEM_PROMPT
    assert "technical terms in English" in DOST_SYSTEM_PROMPT
    assert "HINGLISH LANGUAGE RULE" in SATHI_SYSTEM_PROMPT
    assert "technical terms in English" in SATHI_SYSTEM_PROMPT
    assert "HINGLISH LANGUAGE RULE" in ROUTER_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_simulate_tts_failure_flag(monkeypatch):
    """Verify SIMULATE_TTS_FAILURE=true logs provider_failure_simulated and returns static fallback PCM safely."""
    monkeypatch.setenv("SIMULATE_TTS_FAILURE", "true")
    
    events_logged = []
    def mock_log_event(event_name, **kwargs):
        events_logged.append((event_name, kwargs))

    with patch("src.audio.tts.log_structured_event", side_effect=mock_log_event):
        tts = AzureSpeechSynthesizer(speech_key="mock_key", speech_region="eastus")
        pcm_bytes = await tts.synthesize(text="Testing failure simulation", target_bot="dost")

        # Must return valid static fallback PCM bytes (non-zero len) without crashing
        assert len(pcm_bytes) == len(STATIC_FALLBACK_PCM)
        assert len(pcm_bytes) == 48000

        # Verify provider_failure_simulated event was logged
        simulated_events = [e for e in events_logged if e[0] == "provider_failure_simulated"]
        assert len(simulated_events) == 1
        assert simulated_events[0][1].get("provider") == "azure_tts"
        assert simulated_events[0][1].get("target_bot") == "dost"


@pytest.mark.asyncio
async def test_azure_tts_retry_and_chat_fallback():
    """Verify single retry on network error and text posting to room chat fallback."""
    from unittest.mock import AsyncMock, MagicMock
    from src.audio.tts import AzureSpeechSynthesizer, STATIC_FALLBACK_PCM

    mock_room = MagicMock()
    mock_room.local_participant.publish_data = AsyncMock()

    attempts = []

    def mock_sync_synthesize_once():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("Transient connection reset")
        return b"\x01\x02" * 24000

    tts = AzureSpeechSynthesizer(speech_key="mock_key", speech_region="eastus", room=mock_room)
    
    with patch("src.audio.tts.speechsdk.SpeechSynthesizer") as mock_sdk:
        # Test successful retry
        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.reason = MagicMock()
        # First attempt fail, second succeed
        with patch("src.audio.tts._synthesize_azure_speech") as mock_synth:
            mock_synth.side_effect = RuntimeError("Synthesis failed")
            pcm_bytes = await tts.synthesize(text="Fallback response text", target_bot="sathi", room=mock_room)

            assert pcm_bytes == STATIC_FALLBACK_PCM
            mock_room.local_participant.publish_data.assert_called_once()
            call_kwargs = mock_room.local_participant.publish_data.call_args[1]
            assert call_kwargs["topic"] == "lk.chat"
            assert b"Fallback response text" in call_kwargs["payload"]


@pytest.mark.asyncio

async def test_speech_debounce_window_and_connector_delay():
    """Verify sentence debounce applies brief base delay (350ms) and extended delay for trailing connectors."""
    from src.audio.transcription_handler import UtteranceNormalizer

    worker = VoiceAssistantWorker()
    utt_connector = UtteranceNormalizer(identity="Rahul", text="Cloud computing kya hai aur", source="voice")

    delays = []
    async def mock_process_debounced(identity, source, delay_sec):
        delays.append(delay_sec)

    with patch.object(worker, "_process_debounced_utterance", side_effect=mock_process_debounced):
        await worker.on_utterance_normalized(utt_connector)
        await asyncio.sleep(0.01)

    assert len(delays) == 1
    assert delays[0] == 1.15  # 0.35s base + 0.80s connector delay

    # Test standard completed sentence (350ms brief base debounce)
    worker_std = VoiceAssistantWorker()
    utt_std = UtteranceNormalizer(identity="Rahul", text="Can you explain cloud computing in detail?", source="voice")

    delays_std = []
    async def mock_process_debounced_std(identity, source, delay_sec):
        delays_std.append(delay_sec)

    with patch.object(worker_std, "_process_debounced_utterance", side_effect=mock_process_debounced_std):
        await worker_std.on_utterance_normalized(utt_std)
        await asyncio.sleep(0.01)

    assert len(delays_std) == 1
    assert delays_std[0] == 0.35  # Brief 350ms base debounce


@pytest.mark.asyncio
async def test_alias_normalization_and_fragment_debounce():
    """Verify alias normalization routes dosth -> dost and saathi -> sathi, and fragment debounce holds for 600ms."""
    from src.agent.router import RouterAgent
    from src.audio.transcription_handler import UtteranceNormalizer

    # 1. Test alias normalization in router
    router = RouterAgent(api_key="mock_key")
    with patch("src.agent.router.safe_external_call", new_callable=AsyncMock) as mock_call:
        from src.state.room_state import RouterOutput, TurnPlan
        mock_call.return_value = RouterOutput(
            new_speaker_facts=[],
            salient_entity="general",
            turns=[TurnPlan(target="dost", task="Answer user query")]
        )
        out = await router.route_utterance(
            speaker_identity="Neha",
            text="AI Dosth mujhe bolo",
            speaker_facts={},
            active_topic=None,
            history=[],
        )
        assert out.turns[0].target == "dost"

        out_sathi = await router.route_utterance(
            speaker_identity="Neha",
            text="AI Saathi batao na",
            speaker_facts={},
            active_topic=None,
            history=[],
        )
        assert out_sathi.turns[0].target == "sathi"

    # 2. Test speech fragment debounce (< 3 words ending with "कि", "है", "साल", "और")
    worker = VoiceAssistantWorker()
    utt_fragment = UtteranceNormalizer(identity="Rahul", text="Naam hai", source="voice")

    delays_frag = []
    async def mock_process_debounced_frag(identity, source, delay_sec):
        delays_frag.append(delay_sec)

    with patch.object(worker, "_process_debounced_utterance", side_effect=mock_process_debounced_frag):
        await worker.on_utterance_normalized(utt_fragment)
        await asyncio.sleep(0.01)

    assert len(delays_frag) == 1
    assert delays_frag[0] == 0.60  # 600ms hold for fragment <3 words ending with incomplete word


@pytest.mark.asyncio
async def test_text_chat_deduplication_and_compound_routing():
    """Verify text chat deduplication ignores identical messages within 4s and compound routing preserves two turn plans."""
    from src.agent.router import RouterAgent, normalize_phonetic_aliases

    # 1. Test text chat deduplication
    mock_transcription_handler = AsyncMock()
    worker = VoiceAssistantWorker(transcription_handler=mock_transcription_handler)

    await worker.handle_text_message('{"message": "Hello dost"}', sender="Neha")
    assert mock_transcription_handler.normalize_and_dispatch.call_count == 1

    # Immediate duplicate call within 4s should be ignored
    await worker.handle_text_message("Hello dost", sender="Neha")
    assert mock_transcription_handler.normalize_and_dispatch.call_count == 1

    # 2. Test phonetic alias normalization regex
    assert normalize_phonetic_aliases("yahi saath hi batao") == "AI Sathi batao"
    assert normalize_phonetic_aliases("ai saal explain karo") == "AI Sathi explain karo"
    assert normalize_phonetic_aliases("ai dosth tum bolo") == "AI Dost tum bolo"

    # 3. Test compound multi-bot routing output preservation
    router = RouterAgent(api_key="mock_key")
    with patch("src.agent.router.safe_external_call", new_callable=AsyncMock) as mock_call:
        from src.state.room_state import RouterOutput, TurnPlan
        mock_call.return_value = RouterOutput(
            new_speaker_facts=[],
            salient_entity="general",
            turns=[
                TurnPlan(target="sathi", task="Explain X"),
                TurnPlan(target="dost", task="Explain Y"),
            ]
        )
        out = await router.route_utterance(
            speaker_identity="Neha",
            text="AI Sathi tum batao X, aur AI Dost tum batao Y",
            speaker_facts={},
            active_topic=None,
            history=[],
        )
        assert len(out.turns) == 2
        assert out.turns[0].target == "sathi"
        assert out.turns[1].target == "dost"

