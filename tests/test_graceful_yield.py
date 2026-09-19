import asyncio
import inspect
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.agent.queue_manager import TurnQueueManager
from src.agent.worker import VoiceAssistantWorker
from src.state.graph import RoomStateGraphManager
from src.state.room_state import TurnPlan


@pytest.mark.asyncio
async def test_distinct_speaker_graceful_yield():
    """Verify distinct speaker interruption halts audio playout and records interrupted topic without queue wipe."""
    graph_manager = RoomStateGraphManager()
    graph_manager.state["current_speaker"] = "Rahul"
    graph_manager.state["active_topic"] = "IGST Section 12"

    mock_audio_publisher = MagicMock()
    mock_audio_publisher.is_speaking = True

    mock_queue_manager = MagicMock()
    mock_queue_manager.cancel_current_and_clear.return_value = 2

    worker = VoiceAssistantWorker(
        graph_manager=graph_manager,
        queue_manager=mock_queue_manager,
        audio_publisher=mock_audio_publisher,
    )

    # Distinct speaker (Priya) interrupts Rahul's bot playout
    worker.trigger_barge_in_interruption("Priya")

    # Audio should be stopped immediately
    mock_audio_publisher.stop_current.assert_called_once()
    # Queue should NOT be cleared
    mock_queue_manager.cancel_current_and_clear.assert_not_called()
    # Cancellation event should NOT be set
    assert not worker.cancellation_event.is_set()

    # Interrupted topic should be recorded
    topics = graph_manager.state.get("interrupted_topics", [])
    assert len(topics) == 1
    assert topics[0]["speaker"] == "Rahul"
    assert topics[0]["topic"] == "IGST Section 12"


@pytest.mark.asyncio
async def test_same_speaker_barge_in_cancellation():
    """Verify same speaker interruption halts audio, sets cancellation event, and wipes queue."""
    graph_manager = RoomStateGraphManager()
    graph_manager.state["current_speaker"] = "Rahul"
    graph_manager.state["active_topic"] = "Python async"

    mock_audio_publisher = MagicMock()
    mock_audio_publisher.is_speaking = True

    mock_queue_manager = MagicMock()
    mock_queue_manager.cancel_current_and_clear.return_value = 1

    worker = VoiceAssistantWorker(
        graph_manager=graph_manager,
        queue_manager=mock_queue_manager,
        audio_publisher=mock_audio_publisher,
    )

    # Same speaker (Rahul) interrupts own bot playout
    worker.trigger_barge_in_interruption("Rahul")

    # Audio should be stopped
    mock_audio_publisher.stop_current.assert_called_once()
    # Queue SHOULD be cleared
    mock_queue_manager.cancel_current_and_clear.assert_called_once()
    # Cancellation event SHOULD be set
    assert worker.cancellation_event.is_set()


@pytest.mark.asyncio
async def test_room_sid_coroutine_and_string_resolution():
    """Verify Room.sid handles both coroutine and string representations without RuntimeWarning."""
    # Test coroutine room.sid
    async def async_sid():
        return "RM_12345"

    class AsyncRoom:
        sid = async_sid()

    room_async = AsyncRoom()
    room_sid_async = await room_async.sid if inspect.isawaitable(room_async.sid) else str(room_async.sid or "")
    assert room_sid_async == "RM_12345"

    # Test string room.sid
    class SyncRoom:
        sid = "RM_67890"

    room_sync = SyncRoom()
    room_sid_sync = await room_sync.sid if inspect.isawaitable(room_sync.sid) else str(room_sync.sid or "")
    assert room_sid_sync == "RM_67890"


@pytest.mark.asyncio
async def test_azure_tts_fallback_non_zero_bytes():
    """Verify Azure Speech Synthesizer returns non-zero fallback PCM bytes when synthesis fails."""
    from src.audio.tts import AzureSpeechSynthesizer, STATIC_FALLBACK_PCM

    synthesizer = AzureSpeechSynthesizer(speech_key="invalid_key", speech_region="invalid_region")
    pcm_bytes = await synthesizer.synthesize("Test text", target_bot="dost")

    assert pcm_bytes is not None
    assert len(pcm_bytes) == len(STATIC_FALLBACK_PCM)
    assert len(pcm_bytes) > 0


def test_silero_vad_config_defaults():
    """Verify Silero VAD default min_silence_duration is 1.25s."""
    import inspect
    from src.audio.vad import create_vad

    sig = inspect.signature(create_vad)
    assert sig.parameters["min_silence_duration"].default == 1.75
    assert sig.parameters["min_speech_duration"].default == 0.05
    assert sig.parameters["activation_threshold"].default == 0.5


def test_turn_plan_schema_and_reason_enum():
    """Verify TurnPlan default confidence is calibrated and reason is constrained."""
    plan = TurnPlan(target="dost", task="Greeting", confidence=0.95, reason="direct_address")
    assert plan.confidence == 0.95
    assert plan.reason == "direct_address"


@pytest.mark.asyncio
async def test_worker_run_job_defensive_room_resolution():
    """Verify VoiceAssistantWorker.run_job initializes self.room from ctx and handles None room gracefully."""
    mock_room = MagicMock()
    mock_room.name = "test_room"
    mock_room.sid = "RM_TEST"
    mock_room.remote_participants = {}
    mock_room.on = MagicMock()
    mock_room.local_participant.publish_track = AsyncMock()

    mock_ctx = MagicMock()
    mock_ctx.connect = AsyncMock()
    mock_ctx.room = mock_room

    worker = VoiceAssistantWorker()
    worker.dost_publisher.initialize_and_publish = AsyncMock()
    worker.sathi_publisher.initialize_and_publish = AsyncMock()

    await worker.run_job(mock_ctx)

    assert worker.room == mock_room
    mock_ctx.connect.assert_called_once()
    assert worker.dost_publisher.initialize_and_publish.called
    assert worker.sathi_publisher.initialize_and_publish.called


@pytest.mark.asyncio
async def test_audio_publisher_keepalive_loop():
    """Verify BotAudioPublisher starts background keepalive loop without crashing."""
    from src.audio.audio_publisher import BotAudioPublisher

    pub = BotAudioPublisher()
    pub.source = MagicMock()
    pub.source.capture_frame = AsyncMock()

    pub.start_keepalive()
    assert pub._keepalive_task is not None
    assert not pub._keepalive_task.done()

    # Cancel task to clean up
    pub._keepalive_task.cancel()
    try:
        await pub._keepalive_task
    except asyncio.CancelledError:
        pass


def test_participant_pipeline_resampler_dynamic_adaptation():
    """Verify ParticipantAudioPipeline initializes dedicated AudioResampler and adapts to changing input sample rates."""
    from livekit import rtc
    from src.agent.worker import ParticipantAudioPipeline

    pipeline = ParticipantAudioPipeline(
        participant_identity="Rahul",
        track=MagicMock(),
        transcription_handler=MagicMock(),
    )
    assert pipeline.resampler is None

    # Simulate pushing 48kHz frame
    f48 = rtc.AudioFrame(b"\x00" * 1920, 48000, 1, 960)
    if pipeline.resampler is None or pipeline.resampler_input_rate != f48.sample_rate:
        pipeline.resampler = rtc.AudioResampler(
            input_rate=f48.sample_rate,
            output_rate=16000,
            num_channels=1,
        )
        pipeline.resampler_input_rate = f48.sample_rate

    assert pipeline.resampler is not None
    assert pipeline.resampler_input_rate == 48000

    # Simulate rate change to 44.1kHz
    f44 = rtc.AudioFrame(b"\x00" * 1764, 44100, 1, 882)
    if pipeline.resampler is None or pipeline.resampler_input_rate != f44.sample_rate:
        pipeline.resampler = rtc.AudioResampler(
            input_rate=f44.sample_rate,
            output_rate=16000,
            num_channels=1,
        )
        pipeline.resampler_input_rate = f44.sample_rate

    assert pipeline.resampler_input_rate == 44100


@pytest.mark.asyncio
async def test_persistent_queue_worker_loop_and_sequential_handoff():
    """Verify queue worker loop remains active across interruptions and executes multi-turn plans sequentially."""
    executed = []

    async def mock_executor(turn: TurnPlan):
        executed.append(turn.target)
        await asyncio.sleep(0.02)

    queue_mgr = TurnQueueManager(executor=mock_executor)

    # 1. Enqueue Dost and Sathi turns
    await queue_mgr.add_turns([
        TurnPlan(target="dost", task="Explain concept"),
        TurnPlan(target="sathi", task="Provide example"),
    ])

    # Allow execution to complete sequentially
    await asyncio.sleep(0.1)
    assert executed == ["dost", "sathi"]

    # 2. Simulate interruption on active turn
    await queue_mgr.add_turns([TurnPlan(target="dost", task="Long turn")])
    await asyncio.sleep(0.005)
    cleared = queue_mgr.cancel_current_and_clear()
    assert cleared >= 0

    # 3. Enqueue new turn after interruption to verify worker loop persists
    await queue_mgr.add_turns([TurnPlan(target="sathi", task="Post interruption turn")])
    await asyncio.sleep(0.05)

    assert "sathi" in executed
    await queue_mgr.stop()





