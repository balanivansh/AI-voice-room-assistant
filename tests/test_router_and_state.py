import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agent.queue_manager import TurnQueueManager
from src.agent.router import RouterAgent
from src.audio.transcription_handler import TranscriptionHandler, UtteranceNormalizer
from src.state.graph import RoomStateGraphManager
from src.state.room_state import RouterOutput, TurnPlan


@pytest.mark.asyncio
async def test_json_chat_unwrapping():
    """Verify LiveKit chat protocol JSON strings are unwrapped cleanly."""
    handler = TranscriptionHandler()
    json_payload = json.dumps({"message": "Namaste Dost!", "extra": "data"})

    utt = await handler.normalize_and_dispatch(
        identity="Rahul",
        text=json_payload,
        source="text",
    )
    assert utt is not None
    assert utt.text == "Namaste Dost!"
    assert utt.identity == "Rahul"
    assert utt.source == "text"


@pytest.mark.asyncio
async def test_room_state_graph_and_history_capping():
    """Verify RoomState graph preserves facts and caps turn history at 15 items."""
    mock_router = MagicMock(spec=RouterAgent)
    mock_router.route_utterance = AsyncMock(
        return_value=RouterOutput(
            new_speaker_facts=["Prefers Python", "User preference A"],
            salient_entity="programming",
            turns=[TurnPlan(target="dost", task="Reply warmly")],
        )
    )

    manager = RoomStateGraphManager(router_agent=mock_router)

    # Ingest 20 utterances to test history capping
    for i in range(20):
        utt = UtteranceNormalizer(
            identity="Rahul",
            text=f"Utterance number {i}",
            source="voice",
        )
        turns = await manager.process_utterance(utt)
        assert len(turns) == 1
        assert turns[0].target == "dost"

    # Verify history is capped at 15
    assert len(manager.state["last_n_turns"]) == 15
    assert manager.state["last_n_turns"][0]["text"] == "Utterance number 5"
    assert manager.state["last_n_turns"][-1]["text"] == "Utterance number 19"

    # Verify speaker facts persisted
    assert "Rahul" in manager.state["speaker_facts"]
    assert "Prefers Python" in manager.state["speaker_facts"]["Rahul"]
    assert manager.state["active_topic"] == "programming"


@pytest.mark.asyncio
async def test_turn_queue_manager_execution_and_interruption():
    """Verify TurnQueueManager executes sequentially and supports cancellation on interruption."""
    executed_turns = []

    async def mock_executor(turn: TurnPlan):
        executed_turns.append(turn.target)
        await asyncio.sleep(0.05)

    queue_mgr = TurnQueueManager(executor=mock_executor)

    turns = [
        TurnPlan(target="dost", task="Task 1"),
        TurnPlan(target="sathi", task="Task 2"),
        TurnPlan(target="dost", task="Task 3"),
    ]

    await queue_mgr.add_turns(turns)

    # Allow processor to start first turn
    await asyncio.sleep(0.01)

    # Trigger human interruption clearing remaining turns
    cleared = queue_mgr.cancel_current_and_clear()
    assert cleared >= 1

    # Wait for background task to finish
    await asyncio.sleep(0.1)

    # Verify not all turns were executed due to interruption
    assert len(executed_turns) < 3


@pytest.mark.asyncio
async def test_router_agent_compound_turns_mock():
    """Verify RouterAgent correctly parses compound multi-bot turn routing."""
    mock_router = MagicMock(spec=RouterAgent)
    mock_router.route_utterance = AsyncMock(
        return_value=RouterOutput(
            new_speaker_facts=[],
            salient_entity="travel plan",
            turns=[
                TurnPlan(target="dost", task="Give friendly greeting"),
                TurnPlan(target="sathi", task="Provide itinerary details"),
            ],
        )
    )

    manager = RoomStateGraphManager(router_agent=mock_router)
    utt = UtteranceNormalizer(
        identity="Priya",
        text="Dost hi bol, Sathi itinerary batao",
        source="voice",
    )

    plans = await manager.process_utterance(utt)
    assert len(plans) == 2
    assert plans[0].target == "dost"
    assert plans[1].target == "sathi"


@pytest.mark.asyncio
async def test_phonetic_normalizer_prefix_bound():
    """Verify phonetic normalizer only substitutes when prefixed by AI or arre."""
    from src.agent.router import normalize_phonetic_aliases

    # Conjunction "saath hi" should NOT be replaced by "AI Sathi"
    assert normalize_phonetic_aliases("Python ke saath hi SQL seekho") == "Python ke saath hi SQL seekho"
    assert normalize_phonetic_aliases("Mera dost bol raha tha") == "Mera dost bol raha tha"

    # Explicit AI or arre prefix SHOULD be normalized
    assert normalize_phonetic_aliases("AI saath hi explain karo") == "AI Sathi explain karo"
    assert normalize_phonetic_aliases("Arre dosth suno") == "AI Dost suno"


@pytest.mark.asyncio
async def test_isolated_wake_callout_ignored():
    """Verify isolated wake callouts return empty turn plans (turns_count: 0)."""
    router = RouterAgent(api_key="mock_key")
    for text in ["AI Dost", "dost", "sathi", "AI Sathi", "साथी", "दोस्त"]:

        out = await router.route_utterance(
            speaker_identity="Neha",
            text=text,
            speaker_facts={},
            active_topic=None,
            history=[],
        )
        assert len(out.turns) == 0
        assert out.salient_entity is None


@pytest.mark.asyncio
async def test_speaker_facts_accumulative_and_formatting():
    """Verify speaker facts accumulate across turns without loss and format as bulleted lists in PersonaManager."""
    from src.agent.personas import PersonaManager

    mock_router = MagicMock(spec=RouterAgent)
    mock_router.route_utterance = AsyncMock(
        side_effect=[
            RouterOutput(
                new_speaker_facts=["Lives in Delhi"],
                salient_entity="city",
                turns=[TurnPlan(target="dost", task="Acknowledge location")],
            ),
            RouterOutput(
                new_speaker_facts=["Software engineer at TechCorp"],
                salient_entity="job",
                turns=[TurnPlan(target="sathi", task="Acknowledge career")],
            ),
        ]
    )

    manager = RoomStateGraphManager(router_agent=mock_router)

    utt1 = UtteranceNormalizer(identity="Aman", text="Main Delhi mein rehta hoon", source="voice")
    await manager.process_utterance(utt1)
    assert manager.state["speaker_facts"]["Aman"] == ["Lives in Delhi"]

    utt2 = UtteranceNormalizer(identity="Aman", text="Main software engineer hoon", source="voice")
    await manager.process_utterance(utt2)

    # Verify both facts accumulated without loss
    assert manager.state["speaker_facts"]["Aman"] == [
        "Lives in Delhi",
        "Software engineer at TechCorp",
    ]

    # Verify PersonaManager formats facts as bulleted string
    persona_mgr = PersonaManager()
    with patch("src.agent.personas.safe_external_call", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = "Response text"
        reply = await persona_mgr.generate_reply("dost", "Greet user", manager.state)
        assert reply == "Response text"
        call_kwargs = mock_call.call_args[1]
        assert call_kwargs["known_facts"] == "- Lives in Delhi\n- Software engineer at TechCorp"


def test_worker_main_room_injection(monkeypatch):
    """Verify worker main() automatically injects --room LIVEKIT_ROOM into sys.argv when in connect mode or no args."""
    import sys
    from src.agent.worker import main

    # Case 1: connect command without --room
    test_argv = ["worker.py", "connect"]
    monkeypatch.setattr(sys, "argv", test_argv)
    monkeypatch.setenv("LIVEKIT_ROOM", "test-live-room")

    with patch("livekit.agents.cli.run_app") as mock_run_app:
        main()
        assert "--room" in sys.argv
        assert "test-live-room" in sys.argv
        assert mock_run_app.called

    # Case 2: dev command should NOT get --room
    test_argv_dev = ["worker.py", "dev"]
    monkeypatch.setattr(sys, "argv", test_argv_dev)

    with patch("livekit.agents.cli.run_app") as mock_run_app:
        main()
        assert "--room" not in sys.argv


@pytest.mark.asyncio
async def test_dynamic_conciseness_directives():
    """Verify router appends dual-bot, solo, and greeting/acknowledgment conciseness directives based on task type."""
    router = RouterAgent(api_key="mock_key")

    # Case 1: Dual bot response (> 1 turn)
    with patch("src.agent.router.safe_external_call", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = RouterOutput(
            new_speaker_facts=[],
            salient_entity="cloud",
            turns=[
                TurnPlan(target="dost", task="Comment on query"),
                TurnPlan(target="sathi", task="Explanation"),
            ],
        )
        res_dual = await router.route_utterance("Aman", "Tell me about cloud", {}, None, [])
        assert len(res_dual.turns) == 2
        assert "strictly to 1-2 short sentences" in res_dual.turns[0].task
        assert "strictly to 1-2 short sentences" in res_dual.turns[1].task

    # Case 2: Solo bot response (== 1 turn)
    with patch("src.agent.router.safe_external_call", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = RouterOutput(
            new_speaker_facts=[],
            salient_entity="cloud",
            turns=[
                TurnPlan(target="sathi", task="Detailed answer"),
            ],
        )
        res_solo = await router.route_utterance("Aman", "Explain Sathi", {}, None, [])
        assert len(res_solo.turns) == 1
        assert "to 2-3 natural sentences" in res_solo.turns[0].task

    # Case 3: Greeting / Acknowledgment task
    with patch("src.agent.router.safe_external_call", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = RouterOutput(
            new_speaker_facts=["Lives in Delhi"],
            salient_entity=None,
            turns=[
                TurnPlan(target="dost", task="Acknowledge greeting and user location"),
            ],
        )
        res_greet = await router.route_utterance("Aman", "Hi main Delhi se hoon", {}, None, [])
        assert len(res_greet.turns) == 1
        assert "Acknowledge in 1 single short sentence (under 12 words)." in res_greet.turns[0].task



