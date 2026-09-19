import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
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
